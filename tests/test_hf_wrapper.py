"""Numerical validation of HFEncoderWithLora (hook-injected LoRA on stock HF models).

Uses tiny randomly-initialised checkpoints written to tmp_path, so every test
runs offline on CPU (and on GPU when available). Coverage:

  1. PEFT parity, BERT: hook-injected deltas match `peft.get_peft_model` on the
     same base weights and the same A/B matrices.
  2. PEFT parity, DeBERTa-v2 with relative attention + conv: same check on the
     disentangled-attention architecture (share_att_key=False so PEFT and the
     hooks target exactly the same content projections).
  3. Per-sample isolation, DeBERTa-v2 with share_att_key=True (the
     deberta-v2-xlarge configuration): zero-delta samples in a mixed batch
     reproduce the base forward exactly; nonzero samples diverge. Also
     exercises the hook guard that skips the shared relative-position
     projection calls.
  4. Pooler-less checkpoints (ELECTRA): PEFT parity on the wrapper, which is
     the engine that row runs on, plus the guard that the custom encoder's
     strict loader rejects them rather than serving CLS through a randomly
     initialised pooler.
  5. Cross-engine equivalence: on a checkpoint both engines can load, the hook
     path and the specialized encoder produce the same pooled output — the
     property that lets a wrapper-measured row be compared with the paper's.
  6. share_att_key accounting: the wrapper skips exactly the batch-shared
     projection calls it should, and no others.
  7. PEFT parity, XLM-RoBERTa-XL: the pre-LN architecture the generality row
     measures, and the all-hooks-fired guard that keeps a forward which
     applied no delta from being reported as a timing.
  8. Lifetime. The hooks must not make the wrapper part of a reference cycle,
     because the sweep drops one model per (N, B, r) cell and needs its VRAM
     back before the next cell allocates. Hook registration is tied to the
     wrapper's lifetime from both ends: a dropped wrapper detaches its hooks
     from a base that outlives it, a failed __init__ unwinds the ones it had
     already attached, and close()/`with` release the base at a known point.
  9. Construction guards: duplicate target_modules are rejected where the
     field is named, not as a shape assertion deep in the delta path.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from transformers import (
    AutoModel,
    BertConfig,
    BertModel,
    DebertaV2Config,
    DebertaV2Model,
    ElectraConfig,
    ElectraModel,
    XLMRobertaXLConfig,
    XLMRobertaXLModel,
)

from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.model.hf_wrapper import HFEncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

peft = pytest.importorskip("peft", reason="peft not installed")
from peft import LoraConfig, get_peft_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LORA_RANK = 4
BATCH_SIZE = 4
SEQ_LEN = 16
ATOL = 1e-4
RTOL = 1e-4

TINY = dict(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
    pad_token_id=0,
)


def _save_tiny(model_cls, config, tmp_path) -> str:
    torch.manual_seed(0)
    model = model_cls(config)
    model.eval()
    path = str(tmp_path / "model")
    model.save_pretrained(path)
    return path


def _serving_config(model_path: str, target_modules: list[str]) -> LoraServingConfig:
    return LoraServingConfig(
        model_name=model_path,
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        target_modules=target_modules,
        device=DEVICE,
        dtype=torch.float32,
    )


def _store_with_adapters(cfg: LoraServingConfig, n: int, nonzero_b: bool = True):
    """AdapterStore with n random adapters. load_synthetic zeroes B; optionally
    re-randomise it so the delta path is actually exercised."""
    store = AdapterStore(cfg)
    for i in range(n):
        store.load_synthetic(f"adapter_{i}", seed=100 + i)
        if nonzero_b:
            w = store.get(f"adapter_{i}")
            torch.manual_seed(200 + i)
            for m in cfg.target_modules:
                torch.nn.init.normal_(w.wb[m], std=0.02)
    return store


def _inputs(cfg: LoraServingConfig):
    torch.manual_seed(7)
    input_ids = torch.randint(1, TINY["vocab_size"], (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
    return input_ids, attention_mask


def _peft_with_our_weights(model_path: str, cfg: LoraServingConfig, store, adapter_id: str,
                           peft_targets: list[str], layer_prefix: str):
    """PEFT-wrapped base model carrying the same A/B matrices as `adapter_id`.

    Our convention: delta = x @ wa @ wb with wa (H, R), wb (R, H).
    PEFT: delta = lora_B(lora_A(x)) * (alpha/r), lora_A.weight (r, H),
    lora_B.weight (H, r) — so wa = lora_A.weight.T and wb = lora_B.weight.T,
    with alpha = r making the scale 1.
    """
    base = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    lcfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK, target_modules=peft_targets,
        lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(base, lcfg, adapter_name="a")
    weight = store.get(adapter_id)
    sd = dict(model.named_modules())
    for logical, peft_name in zip(cfg.target_modules, peft_targets):
        for layer in range(cfg.num_layers):
            mod = sd[f"base_model.model.encoder.layer.{layer}.{layer_prefix}.{peft_name}"]
            with torch.no_grad():
                mod.lora_A["a"].weight.copy_(weight.wa[logical][layer].T)
                mod.lora_B["a"].weight.copy_(weight.wb[logical][layer].T)
    model.to(DEVICE)
    model.eval()
    return model


def _assert_parity(model_path: str, cfg: LoraServingConfig, peft_targets: list[str],
                   use_pooler_output: bool):
    """Shared-adapter batch: wrapper output must match PEFT batched forward."""
    store = _store_with_adapters(cfg, 1)
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    assembler = BatchAssembler(store, cfg)
    lora_w = assembler.assemble_lora(["adapter_0"] * BATCH_SIZE)
    input_ids, attention_mask = _inputs(cfg)

    with torch.no_grad():
        ours = wrapper.encode_pooled(input_ids, attention_mask, lora_w)

    peft_model = _peft_with_our_weights(
        model_path, cfg, store, "adapter_0", peft_targets, "attention.self"
    )
    with torch.no_grad():
        out = peft_model(input_ids=input_ids, attention_mask=attention_mask)
    theirs = out.pooler_output if use_pooler_output else out.last_hidden_state[:, 0]

    torch.testing.assert_close(ours, theirs, atol=ATOL, rtol=RTOL)


def test_parity_bert(tmp_path):
    config = BertConfig(**TINY, type_vocab_size=2)
    path = _save_tiny(BertModel, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    _assert_parity(path, cfg, ["query", "value"], use_pooler_output=True)


def test_parity_deberta_v2_disentangled(tmp_path):
    config = DebertaV2Config(
        **TINY,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=4,
        pos_att_type=["p2c", "c2p"],
        share_att_key=False,
        conv_kernel_size=3,
    )
    path = _save_tiny(DebertaV2Model, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    _assert_parity(path, cfg, ["query_proj", "value_proj"], use_pooler_output=False)


def test_isolation_deberta_v2_shared_att_key(tmp_path):
    """deberta-v2-xlarge-shaped config: mixed batch where only some samples
    carry a nonzero delta. Zero-delta rows must equal the base forward."""
    config = DebertaV2Config(
        **TINY,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=4,
        pos_att_type=["p2c", "c2p"],
        share_att_key=True,
        conv_kernel_size=3,
    )
    path = _save_tiny(DebertaV2Model, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])

    store = AdapterStore(cfg)
    store.load_synthetic("zero", seed=1)  # B stays zero → delta = 0
    store.load_synthetic("live", seed=2)
    live = store.get("live")
    torch.manual_seed(3)
    for m in cfg.target_modules:
        torch.nn.init.normal_(live.wb[m], std=0.02)

    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    assembler = BatchAssembler(store, cfg)
    lora_w = assembler.assemble_lora(["live", "zero", "live", "zero"])
    input_ids, attention_mask = _inputs(cfg)

    with torch.no_grad():
        mixed = wrapper.encode_pooled(input_ids, attention_mask, lora_w)
        base = wrapper.encode_pooled(input_ids, attention_mask, lora_w, apply_lora=False)

    # zero-delta samples: exact base output (isolation by construction)
    torch.testing.assert_close(mixed[[1, 3]], base[[1, 3]], atol=ATOL, rtol=RTOL)
    # live samples must actually diverge from base
    assert not torch.allclose(mixed[[0, 2]], base[[0, 2]], atol=ATOL, rtol=RTOL)


def test_parity_electra_vs_peft(tmp_path):
    """ELECTRA on the hook wrapper must match PEFT.

    The ELECTRA generality row runs on HFEncoderWithLora (the checkpoint ships
    no pooler, which the custom encoder's strict loader rejects — see
    test_electra_rejected_by_custom_encoder). Pooled output is the raw CLS
    state on both sides. Mirrors the real-checkpoint check in
    tests/test_real_checkpoint_parity.py.
    """
    config = ElectraConfig(**TINY, type_vocab_size=2, embedding_size=TINY["hidden_size"])
    path = _save_tiny(ElectraModel, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    _assert_parity(path, cfg, ["query", "value"], use_pooler_output=False)


def _deberta_shared_key_path(tmp_path):
    """A deberta-v2-xlarge-shaped config: relative attention with share_att_key."""
    config = DebertaV2Config(
        **TINY, type_vocab_size=0, relative_attention=True, position_buckets=4,
        pos_att_type=["p2c", "c2p"], share_att_key=True, conv_kernel_size=3,
    )
    return _save_tiny(DebertaV2Model, config, tmp_path)


def _deberta_vs_peft(path, targets, peft_targets):
    cfg = _serving_config(path, targets)
    store = _store_with_adapters(cfg, 1)
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    lora_w = BatchAssembler(store, cfg).assemble_lora(["adapter_0"] * BATCH_SIZE)
    input_ids, attention_mask = _inputs(cfg)
    with torch.no_grad():
        ours = wrapper.encode_pooled(input_ids, attention_mask, lora_w)
    peft_model = _peft_with_our_weights(
        path, cfg, store, "adapter_0", peft_targets, "attention.self"
    )
    with torch.no_grad():
        theirs = peft_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
    rel = ((ours - theirs).abs().max() / theirs.abs().max()).item()
    return rel, wrapper.shared_projection_skips, cfg.num_layers


def test_deberta_value_only_matches_peft(tmp_path):
    """value_proj is never used for the relative-position bias, so targeting it
    alone is exactly equivalent to PEFT — the configuration the DeBERTa
    generality row uses."""
    rel, skips, _ = _deberta_vs_peft(
        _deberta_shared_key_path(tmp_path), ["value"], ["value_proj"]
    )
    assert rel < 1e-4, f"value-only should match PEFT, got rel={rel:.2e}"
    assert skips == 0


def test_deberta_query_target_still_adapts_the_content_path(tmp_path):
    """Targeting query under share_att_key=True adapts the content projection
    and skips only the batch-shared position one.

    Deliberately NOT asserted here: how far our output then sits from PEFT's.
    PEFT wraps the Linear, so it also adapts the relative-position call that we
    skip. That gap does not reproduce on a randomly initialised model — the
    position bias only carries a strong signal once trained — so pinning a
    magnitude here would pass for the wrong reason. It is asserted against the
    real checkpoint in tests/test_real_checkpoint_parity.py, which measures
    ~10% relative and is why the DeBERTa generality row targets value only.
    """
    path = _deberta_shared_key_path(tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    store = _store_with_adapters(cfg, 1)
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    lora_w = BatchAssembler(store, cfg).assemble_lora(["adapter_0"] * BATCH_SIZE)
    input_ids, attention_mask = _inputs(cfg)
    with torch.no_grad():
        adapted = wrapper.encode_pooled(input_ids, attention_mask, lora_w)
        base = wrapper.encode_pooled(input_ids, attention_mask, lora_w, apply_lora=False)

    assert not torch.allclose(adapted, base, atol=ATOL, rtol=RTOL)
    assert wrapper.shared_projection_skips == cfg.num_layers


def test_cross_engine_equivalence_bert(tmp_path):
    """The hook wrapper and the paper's specialized encoder must agree.

    Both engines can load a BERT-architecture checkpoint, so this pins the
    claim that a wrapper-measured generality row reflects the same computation
    as the paper's system — only the base forward's provenance differs.
    """
    config = BertConfig(**TINY, type_vocab_size=2)
    path = _save_tiny(BertModel, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    store = _store_with_adapters(cfg, BATCH_SIZE)
    ids = [f"adapter_{i}" for i in range(BATCH_SIZE)]
    lora_w = BatchAssembler(store, cfg).assemble_lora(ids)
    input_ids, attention_mask = _inputs(cfg)

    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    custom = EncoderWithLora.from_pretrained_serving(cfg)
    custom.eval()

    with torch.no_grad():
        theirs = wrapper.encode_pooled(input_ids, attention_mask, lora_w)
        # Mirror EncoderWithLora.forward up to the pooled CLS output.
        B, S = input_ids.shape
        tt = torch.zeros(B, S, dtype=torch.long, device=DEVICE)
        x = (
            custom.embeddings["word_embeddings"](input_ids)
            + custom.embeddings["position_embeddings"](custom.position_ids(input_ids))
            + custom.embeddings["token_type_embeddings"](tt)
        )
        x = custom.embeddings["LayerNorm"](x)
        for layer, lw in zip(custom.encoder["layer"], lora_w):
            x = layer(x, attention_mask, lw)
        ours = custom.pooler_act(custom.pooler["dense"](x[:, 0]))

    torch.testing.assert_close(ours, theirs, atol=ATOL, rtol=RTOL)
    # A BERT checkpoint has no batch-shared projections to skip.
    assert wrapper.shared_projection_skips == 0


def test_shared_projection_skip_accounting(tmp_path):
    """With share_att_key=True, query_proj/key_proj are also called on the
    relative-position embeddings. Targeting query+value must skip exactly one
    such call per layer (query only — value_proj is never used for position
    bias), and targeting value alone must skip none."""
    kwargs = dict(
        **TINY, type_vocab_size=0, relative_attention=True, position_buckets=4,
        pos_att_type=["p2c", "c2p"], share_att_key=True, conv_kernel_size=3,
    )
    path = _save_tiny(DebertaV2Model, DebertaV2Config(**kwargs), tmp_path)
    input_ids, attention_mask = _inputs(_serving_config(path, ["value"]))

    for targets, expected_per_layer in [(["query", "value"], 1), (["value"], 0)]:
        cfg = _serving_config(path, targets)
        store = _store_with_adapters(cfg, 1)
        lora_w = BatchAssembler(store, cfg).assemble_lora(["adapter_0"] * BATCH_SIZE)
        wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
        with torch.no_grad():
            wrapper.encode_pooled(input_ids, attention_mask, lora_w)
        assert wrapper.shared_projection_skips == expected_per_layer * cfg.num_layers, (
            f"targets={targets}: expected "
            f"{expected_per_layer * cfg.num_layers} skips, got "
            f"{wrapper.shared_projection_skips}"
        )


def test_unfired_hooks_raise(tmp_path):
    """A forward that applied no delta must fail, not return a base-only result.

    The hooks only fire on an input of shape (B, S, hidden_size); an
    architecture whose target Linear sees anything else — flattened (B*S, H)
    hidden states, packed sequences, in_features != hidden_size — takes the
    skip path everywhere and the forward completes with zero LoRA math. The
    output is a well-formed (B, H) tensor, so the benchmark would happily
    record latency for a model serving no adapter.

    Corrupting hidden_size after construction is the surrogate for such an
    architecture: it is the same mechanism (expected shape never matches) with
    no need for a checkpoint we do not model.
    """
    config = BertConfig(**TINY, type_vocab_size=2)
    path = _save_tiny(BertModel, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    store = _store_with_adapters(cfg, 1)
    lora_w = BatchAssembler(store, cfg).assemble_lora(["adapter_0"] * BATCH_SIZE)
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    input_ids, attention_mask = _inputs(cfg)

    wrapper.config.hidden_size += 1
    with torch.no_grad(), pytest.raises(RuntimeError, match=r"fired 0/4"):
        wrapper.encode_pooled(input_ids, attention_mask, lora_w)

    # The base-only path applies no delta by design, so it must stay allowed.
    with torch.no_grad():
        wrapper.encode_pooled(input_ids, attention_mask, lora_w, apply_lora=False)


def test_parity_xlm_roberta_xl(tmp_path):
    """PEFT parity on the pre-LN XLM-R-XL architecture.

    xlm-roberta-xl is the checkpoint the generality row actually measures
    through this wrapper, and it is the one family here that is not post-LN.
    tests/test_real_checkpoint_parity.py pins it on the real 3.5B checkpoint,
    but that file needs CUDA and a 14 GB download; this runs the same claim in
    CI. It also exercises the all-hooks-fired guard on the architecture that
    matters most, since a shape mismatch there is exactly what would go
    unnoticed in a latency sweep.
    """
    config = XLMRobertaXLConfig(**TINY, type_vocab_size=2)
    path = _save_tiny(XLMRobertaXLModel, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    _assert_parity(path, cfg, ["query", "value"], use_pooler_output=True)


def test_electra_rejected_by_custom_encoder(tmp_path):
    """ELECTRA checkpoints have no pooler, so the custom encoder must refuse them.

    Loading anyway would leave EncoderWithLora's pooler at its random
    initialisation and run CLS through it, while HFEncoderWithLora returns the
    raw CLS state for the same checkpoint — two engines the benchmark presents
    as interchangeable silently disagreeing. Failing here is what routes
    pooler-less checkpoints to --engine hf, and the message has to say so.
    """
    config = ElectraConfig(**TINY, type_vocab_size=2, embedding_size=TINY["hidden_size"])
    path = _save_tiny(ElectraModel, config, tmp_path)
    cfg = LoraServingConfig(
        model_name=path,
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        target_modules=["query", "value"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match="pooler") as exc:
        EncoderWithLora.from_pretrained_serving(cfg)
    assert "--engine hf" in str(exc.value)


def test_wrapper_is_refcount_collectable(tmp_path):
    """Dropping the wrapper must free it and its base without a GC pass.

    Each hook closure is reachable from the base model's _forward_hooks, and
    the base is a submodule of the wrapper, so a closure that captured `self`
    would put the wrapper in a reference cycle. It would still be collected
    eventually, which is why this never showed up as a leak in a test — but
    only on a gen-2 pass, long after benchmark/run.py's per-cell model went out
    of scope. The sweep builds a fresh wrapper for every (N, B, r) cell and the
    large-N cells sit close to the card's ceiling, so a stale base model still
    holding VRAM turns into an OOM that the sweep records as a dropped cell.

    gc is disabled here so the assertion is specifically about refcounting;
    with the cycle present the wrapper survives the `del` and this fails.
    """
    path = _save_tiny(BertModel, BertConfig(**TINY, type_vocab_size=2), tmp_path)
    model = HFEncoderWithLora.from_pretrained_serving(_serving_config(path, ["query", "value"]))
    assert model._num_hooks > 0, "no hooks registered; the test would pass vacuously"
    wrapper_ref, base_ref = weakref.ref(model), weakref.ref(model.base)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del model
        assert wrapper_ref() is None, (
            "wrapper survived `del` with gc off — a hook closure is holding a "
            "strong reference to it (use weakref in _make_hook)"
        )
        assert base_ref() is None, "base model outlived the wrapper"
    finally:
        # Restore what we found rather than assuming it was on: this test's
        # own premise is that gc state is load-bearing, so it must not decide
        # that state for whatever runs next.
        if was_enabled:
            gc.enable()


def test_close_releases_base_weights_and_forward_then_fails_loudly(tmp_path):
    """Leaving the `with` block must free the base model, not merely unhook it.

    `with` does not unbind its target, so `model` is still a live name here.
    That is the whole trap: a close() that only removed the forward hooks would
    leave every parameter allocated and the next multi-GB load would still OOM,
    while looking like it had cleaned up. Assert on the weights, via a weakref
    that can only die if the submodule was actually dropped.
    """
    path = _save_tiny(BertModel, BertConfig(**TINY, type_vocab_size=2), tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    store = _store_with_adapters(cfg, 1)
    lora_w = BatchAssembler(store, cfg).assemble_lora(["adapter_0"] * BATCH_SIZE)
    ids, mask = _inputs(cfg)

    with HFEncoderWithLora.from_pretrained_serving(cfg) as model:
        model.eval()
        with torch.no_grad():
            model.encode_pooled(ids, mask, lora_w)
        base_ref = weakref.ref(model.base)
        weight_ref = weakref.ref(model.base.embeddings.word_embeddings.weight)
        # Track our own hooks by identity. Asserting "this module has no
        # forward hooks at all" would also fail on a hook transformers or
        # accelerate attached (device_map/offload paths do), reporting a
        # close() bug that isn't one.
        ours = [(h.hooks_dict_ref(), h.id) for h in model._hook_handles]
        assert ours, "no hooks attached; the detach assertion would be vacuous"

    gc.collect()
    assert base_ref() is None, "close() left the base model allocated"
    assert weight_ref() is None, "close() left the base weights allocated"
    assert not model._hook_handles
    assert not any(d is not None and hid in d for d, hid in ours), (
        "close() left LoRA hooks registered"
    )

    model.close()  # idempotent
    with pytest.raises(RuntimeError, match="has been closed"):
        with torch.no_grad():
            model.encode_pooled(ids, mask, lora_w)


def test_failed_init_strands_no_hooks(tmp_path):
    """A __init__ that raises partway must unwind the hooks it already attached.

    The half-built wrapper holds the only handles for them and is discarded
    with the exception, so anything left on `base` is unreachable forever —
    and this is the path a caller is most likely to retry on, having just been
    told which target name was wrong.
    """
    path = _save_tiny(BertModel, BertConfig(**TINY, type_vocab_size=2), tmp_path)
    base = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    cfg = _serving_config(path, ["query", "value"])
    cfg.num_layers += 1  # one more layer than the checkpoint has

    with pytest.raises(ValueError, match="not found"):
        HFEncoderWithLora(base, cfg)

    gc.collect()
    left = sum(len(m._forward_hooks) for m in base.modules())
    assert left == 0, f"{left} hooks stranded on base by the failed __init__"

    # base is still reusable: a corrected config attaches and runs cleanly.
    cfg.num_layers -= 1
    model = HFEncoderWithLora(base, cfg)
    assert sum(len(m._forward_hooks) for m in base.modules()) == model._num_hooks


def test_duplicate_target_modules_rejected(tmp_path):
    """Two hooks on one Linear would otherwise fail as an opaque shape
    assertion inside LoraOps, naming neither the duplicate nor the config
    field. run.py's --target-modules takes nargs="+", so it is one typo away."""
    path = _save_tiny(BertModel, BertConfig(**TINY, type_vocab_size=2), tmp_path)
    base = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    cfg = _serving_config(path, ["query", "query"])

    with pytest.raises(ValueError, match="duplicate"):
        HFEncoderWithLora(base, cfg)
    assert sum(len(m._forward_hooks) for m in base.modules()) == 0


def test_dropped_wrapper_detaches_hooks_from_shared_base(tmp_path):
    """Hook lifetime is tied to the wrapper, not to the base.

    Callers who pass their own `base` can outlive the wrapper with it. Without
    the finalizer the dead closures stay in _forward_hooks and pile up with
    every wrapper built over that base, each one running and returning None on
    every forward thereafter.
    """
    path = _save_tiny(BertModel, BertConfig(**TINY, type_vocab_size=2), tmp_path)
    base = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    cfg = _serving_config(path, ["query", "value"])
    count = lambda: sum(len(m._forward_hooks) for m in base.modules())

    for _ in range(3):
        model = HFEncoderWithLora(base, cfg)
        assert count() == model._num_hooks
        del model
        gc.collect()
        assert count() == 0, "dropped wrapper left its hooks on the shared base"
