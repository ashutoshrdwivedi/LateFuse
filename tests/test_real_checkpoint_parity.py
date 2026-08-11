"""Numerical equivalence vs HuggingFace PEFT on the REAL generality-row checkpoints.

tests/test_hf_wrapper.py pins these properties on tiny random models, which is
fast but not sufficient: the DeBERTa shared-position-bias divergence only
appears with *trained* weights (on a randomly initialised model the position
bias is near-noise, and our output matches PEFT to ~1e-7). These tests
therefore run against the actual checkpoints the generality rows measure.

Requires CUDA and multi-GB model downloads (xlm-roberta-xl 14G,
deberta-v2-xlarge 3.4G, electra-large 1.3G), so they are marked
`real_checkpoints`. The XLM-R-XL case holds two fp32 copies of a 3.5B model at
once (ours + PEFT's, ~28 GB) and so needs an 80 GB card. They run by default on
a GPU box; to skip them:

    uv run pytest tests/ -m "not real_checkpoints"

Run with -s to print the measured relative differences.

Everything is fp32 so tolerances reflect the implementations rather than
half-precision accumulation.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModel

from lora_serving.config import LoraServingConfig
from lora_serving.model.hf_wrapper import HFEncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

peft = pytest.importorskip("peft", reason="peft not installed")
from peft import LoraConfig, get_peft_model

pytestmark = [
    pytest.mark.real_checkpoints,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

ELECTRA = "google/electra-large-discriminator"
DEBERTA = "microsoft/deberta-v2-xlarge"
XLMR = "facebook/xlm-roberta-xl"
DEVICE = torch.device("cuda:0")
DTYPE = torch.float32
BATCH_SIZE, SEQ_LEN, RANK = 4, 32, 8

# electra-large is a plain post-LN encoder: the two implementations agree to
# fp32 round-off. deberta-v2-xlarge is 900M params, so its residual chain
# accumulates more, hence the looser bound.
TOL_TIGHT = 1e-4
TOL_DEBERTA = 1e-3
# xlm-roberta-xl is 3.5B over 36 pre-LN layers, but depth costs nothing here:
# measured 1.10e-07 on 2026-07-29, in line with electra-large, so it takes the
# tight bound. Pre-LN keeps the residual chain better conditioned than
# deberta's post-LN stack, which is what needs the looser allowance.
TOL_XLMR = TOL_TIGHT
# The documented shared-position-bias gap, measured at 1.03e-1 on 2026-07-27.
# Bounded rather than pinned so transformers version drift does not cause noise.
MIN_KNOWN_DIVERGENCE = 1e-2


def _cfg(name: str, targets: list[str]) -> LoraServingConfig:
    return LoraServingConfig(
        model_name=name, lora_rank=RANK, batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN, target_modules=targets, device=DEVICE, dtype=DTYPE,
    )


def _store(cfg: LoraServingConfig, ids: list[str], nonzero: bool = True) -> AdapterStore:
    """load_synthetic zeroes B (delta=0); optionally randomise it so the delta
    path is actually exercised."""
    store = AdapterStore(cfg)
    for i, aid in enumerate(ids):
        store.load_synthetic(aid, seed=100 + i)
        if nonzero:
            w = store.get(aid)
            torch.manual_seed(200 + i)
            for m in cfg.target_modules:
                torch.nn.init.normal_(w.wb[m], std=0.02)
    return store


def _peft_with(name, cfg, store, aid, peft_targets):
    """PEFT model carrying the same A/B matrices as `aid`.

    Ours: delta = x @ wa @ wb, wa (H,R), wb (R,H).
    PEFT: delta = lora_B(lora_A(x)) * (alpha/r) with lora_A.weight (R,H) and
    lora_B.weight (H,R) — so transpose both, and alpha=r makes the scale 1.
    """
    base = AutoModel.from_pretrained(name, dtype=DTYPE)
    lcfg = LoraConfig(r=RANK, lora_alpha=RANK, target_modules=peft_targets,
                      lora_dropout=0.0, bias="none")
    model = get_peft_model(base, lcfg, adapter_name="x")
    w, mods = store.get(aid), dict(model.named_modules())
    for logical, pname in zip(cfg.target_modules, peft_targets):
        for layer in range(cfg.num_layers):
            mod = mods[f"base_model.model.encoder.layer.{layer}.attention.self.{pname}"]
            with torch.no_grad():
                mod.lora_A["x"].weight.copy_(w.wa[logical][layer].T)
                mod.lora_B["x"].weight.copy_(w.wb[logical][layer].T)
    return model.to(DEVICE).eval()


def _rel(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    return float((ours - theirs).abs().max() / theirs.abs().max())


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    ids = torch.randint(1, 1000, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
    return ids, mask


# --------------------------------------------------------------------------
# ELECTRA-large — hook wrapper vs PEFT
#
# This row moved from the custom encoder to HFEncoderWithLora: ELECTRA
# discriminators ship no pooler, and EncoderWithLora declares one, so its
# strict loader rejects the checkpoint rather than running CLS through a
# randomly initialised projection. Parity is taken on the CLS hidden state,
# which is what encode_pooled returns when the base has no pooler.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def electra():
    cfg = _cfg(ELECTRA, ["query", "value"])
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    wrapper.eval()
    store = _store(cfg, [f"a{i}" for i in range(BATCH_SIZE)])
    yield cfg, wrapper, store
    del wrapper, store
    torch.cuda.empty_cache()


def test_electra_zero_delta_matches_bare_hf(electra, batch):
    """With B=0 the LoRA path must vanish, reproducing the stock HF model."""
    cfg, wrapper, _ = electra
    ids, mask = batch
    zstore = _store(cfg, ["z"], nonzero=False)
    lora_w = BatchAssembler(zstore, cfg).assemble_lora(["z"] * BATCH_SIZE)
    hf = AutoModel.from_pretrained(ELECTRA, dtype=DTYPE).to(DEVICE).eval()
    with torch.no_grad():
        ours = wrapper.encode_pooled(ids, mask, lora_w)
        theirs = hf(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
    del hf
    torch.cuda.empty_cache()
    rel = _rel(ours, theirs)
    print(f"\n  electra zero-delta vs bare HF: rel={rel:.3e}")
    assert rel < TOL_TIGHT


def test_electra_shared_adapter_matches_peft(electra, batch):
    """One adapter replicated across the batch vs PEFT's batched forward."""
    cfg, wrapper, store = electra
    ids, mask = batch
    lora_w = BatchAssembler(store, cfg).assemble_lora(["a0"] * BATCH_SIZE)
    pm = _peft_with(ELECTRA, cfg, store, "a0", ["query", "value"])
    with torch.no_grad():
        ours = wrapper.encode_pooled(ids, mask, lora_w)
        theirs = pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        base = wrapper.encode_pooled(ids, mask, lora_w, apply_lora=False)
    del pm
    torch.cuda.empty_cache()
    rel = _rel(ours, theirs)
    delta = float((ours - base).abs().max() / base.abs().max())
    skips = wrapper.shared_projection_skips
    print(f"\n  electra shared adapter vs PEFT: rel={rel:.3e}, skips={skips}, "
          f"|lora delta|={delta:.3e}")
    # ELECTRA keeps BERT's attention naming and has no batch-shared projection,
    # so it takes the "default" suffix map and nothing may be skipped.
    assert skips == 0
    assert delta > 10 * rel, (
        f"LoRA delta {delta:.2e} is not clearly above the parity residual "
        f"{rel:.2e}; the metric cannot resolve a real difference"
    )
    assert rel < TOL_TIGHT


def test_electra_mixed_tenant_matches_peft_loop(electra, batch):
    """The multi-tenant claim: one mixed batch must equal a per-sample PEFT loop."""
    cfg, wrapper, store = electra
    ids, mask = batch
    aids = [f"a{i}" for i in range(BATCH_SIZE)]
    lora_w = BatchAssembler(store, cfg).assemble_lora(aids)
    with torch.no_grad():
        ours = wrapper.encode_pooled(ids, mask, lora_w)
        rows = []
        for i, aid in enumerate(aids):
            pm = _peft_with(ELECTRA, cfg, store, aid, ["query", "value"])
            rows.append(pm(input_ids=ids[i:i + 1],
                           attention_mask=mask[i:i + 1]).last_hidden_state[:, 0])
            del pm
            torch.cuda.empty_cache()
        theirs = torch.cat(rows, 0)
    rel = _rel(ours, theirs)
    print(f"\n  electra mixed-tenant vs per-sample PEFT loop: rel={rel:.3e}")
    assert rel < TOL_TIGHT


# --------------------------------------------------------------------------
# DeBERTa-v2-xlarge — hook wrapper vs PEFT, share_att_key=True
# --------------------------------------------------------------------------

def _deberta_vs_peft(targets, peft_targets, batch):
    ids, mask = batch
    cfg = _cfg(DEBERTA, targets)
    store = _store(cfg, ["a0"])
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    wrapper.eval()
    lora_w = BatchAssembler(store, cfg).assemble_lora(["a0"] * BATCH_SIZE)
    pm = _peft_with(DEBERTA, cfg, store, "a0", peft_targets)
    with torch.no_grad():
        ours = wrapper.encode_pooled(ids, mask, lora_w)
        theirs = pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        base = wrapper.encode_pooled(ids, mask, lora_w, apply_lora=False)
    skips, n_layers = wrapper.shared_projection_skips, cfg.num_layers
    del wrapper, pm, store
    torch.cuda.empty_cache()
    return _rel(ours, theirs), float((ours - base).abs().max()), skips, n_layers


def test_deberta_value_only_matches_peft(batch):
    """The configuration the DeBERTa generality row uses.

    value_proj is never applied to the relative-position embeddings, so there
    is no batch-shared call to skip and the comparison against PEFT is exact.
    """
    rel, _, skips, _ = _deberta_vs_peft(["value"], ["value_proj"], batch)
    print(f"\n  deberta value-only vs PEFT: rel={rel:.3e}, skips={skips}")
    assert skips == 0
    assert rel < TOL_DEBERTA


def test_deberta_query_target_diverges_from_peft(batch):
    """Pins the documented limitation that motivates targeting value only.

    Under share_att_key=True, query_proj is also applied to the batch-shared
    relative-position embeddings and the result is tiled across the batch
    (`.repeat(...)` in DisentangledSelfAttention.disentangled_attention_bias).
    PEFT wraps the Linear and so adapts that call; per-tenant serving has no
    per-sample counterpart for a batch-shared tensor, so we skip it and the
    outputs genuinely differ.

    If this ever starts matching, the shared-projection handling changed and
    the DeBERTa row could move back to query+value.
    """
    rel, delta, skips, n_layers = _deberta_vs_peft(
        ["query", "value"], ["query_proj", "value_proj"], batch
    )
    print(f"\n  deberta query+value vs PEFT: rel={rel:.3e} ({100 * rel:.1f}%), "
          f"skips={skips}/{n_layers}, |lora delta|={delta:.3e}")
    assert skips == n_layers, "expected one skipped shared-projection call per layer"
    assert rel > MIN_KNOWN_DIVERGENCE, (
        f"expected the un-adapted position bias to show up, got rel={rel:.2e}"
    )


# --------------------------------------------------------------------------
# XLM-RoBERTa-XL — hook wrapper vs PEFT, the largest checkpoint in the ladder
# --------------------------------------------------------------------------

def test_xlmr_xl_query_value_matches_peft(batch):
    """Parity for the XLM-R-XL generality row (query+value, the row's config).

    This is the only row whose parity used to be argued from "it shares the
    hook path" rather than measured, so Q4's unit-tested claim rested on an
    inference for the largest model in the ladder. It is also the strongest
    test of the claim: at 3.5B and 36 pre-LN layers the residual chain
    accumulates far more than anything else here, so agreement is not a
    foregone conclusion.

    XLM-R keeps BERT's attention naming and has no batch-shared projection, so
    it takes the "default" suffix map and skips must be 0 -- unlike DeBERTa-v2,
    the comparison against PEFT is exact rather than deliberately divergent.

    Parity is taken on the CLS hidden state, not the pooled output, because
    this checkpoint ships *no pooler weights*: HF reports `pooler.dense.weight`
    and `pooler.dense.bias` as "newly initialized", and it initialises them
    independently for each model instance. Comparing pooler_output therefore
    compares two different random projections and shows a ~124% difference that
    has nothing to do with LoRA. Dropping the pooler makes encode_pooled fall
    back to CLS, which is also what the ELECTRA and DeBERTa cases above use.

    The test additionally requires the LoRA delta to dominate the parity
    residual, which pins the metric's resolution rather than assuming it. That
    guard is what caught the pooler problem.
    """
    ids, mask = batch
    cfg = _cfg(XLMR, ["query", "value"])
    store = _store(cfg, ["a0"])
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    # Untrained pooler (see docstring); None makes encode_pooled return CLS.
    wrapper.base.pooler = None
    wrapper.eval()
    lora_w = BatchAssembler(store, cfg).assemble_lora(["a0"] * BATCH_SIZE)
    pm = _peft_with(XLMR, cfg, store, "a0", ["query", "value"])
    with torch.no_grad():
        ours = wrapper.encode_pooled(ids, mask, lora_w)
        theirs = pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        base = wrapper.encode_pooled(ids, mask, lora_w, apply_lora=False)
    rel = _rel(ours, theirs)
    delta = float((ours - base).abs().max() / base.abs().max())
    skips = wrapper.shared_projection_skips
    del wrapper, pm, store, lora_w
    torch.cuda.empty_cache()

    print(f"\n  xlm-r-xl query+value vs PEFT: rel={rel:.3e}, skips={skips}, "
          f"|lora delta|={delta:.3e}")
    assert skips == 0, "XLM-R has no batch-shared projection; nothing should be skipped"
    assert delta > 10 * rel, (
        f"LoRA delta {delta:.2e} is not clearly above the parity residual "
        f"{rel:.2e}; the metric cannot resolve a real difference"
    )
    assert rel < TOL_XLMR
