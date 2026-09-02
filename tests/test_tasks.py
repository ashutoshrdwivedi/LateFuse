"""The task-level wrappers: what a `pip install latefuse` user actually calls.

Everything below runs offline on CPU against a tiny randomly-initialised BERT,
following the fixtures in tests/test_hf_wrapper.py. Coverage:

  1. Arbitrary-length input. The engine's LoRA buffers are pre-allocated to
     exactly (batch_size, max_seq_len) and assert on anything else, so before
     these wrappers existed `embed(["one text"])` raised. Requests of any
     length must chunk, pad, and come back with one row per input.
  2. Padding must not leak. A short call has to produce bit-identical rows to
     the same texts embedded as part of a full batch — otherwise the padding
     repeated to fill the batch is influencing real results.
  3. Tenant isolation at the API surface. The paper's central claim, re-checked
     one level up: a mixed batch must give each tenant what a batch of their
     own would have.
  4. Ragged label counts. Tenants disagree on how many classes they have; the
     zero-padded lanes must never be read as classes.
  5. Assembler and engine equivalence. Swapping the CPU assembler for the
     GPU-resident one, or the hook engine for the BERT-family one, must not
     move the answer.
  6. PEFT round-trip. An adapter saved by `peft.save_pretrained` loads through
     `load_peft_adapter` and reproduces PEFT's own output.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import BertConfig, BertModel

from latefuse import (
    HeadStore,
    LoraServingConfig,
    MultiTenantClassifier,
    MultiTenantEmbedder,
    MultiTenantReranker,
    load_peft_adapter,
    peft_key_fn,
)
from latefuse.tasks.batching import chunk_and_pad
from latefuse.weights.store import AdapterStore

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANK = 4
BATCH = 8
SEQ = 16

TINY = dict(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
    pad_token_id=0,
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> str:
    """A tiny BERT checkpoint plus tokenizer files, saved once for the module."""
    path = tmp_path_factory.mktemp("tiny") / "model"
    torch.manual_seed(0)
    BertModel(BertConfig(**TINY)).eval().save_pretrained(str(path))
    # A minimal WordPiece vocab: the tokenizer only has to be deterministic and
    # produce ids inside TINY["vocab_size"], not to be linguistically sensible.
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        f"tok{i}" for i in range(TINY["vocab_size"] - 5)
    ]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 512, "tokenizer_class": "BertTokenizer"})
    )
    return str(path)


def _texts(n: int) -> list[str]:
    return [f"tok{i % 40} tok{(i * 7) % 40}" for i in range(n)]


def _embedder(tiny_model: str, **kwargs) -> MultiTenantEmbedder:
    emb = MultiTenantEmbedder.from_pretrained(
        tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        device=DEVICE,
        dtype=torch.float32,
        **kwargs,
    )
    # load_synthetic zeroes B, which zeroes the delta; re-randomise it so the
    # adapter path actually changes the output and isolation is meaningful.
    for i, tenant in enumerate(("acme", "globex")):
        emb.register_synthetic(tenant, seed=100 + i)
        weight = emb.store.get(tenant)
        torch.manual_seed(200 + i)
        for module in emb.config.target_modules:
            torch.nn.init.normal_(weight.wb[module], std=0.05)
    emb._invalidate_assembler()
    return emb


# --------------------------------------------------------------------------
# 1. chunking and padding
# --------------------------------------------------------------------------


def test_chunk_and_pad_fills_the_last_chunk():
    chunks = list(chunk_and_pad(list(range(5)), 3))
    assert [c for c, _ in chunks] == [[0, 1, 2], [3, 4, 4]]
    assert [n for _, n in chunks] == [3, 2]


def test_chunk_and_pad_exact_multiple_pads_nothing():
    chunks = list(chunk_and_pad(list(range(4)), 2))
    assert [n for _, n in chunks] == [2, 2]


def test_chunk_and_pad_empty_yields_nothing():
    assert list(chunk_and_pad([], 4)) == []


def test_chunk_and_pad_rejects_zero_batch():
    with pytest.raises(ValueError, match="must be positive"):
        list(chunk_and_pad([1], 0))


@pytest.mark.parametrize("n", [1, 7, 8, 9, 20])
def test_embed_accepts_any_length(tiny_model, n):
    """The case that raised before these wrappers existed."""
    emb = _embedder(tiny_model)
    out = emb.embed(_texts(n), "acme")
    assert out.shape == (n, TINY["hidden_size"])


def test_embed_empty_returns_empty(tiny_model):
    emb = _embedder(tiny_model)
    assert emb.embed([], []).shape == (0, TINY["hidden_size"])


def test_embed_rejects_length_mismatch(tiny_model):
    emb = _embedder(tiny_model)
    with pytest.raises(ValueError, match="tenant"):
        emb.embed(_texts(3), ["acme", "globex"])


# --------------------------------------------------------------------------
# 2. padding must not influence real rows
# --------------------------------------------------------------------------


def test_padding_does_not_leak(tiny_model):
    """A 3-text call is padded to the batch; a 16-text call is not. The three
    rows they share must be identical, or the padding is changing results."""
    emb = _embedder(tiny_model)
    texts, tenants = _texts(16), ["acme", "globex"] * 8
    full = emb.embed(texts, tenants)
    short = emb.embed(texts[:3], tenants[:3])
    torch.testing.assert_close(short, full[:3], atol=1e-5, rtol=1e-5)


def test_chunk_boundary_is_invisible(tiny_model):
    """Row 8 starts a new chunk at BATCH=8; it must match a single-item call."""
    emb = _embedder(tiny_model)
    texts, tenants = _texts(20), ["acme", "globex"] * 10
    full = emb.embed(texts, tenants)
    one = emb.embed([texts[8]], [tenants[8]])
    torch.testing.assert_close(one[0], full[8], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 3. tenant isolation through the wrapper
# --------------------------------------------------------------------------


def test_mixed_batch_matches_per_tenant_batches(tiny_model):
    """Each tenant's rows in a mixed batch equal that tenant's own batch."""
    emb = _embedder(tiny_model)
    texts = _texts(6)
    mixed = emb.embed(texts, ["acme", "globex"] * 3)
    acme = emb.embed(texts[0::2], "acme")
    globex = emb.embed(texts[1::2], "globex")
    torch.testing.assert_close(mixed[0::2], acme, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mixed[1::2], globex, atol=1e-5, rtol=1e-5)


def test_different_adapters_give_different_vectors(tiny_model):
    """Guards the test above: identical output would satisfy it vacuously."""
    emb = _embedder(tiny_model)
    text = _texts(1)
    assert not torch.allclose(
        emb.embed(text, "acme"), emb.embed(text, "globex"), atol=1e-4
    )


def test_unknown_tenant_raises(tiny_model):
    emb = _embedder(tiny_model)
    with pytest.raises(KeyError, match="nobody"):
        emb.embed(_texts(1), ["nobody"])


# --------------------------------------------------------------------------
# 4. embeddings: normalisation and pooling
# --------------------------------------------------------------------------


def test_embed_normalises_by_default(tiny_model):
    emb = _embedder(tiny_model)
    norms = emb.embed(_texts(5), "acme").norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_embed_can_skip_normalisation(tiny_model):
    emb = _embedder(tiny_model)
    norms = emb.embed(_texts(5), "acme", normalize=False).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_pooling_strategies_differ(tiny_model):
    emb = _embedder(tiny_model)
    texts = _texts(4)
    cls = emb.embed(texts, "acme", pooling="cls")
    mean = emb.embed(texts, "acme", pooling="mean")
    assert not torch.allclose(cls, mean, atol=1e-3)


def test_unknown_pooling_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="unknown pooling"):
        _embedder(tiny_model, pooling="first_token")


# --------------------------------------------------------------------------
# 5. classification with ragged per-tenant label counts
# --------------------------------------------------------------------------


def _classifier(tiny_model) -> MultiTenantClassifier:
    clf = MultiTenantClassifier.from_pretrained(
        tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        device=DEVICE,
        dtype=torch.float32,
    )
    clf.register_synthetic("small", seed=1)
    clf.register_synthetic("big", seed=2)
    H = TINY["hidden_size"]
    generator = torch.Generator().manual_seed(11)
    clf.heads.load("small", torch.randn(3, H, generator=generator), torch.zeros(3))
    clf.heads.load("big", torch.randn(10, H, generator=generator), torch.zeros(10))
    return clf


def test_each_tenant_gets_its_own_label_count(tiny_model):
    clf = _classifier(tiny_model)
    probs = clf.predict_proba(_texts(4), ["small", "big", "small", "big"])
    assert [len(p) for p in probs] == [3, 10, 3, 10]


def test_probabilities_sum_to_one_over_real_labels(tiny_model):
    """The padded lanes are not classes: a 3-label tenant batched with a
    10-label one must still sum to 1 over its own three."""
    clf = _classifier(tiny_model)
    for row in clf.predict_proba(_texts(4), ["small", "big", "small", "big"]):
        assert abs(sum(row) - 1.0) == pytest.approx(0.0, abs=1e-5)


def test_predictions_are_in_range(tiny_model):
    clf = _classifier(tiny_model)
    tenants = ["small", "big", "small", "big"]
    preds = clf.predict(_texts(4), tenants)
    limits = [clf.heads.num_labels(t) for t in tenants]
    assert all(0 <= p < n for p, n in zip(preds, limits))


def test_single_logit_head_is_binary(tiny_model):
    """A binary sklearn classifier keeps one row; softmax over one class is the
    constant 1, so that head must take the sigmoid path instead."""
    clf = _classifier(tiny_model)
    H = TINY["hidden_size"]
    clf.heads.load("small", torch.randn(1, H, generator=torch.Generator().manual_seed(3)), torch.zeros(1))
    probs = clf.predict_proba(_texts(2), ["small", "small"])
    assert all(len(p) == 1 and 0.0 < p[0] < 1.0 for p in probs)
    assert all(p in (0, 1) for p in clf.predict(_texts(2), ["small", "small"]))


def test_head_rejects_wrong_hidden_size(tiny_model):
    clf = _classifier(tiny_model)
    with pytest.raises(ValueError, match="hidden size"):
        clf.heads.load("small", torch.randn(3, TINY["hidden_size"] + 1), torch.zeros(3))


def test_head_rejects_mismatched_intercept(tiny_model):
    clf = _classifier(tiny_model)
    with pytest.raises(ValueError, match="intercept"):
        clf.heads.load("small", torch.randn(3, TINY["hidden_size"]), torch.zeros(5))


def test_missing_head_names_the_tenant(tiny_model):
    clf = _classifier(tiny_model)
    clf.register_synthetic("headless", seed=9)
    with pytest.raises(KeyError, match="headless"):
        clf.predict(_texts(1), ["headless"])


def test_head_store_accounting(tiny_model):
    cfg = LoraServingConfig(
        model_name=tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        target_modules=["query", "value"],
        device=DEVICE,
        dtype=torch.float32,
    )
    heads = HeadStore(cfg)
    heads.load("a", torch.zeros(3, TINY["hidden_size"]), torch.zeros(3))
    assert len(heads) == 1 and "a" in heads and heads.memory_bytes() > 0
    heads.evict("a")
    assert len(heads) == 0 and heads.tenant_ids() == []


# --------------------------------------------------------------------------
# 6. reranking
# --------------------------------------------------------------------------


def test_rerank_orders_descending_and_keeps_indices(tiny_model):
    rr = MultiTenantReranker.from_pretrained(
        tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        device=DEVICE,
        dtype=torch.float32,
    )
    rr.register_synthetic("acme", seed=5)
    rr.heads.load(
        "acme",
        torch.randn(1, TINY["hidden_size"], generator=torch.Generator().manual_seed(4)),
        torch.zeros(1),
    )
    docs = _texts(9)
    ranked = rr.rerank("tok1 tok2", docs, "acme")

    assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)
    assert sorted(r.index for r in ranked) == list(range(9))
    assert all(ranked_doc.document == docs[ranked_doc.index] for ranked_doc in ranked)

    top3 = rr.rerank("tok1 tok2", docs, "acme", top_k=3)
    assert [r.index for r in top3] == [r.index for r in ranked[:3]]

    # score() must agree with what rerank() sorted on
    scores = rr.score(["tok1 tok2"] * 9, docs, "acme")
    by_index = {r.index: r.score for r in ranked}
    assert scores == pytest.approx([by_index[i] for i in range(9)], abs=1e-5)


def test_rerank_rejects_unequal_pair_lists(tiny_model):
    rr = MultiTenantReranker.from_pretrained(
        tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        device=DEVICE,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="one entry per pair"):
        rr.score(["q", "q"], ["d"], "acme")


# --------------------------------------------------------------------------
# 7. assembler / engine equivalence
# --------------------------------------------------------------------------


def test_index_select_assembler_matches_cpu(tiny_model):
    cpu = _embedder(tiny_model)
    gpu = _embedder(tiny_model, assembler="index_select")
    texts, tenants = _texts(10), ["acme", "globex"] * 5
    torch.testing.assert_close(
        cpu.embed(texts, tenants), gpu.embed(texts, tenants), atol=1e-5, rtol=1e-5
    )


def test_registering_after_first_use_is_visible(tiny_model):
    """IndexSelectBatchAssembler packs the store at construction, so a stale one
    would not see a tenant added later — or would serve an evicted one."""
    emb = _embedder(tiny_model, assembler="index_select")
    emb.embed(_texts(2), "acme")  # builds the packed tensors
    emb.register_synthetic("newcomer", seed=42)
    assert emb.embed(_texts(2), "newcomer").shape == (2, TINY["hidden_size"])


def test_bert_engine_matches_hook_engine(tiny_model):
    """Both engines must serve the same checkpoint identically — the property
    that lets a wrapper-measured row be compared with the paper's."""
    hooks = _embedder(tiny_model, engine="hf", pooling="cls")
    bert = _embedder(tiny_model, engine="bert", pooling="cls")
    texts, tenants = _texts(5), ["acme", "globex", "acme", "globex", "acme"]
    torch.testing.assert_close(
        hooks.embed(texts, tenants), bert.embed(texts, tenants), atol=1e-4, rtol=1e-4
    )


def test_unknown_engine_and_assembler_are_rejected(tiny_model):
    with pytest.raises(ValueError, match="unknown engine"):
        _embedder(tiny_model, engine="tensorrt")
    with pytest.raises(ValueError, match="unknown assembler"):
        _embedder(tiny_model, assembler="magic")


# --------------------------------------------------------------------------
# 8. PEFT round-trip
# --------------------------------------------------------------------------


def test_peft_key_fn_matches_peft_naming(tiny_model):
    key_a, key_b = peft_key_fn("bert")(3, "value")
    assert key_a == "base_model.model.encoder.layer.3.attention.self.value.lora_A.weight"
    assert key_b == "base_model.model.encoder.layer.3.attention.self.value.lora_B.weight"


def test_peft_key_fn_uses_deberta_projection_names():
    key_a, _ = peft_key_fn("deberta-v2")(0, "query")
    assert "query_proj" in key_a


def test_peft_adapter_round_trip(tiny_model, tmp_path):
    """An adapter saved by PEFT loads and reproduces PEFT's own output."""
    pytest.importorskip("peft", reason="peft not installed")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    torch.manual_seed(3)
    base = AutoModel.from_pretrained(tiny_model, dtype=torch.float32)
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=RANK,
            lora_alpha=RANK,  # scale 1.0
            target_modules=["query", "value"],
            lora_dropout=0.0,
            bias="none",
        ),
    )
    # A freshly initialised LoRA has B = 0, so the delta is zero and the check
    # would pass against the base model alone. Randomise B first.
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    peft_model.eval()

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))

    cfg = LoraServingConfig(
        model_name=tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        target_modules=["query", "value"],
        device=DEVICE,
        dtype=torch.float32,
    )
    store = AdapterStore(cfg)
    saved = adapter_dir / "default" if (adapter_dir / "default").is_dir() else adapter_dir
    load_peft_adapter(store, "tenant", str(saved))
    assert len(store) == 1

    emb = MultiTenantEmbedder(cfg, engine="hf", pooling="cls", store=store)
    ours = emb.embed(_texts(3), "tenant", normalize=False)

    input_ids = emb.tokenizer(
        _texts(3), padding="max_length", truncation=True,
        max_length=SEQ, return_tensors="pt",
    )
    with torch.no_grad():
        theirs = peft_model(
            input_ids=input_ids["input_ids"].to(DEVICE),
            attention_mask=input_ids["attention_mask"].to(DEVICE),
        ).last_hidden_state[:, 0]

    torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-4)


def test_peft_loader_rejects_rank_mismatch(tiny_model, tmp_path):
    """A rank mismatch is otherwise a shape assertion naming neither file."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": RANK * 2, "lora_alpha": RANK * 2, "target_modules": ["query", "value"]})
    )
    (adapter_dir / "adapter_model.bin").write_bytes(b"")
    cfg = LoraServingConfig(
        model_name=tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        target_modules=["query", "value"],
        device=DEVICE,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="rank"):
        load_peft_adapter(AdapterStore(cfg), "t", str(adapter_dir))


def test_peft_loader_rejects_target_module_mismatch(tiny_model, tmp_path):
    """Silent otherwise: the deltas would load into the wrong projections."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": RANK, "lora_alpha": RANK, "target_modules": ["query", "key"]})
    )
    (adapter_dir / "adapter_model.bin").write_bytes(b"")
    cfg = LoraServingConfig(
        model_name=tiny_model,
        lora_rank=RANK,
        batch_size=BATCH,
        max_seq_len=SEQ,
        target_modules=["query", "value"],
        device=DEVICE,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="wrong projections"):
        load_peft_adapter(AdapterStore(cfg), "t", str(adapter_dir))


# --------------------------------------------------------------------------
# 9. exports
# --------------------------------------------------------------------------


def test_every_export_is_importable():
    import latefuse

    missing = [name for name in latefuse.__all__ if not hasattr(latefuse, name)]
    assert not missing, f"__all__ names nothing: {missing}"
