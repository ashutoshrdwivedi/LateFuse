# LateFuse

Efficient multi-tenant LoRA serving for **encoder** models via late-fusion
batched matrix multiplication.

LateFuse serves many different LoRA-adapted encoders in a single shared forward
pass on one GPU. The shared base projections run once over a mixed-tenant batch,
while per-sample low-rank deltas and per-tenant classification heads are applied
as batched matrix multiplications (`torch.bmm`) — no custom CUDA kernels, pure
PyTorch. Tenant weights are never merged into the shared base, so cross-tenant
contamination is ruled out by construction.

## Why

- **Flat latency in adapter count.** Serving latency stays flat as the adapter
  pool grows to tens of thousands on a single GPU; adapter count affects only
  CPU-side batch assembly, not the GPU forward pass.
- **No kernel build step.** Standard `torch.bmm` deploys on managed GPU
  platforms with no compilation step.
- **Isolation by construction.** Per-tenant adapters and heads are applied in
  activation space; the frozen base weights are never modified.

## Install

```bash
pip install latefuse
```

## Quickstart

Every text carries its own tenant id. The batch is mixed; the base model runs
once.

```python
from latefuse import MultiTenantEmbedder

embedder = MultiTenantEmbedder.from_pretrained("BAAI/bge-m3")
embedder.register_adapter("acme",   "adapters/acme")     # a PEFT adapter dir
embedder.register_adapter("globex", "adapters/globex")

vectors = embedder.embed(
    ["renew my policy", "where is my order"],
    ["acme", "globex"],
)                                  # (2, 1024), L2-normalised
```

Request lists can be any length: they are chunked to the configured batch size
and the short chunk is padded internally, so `embed(["one text"])` works.

## Task wrappers

Three wrappers cover the common encoder workloads. All three share the same
adapter store, engine, and batching; they differ only in what they do with the
pooled output.

**Embedding / search** — `MultiTenantEmbedder`

```python
vectors = embedder.embed(texts, tenants)            # normalised → dot = cosine
vectors = embedder.embed(texts, "acme", pooling="mean")
```

Pooling defaults to `"cls"` (the `bge-m3` convention). E5- and
sentence-transformers-style checkpoints want `pooling="mean"`; `"auto"` returns
the checkpoint's trained pooler output. Check your model card — a wrong choice
returns well-formed vectors that are simply the wrong ones.

**Classification** — `MultiTenantClassifier`

```python
from latefuse import MultiTenantClassifier

clf = MultiTenantClassifier.from_pretrained("BAAI/bge-m3")
clf.register_adapter("acme", "adapters/acme")
clf.heads.load_from_sklearn("acme", fitted_logreg)   # or .load(coef, intercept)

clf.predict_proba(["renew my policy"], ["acme"])     # [[0.1, 0.2, 0.7]]
```

Tenants need not agree on how many labels they have: the batch is zero-padded to
the widest head present and each row is sliced back to its own label count
before softmax.

**Reranking** — `MultiTenantReranker`

```python
from latefuse import MultiTenantReranker

rr = MultiTenantReranker.from_pretrained("BAAI/bge-m3", max_seq_len=256)
rr.register_adapter("acme", "adapters/acme")
rr.heads.load("acme", coef, intercept)               # one logit

for hit in rr.rerank("how do I renew?", documents, "acme", top_k=5):
    print(hit.index, hit.score, hit.document)
```

A cross-encoder pair is one sequence through the same measured serving path.
Note that pairs consume roughly twice the sequence budget per request, and that
reranking has not been load-tested end-to-end — measure before assuming the
published throughput figures transfer.

## Adapters

`register_adapter` reads an ordinary PEFT adapter directory
(`adapter_config.json` + `adapter_model.safetensors`), checks its rank and
target modules against the pipeline's, and folds `alpha/r` into the stored
weights. A mismatch raises rather than serving silently-wrong output.

```python
pipeline.register_adapter("acme", "adapters/acme")
pipeline.register_synthetic("bench-tenant", seed=0)  # random, for benchmarking
pipeline.evict_adapter("acme")
```

One pipeline serves one rank and one set of target modules — a batch is
rank-homogeneous. Build a second pipeline for a second rank.

## Tuning

```python
MultiTenantEmbedder.from_pretrained(
    "BAAI/bge-m3",
    lora_rank=8,
    batch_size=32,          # fixes the engine's pre-allocated buffers
    max_seq_len=128,
    target_modules=("query", "value"),
    engine="hf",            # "hf" hooks any HF encoder; "bert" is the BERT-family path
    assembler="cpu",        # "index_select" moves batch assembly onto the GPU
)
```

`assembler="index_select"` holds the adapter cache in one contiguous GPU tensor
and gathers with `torch.index_select`, removing the host-side loop that
dominates per-request overhead on small models. Every published benchmark number
uses the default CPU assembler and is therefore conservative.

## Development

```bash
uv sync --extra dev                   # + pytest / ruff (run the test suite)
uv sync --extra dev --extra quality   # + setfit / datasets / scikit-learn
uv run pytest tests/ -m "not real_checkpoints"
```

The `quality` extra is needed to run the accuracy benchmarks
(`benchmarks/quality/setfit_compare.py`) and the full test suite —
`tests/test_setfit_equivalence.py` skips without `scikit-learn`.

Uses [uv](https://docs.astral.sh/uv/). On Linux, CUDA torch is pulled from the
PyTorch cu128 index; on macOS/CPU the regular PyPI build resolves automatically.
The published package declares dependency *ranges* so it installs alongside
other torch projects; the exact versions every benchmark was measured with are
pinned in `uv.lock` (`uv sync --locked`).

## Layout

```
src/latefuse/            Core serving engine
  tasks/                 Task-level wrappers (embedding, classification, reranking)
  tasks/heads.py         Per-tenant classification head store
  tasks/batching.py      Chunk + pad arbitrary request lists to the fixed batch
  model/encoder.py       Encoder forward with late-fusion LoRA (BERT family)
  model/hf_wrapper.py    Same LoRA path via forward hooks on any HF encoder
  ops/lora.py            Batched LoRA shrink/expand (torch.bmm)
  ops/head.py            Zero-padded batched per-tenant classification heads
  ops/pooling.py         cls / mean / auto pooling strategies
  weights/store.py       GPU-resident AdapterStore (pre-loaded adapters)
  weights/batch.py       CPU and index_select batch assemblers
  loaders.py             PEFT adapter loading
  config.py              LoraServingConfig
  benchmark/             Synthetic-adapter microbenchmark harness
deploy/                  Reference horizontal deployment
  server/                FastAPI serving layer, dynamic batcher, sticky routing
  k8s/                   Kubernetes manifests (Deployment, HPA, Ingress, Redis)
  Dockerfile
```

## Deployment

See [`deploy/README.md`](deploy/README.md) for the containerized, horizontally
scaled reference deployment (stateless replicas, tenant-sticky consistent-hash
routing, HPA autoscaling, Redis-driven hot adapter reload). The manifests are
intentionally minimal — adapt batching, autoscaling, and adapter-placement
policy to your workload.

## License

Apache-2.0. See [LICENSE](LICENSE).
