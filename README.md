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
uv sync                               # core runtime
uv sync --extra dev                   # + pytest / ruff (run the test suite)
uv sync --extra dev --extra quality   # + setfit / datasets / scikit-learn
```

The `quality` extra is needed to run the accuracy benchmarks
(`benchmarks/quality/setfit_compare.py`) and the full test suite —
`tests/test_setfit_equivalence.py` skips without `scikit-learn`.

Uses [uv](https://docs.astral.sh/uv/). On Linux, CUDA torch is pulled from the
PyTorch cu128 index; on macOS/CPU the regular PyPI build resolves automatically.

## Layout

```
src/lora_serving/        Core serving engine
  model/encoder.py       Encoder forward with late-fusion LoRA
  ops/lora.py            Batched LoRA shrink/expand (torch.bmm)
  ops/head.py            Zero-padded batched per-tenant classification heads
  weights/store.py       GPU-resident AdapterStore (pre-loaded adapters)
  weights/batch.py       BatchAssembler (per-request gather of adapter slices)
  config.py             LoraServingConfig
  benchmark/            Synthetic-adapter microbenchmark harness
deploy/                  Reference horizontal deployment
  server/               FastAPI serving layer, dynamic batcher, sticky routing
  k8s/                  Kubernetes manifests (Deployment, HPA, Ingress, Redis)
  Dockerfile
```

## Quickstart

```python
import torch
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.store import AdapterStore
from lora_serving.weights.batch import BatchAssembler

cfg = LoraServingConfig(
    model_name="BAAI/bge-m3",
    lora_rank=8,
    batch_size=32,
    max_seq_len=128,
    target_modules=["query", "value"],
    device=torch.device("cuda"),
    dtype=torch.float16,
)

model = EncoderWithLora.from_pretrained_serving(cfg).eval()
store = AdapterStore(cfg)
store.load_synthetic("tenant-a", seed=0)   # or load trained adapter weights
assembler = BatchAssembler(store, cfg)
```

The full request path — tokenize → assemble per-sample adapter slices → single
batched forward → per-tenant head — is wired up in
[`deploy/server/app.py`](deploy/server/app.py) (`run_inference`).

## Deployment

See [`deploy/README.md`](deploy/README.md) for the containerized, horizontally
scaled reference deployment (stateless replicas, tenant-sticky consistent-hash
routing, HPA autoscaling, Redis-driven hot adapter reload). The manifests are
intentionally minimal — adapt batching, autoscaling, and adapter-placement
policy to your workload.