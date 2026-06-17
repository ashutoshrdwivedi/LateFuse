"""FastAPI inference server for multi-tenant LoRA serving.

Endpoints:
    POST /predict          — inference for a single tenant+text
    GET  /health           — liveness probe (always 200)
    GET  /ready            — readiness probe (200 after model loaded)
    POST /admin/reload     — manually trigger adapter reload

On startup:
    1. Loads the frozen base encoder onto GPU
    2. Pre-loads adapters listed in LORA_PRELOAD_ADAPTER_IDS
    3. Starts the dynamic request batcher
    4. Starts the Redis pub/sub adapter reloader
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

from deploy.server.batcher import DynamicBatcher, InferenceRequest, InferenceResult
from deploy.server.config import ServerConfig
from deploy.server.reload import AdapterReloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (initialized in lifespan)
# ---------------------------------------------------------------------------
_server_config: ServerConfig | None = None
_serving_config: LoraServingConfig | None = None
_model: EncoderWithLora | None = None
_tokenizer: AutoTokenizer | None = None
_store: AdapterStore | None = None
_assembler: BatchAssembler | None = None
_batcher: DynamicBatcher | None = None
_reloader: AdapterReloader | None = None
_inference_lock = threading.Lock()
_ready = False


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    tenant_id: str
    text: str


class PredictResponse(BaseModel):
    tenant_id: str
    scores: list[float]
    latency_ms: float


class ReloadRequest(BaseModel):
    adapter_id: str
    s3_key: str


# ---------------------------------------------------------------------------
# Inference function (called by DynamicBatcher in a thread executor)
# ---------------------------------------------------------------------------
def run_inference(batch: list[InferenceRequest]) -> list[InferenceResult]:
    """Run a batched forward pass through the model.

    This is the hot path — runs on the GPU in a worker thread.
    The inference_lock prevents adapter reloads mid-forward-pass.
    """
    assert _model is not None
    assert _tokenizer is not None
    assert _store is not None
    assert _assembler is not None
    assert _serving_config is not None

    t0 = time.monotonic()

    # Tokenize all texts in the batch
    texts = [req.text for req in batch]
    tenant_ids = [req.tenant_id for req in batch]
    actual_batch_size = len(batch)

    encoded = _tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=_serving_config.max_seq_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(_serving_config.device)
    attention_mask = encoded["attention_mask"].to(_serving_config.device)

    # Build synthetic LR weights (in production, these come from a per-tenant store)
    num_labels = _server_config.num_labels
    lr_coefs = []
    lr_intercepts = []
    for _ in tenant_ids:
        coef = torch.randn(1, num_labels, _serving_config.hidden_size,
                           dtype=_serving_config.dtype, device=_serving_config.device)
        intercept = torch.zeros(1, num_labels,
                                dtype=_serving_config.dtype, device=_serving_config.device)
        lr_coefs.append(coef)
        lr_intercepts.append(intercept)

    output_lr = torch.zeros(actual_batch_size, 1, num_labels,
                            dtype=_serving_config.dtype, device=_serving_config.device)

    with _inference_lock:
        lora_w, lr_w = _assembler.assemble(tenant_ids, lr_coefs, lr_intercepts)
        with torch.no_grad():
            _model(input_ids, attention_mask, lora_w, lr_w, output_lr)
        torch.cuda.synchronize(_serving_config.device)

    latency_ms = (time.monotonic() - t0) * 1000

    # Fan out results
    results = []
    for i in range(actual_batch_size):
        scores = output_lr[i, 0, :num_labels].cpu().tolist()
        results.append(InferenceResult(scores=scores, latency_ms=latency_ms))

    return results


# ---------------------------------------------------------------------------
# Adapter reload callback (called from AdapterReloader thread)
# ---------------------------------------------------------------------------
def reload_adapter_callback(adapter_id: str, local_path: str) -> None:
    """Reload an adapter from a local file into GPU memory."""
    assert _store is not None
    assert _serving_config is not None

    # For synthetic/benchmark mode, just reload synthetic weights
    # In production, use store.load_from_file() with proper key_fn
    _store.load_synthetic(adapter_id, seed=hash(adapter_id) % (2**31))
    logger.info("Adapter '%s' reloaded (synthetic mode)", adapter_id)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server_config, _serving_config, _model, _tokenizer
    global _store, _assembler, _batcher, _reloader, _ready

    logger.info("Starting up...")
    _server_config = ServerConfig()

    device = torch.device(_server_config.device)
    dtype = getattr(torch, _server_config.dtype)

    _serving_config = LoraServingConfig(
        model_name=_server_config.model_name,
        lora_rank=_server_config.lora_rank,
        batch_size=_server_config.max_batch_size,
        max_seq_len=_server_config.max_seq_len,
        target_modules=_server_config.target_modules,
        device=device,
        dtype=dtype,
    )

    # 1. Load tokenizer
    logger.info("Loading tokenizer for %s...", _server_config.model_name)
    _tokenizer = AutoTokenizer.from_pretrained(_server_config.model_name)

    # 2. Load base model
    logger.info("Loading base model %s onto %s...", _server_config.model_name, device)
    _model = EncoderWithLora.from_pretrained_serving(_serving_config)
    _model.eval()
    logger.info("Base model loaded (%.2f GB GPU)", torch.cuda.max_memory_allocated(device) / 1e9)

    # 3. Initialize adapter store and pre-load adapters
    _store = AdapterStore(_serving_config)
    for adapter_id in _server_config.preload_adapter_ids:
        logger.info("Pre-loading adapter '%s'...", adapter_id)
        _store.load_synthetic(adapter_id, seed=hash(adapter_id) % (2**31))
    logger.info("Adapter store: %d adapters (%.3f GB)", len(_store), _store.memory_gb())

    # 4. Create batch assembler
    _assembler = BatchAssembler(_store, _serving_config)

    # 5. Start dynamic batcher
    _batcher = DynamicBatcher(
        max_batch_size=_server_config.max_batch_size,
        batch_timeout_ms=_server_config.batch_timeout_ms,
        inference_fn=run_inference,
    )
    await _batcher.start()

    # 6. Start adapter reloader (Redis pub/sub)
    try:
        _reloader = AdapterReloader(
            config=_server_config,
            reload_callback=reload_adapter_callback,
            inference_lock=_inference_lock,
        )
        _reloader.start()
    except Exception as e:
        logger.warning("Could not start adapter reloader (Redis unavailable?): %s", e)
        _reloader = None

    _ready = True
    logger.info("Server ready for inference")

    yield  # --- server runs ---

    # Shutdown
    logger.info("Shutting down...")
    _ready = False
    if _batcher:
        await _batcher.stop()
    if _reloader:
        _reloader.stop()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LoRA Multi-Tenant Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    """Readiness probe — returns 200 only after model and adapters are loaded."""
    if not _ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "adapters_loaded": len(_store) if _store else 0}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run inference for a single tenant.

    The request is queued into the dynamic batcher, which collects multiple
    requests into a single GPU batch for maximum throughput.

    Headers:
        X-Tenant-Id is used by NGINX Ingress for consistent-hash routing.
    """
    if not _ready:
        raise HTTPException(status_code=503, detail="Server not ready")
    if _batcher is None:
        raise HTTPException(status_code=500, detail="Batcher not initialized")

    # Validate tenant has a loaded adapter
    if _store and req.tenant_id not in _store._store:
        raise HTTPException(status_code=404, detail=f"Adapter '{req.tenant_id}' not loaded")

    result = await _batcher.submit(req.tenant_id, req.text)
    return PredictResponse(
        tenant_id=req.tenant_id,
        scores=result.scores,
        latency_ms=round(result.latency_ms, 2),
    )


@app.post("/admin/reload")
async def admin_reload(req: ReloadRequest):
    """Manually trigger an adapter reload from S3."""
    if _reloader is None:
        raise HTTPException(status_code=503, detail="Reloader not available")

    try:
        local_path = _reloader.download_adapter(req.adapter_id, req.s3_key)
        reload_adapter_callback(req.adapter_id, local_path)
        return {"status": "reloaded", "adapter_id": req.adapter_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
