"""LateFuse: multi-tenant LoRA serving for encoder models.

Many LoRA-adapted encoders served in one shared forward pass on one GPU. The
frozen base projections run once over a mixed-tenant batch; each sample's
low-rank delta and each tenant's classification head are applied as batched
matrix multiplications. Pure PyTorch — no custom kernels, no merge step, and so
no cross-tenant contamination by construction.

Start with a task wrapper::

    from latefuse import MultiTenantEmbedder

    embedder = MultiTenantEmbedder.from_pretrained("BAAI/bge-m3")
    embedder.register_adapter("acme", "adapters/acme")
    embedder.register_adapter("globex", "adapters/globex")

    vectors = embedder.embed(
        ["renew my policy", "where is my order"],
        ["acme", "globex"],          # one tenant per text, one forward pass
    )

The engines, stores and assemblers underneath are exported too, for callers who
want to drive the serving path directly.
"""

from latefuse.config import LoraServingConfig
from latefuse.loaders import load_peft_adapter, peft_key_fn
from latefuse.model.encoder import EncoderWithLora
from latefuse.model.hf_wrapper import HFEncoderWithLora
from latefuse.ops.head import LRHeadOps
from latefuse.ops.lora import LoraOps
from latefuse.ops.pooling import POOLING_STRATEGIES
from latefuse.tasks import (
    HeadStore,
    LRHead,
    MultiTenantClassifier,
    MultiTenantEmbedder,
    MultiTenantReranker,
    RerankResult,
    TaskPipeline,
)
from latefuse.weights.batch import (
    BatchAssembler,
    BatchedLRWeights,
    IndexSelectBatchAssembler,
    LayerwiseBatchedWeights,
)
from latefuse.weights.store import AdapterStore, LoraWeight

__version__ = "0.1.0"

__all__ = [
    # Task-level API — start here.
    "MultiTenantEmbedder",
    "MultiTenantClassifier",
    "MultiTenantReranker",
    "RerankResult",
    "TaskPipeline",
    # Configuration and adapter loading.
    "LoraServingConfig",
    "load_peft_adapter",
    "peft_key_fn",
    "POOLING_STRATEGIES",
    # Stores.
    "AdapterStore",
    "LoraWeight",
    "HeadStore",
    "LRHead",
    # Engines.
    "EncoderWithLora",
    "HFEncoderWithLora",
    # Batch assembly and the low-level ops.
    "BatchAssembler",
    "IndexSelectBatchAssembler",
    "BatchedLRWeights",
    "LayerwiseBatchedWeights",
    "LoraOps",
    "LRHeadOps",
    "__version__",
]
