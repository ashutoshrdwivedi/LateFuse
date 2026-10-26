"""Task-level wrappers over the LateFuse serving path."""

from latefuse.tasks.base import ASSEMBLERS, ENGINES, TaskPipeline
from latefuse.tasks.batching import chunk_and_pad
from latefuse.tasks.classification import MultiTenantClassifier
from latefuse.tasks.embedding import MultiTenantEmbedder
from latefuse.tasks.heads import HeadStore, LRHead
from latefuse.tasks.reranking import MultiTenantReranker, RerankResult

__all__ = [
    "ASSEMBLERS",
    "ENGINES",
    "HeadStore",
    "LRHead",
    "MultiTenantClassifier",
    "MultiTenantEmbedder",
    "MultiTenantReranker",
    "RerankResult",
    "TaskPipeline",
    "chunk_and_pad",
]
