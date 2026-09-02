"""Multi-tenant embedding generation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from latefuse.tasks.base import TaskPipeline


class MultiTenantEmbedder(TaskPipeline):
    """Embed texts under many tenants' adapters in one shared forward pass.

    Every text carries its own tenant id; the batch is mixed, the base
    projections run once, and each sample gets its own low-rank delta.

    Example:
        embedder = MultiTenantEmbedder.from_pretrained("BAAI/bge-m3")
        embedder.register_adapter("acme", "adapters/acme")
        embedder.register_adapter("globex", "adapters/globex")

        vectors = embedder.embed(
            ["renew my policy", "where is my order"],
            ["acme", "globex"],
        )                                   # (2, H), L2-normalised

    Pooling defaults to ``"cls"`` rather than the engine's ``"auto"``: for the
    encoders this serves, the sentence embedding is the CLS hidden state
    (``bge-m3``) or a mask-weighted mean (most sentence-transformers and E5
    checkpoints), whereas ``"auto"`` returns the checkpoint's trained *pooler*
    output — a different vector, trained for next-sentence prediction rather
    than similarity. Pass ``pooling="mean"`` for an E5- or MPNet-style
    checkpoint, and check what your model card specifies; nothing downstream can
    tell you the choice was wrong.
    """

    def __init__(self, *args, pooling: str = "cls", **kwargs):
        super().__init__(*args, pooling=pooling, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name: str, *, pooling: str = "cls", **kwargs):
        return super().from_pretrained(model_name, pooling=pooling, **kwargs)

    def embed(
        self,
        texts: Sequence[str],
        tenants: Sequence[str] | str,
        *,
        normalize: bool = True,
        pooling: str | None = None,
    ) -> Tensor:
        """Embed ``texts``, each under its own tenant's adapter.

        Args:
            texts:     The texts to embed. Any length; chunked internally.
            tenants:   One tenant id per text, or a single id applied to all.
            normalize: L2-normalise each row (default), so a dot product between
                       two results is their cosine similarity — what a vector
                       index will assume.
            pooling:   Override this embedder's pooling for one call.

        Returns:
            ``(len(texts), hidden_size)``, rows in input order.
        """
        tenants = self._broadcast_tenants(tenants, len(texts))
        vectors = self._encode(texts, tenants, pooling=pooling)
        if normalize and vectors.numel():
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors

    @staticmethod
    def _broadcast_tenants(tenants: Sequence[str] | str, n: int) -> list[str]:
        """Accept one id for a whole call, since embedding a corpus for a single
        tenant is the common case and repeating the id ``n`` times is noise."""
        if isinstance(tenants, str):
            return [tenants] * n
        return list(tenants)
