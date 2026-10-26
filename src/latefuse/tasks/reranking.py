"""Multi-tenant cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from latefuse.ops.head import LRHeadOps
from latefuse.tasks.base import TaskPipeline
from latefuse.tasks.heads import HeadStore


@dataclass(frozen=True)
class RerankResult:
    """One scored document, carrying its position in the original list."""

    index: int
    document: str
    score: float


class MultiTenantReranker(TaskPipeline):
    """Score (query, document) pairs under many tenants' adapters at once.

    A cross-encoder reranker is an encoder forward over a concatenated pair plus
    a single-logit head, so it runs on exactly the serving path the paper
    measures: the pair is one sequence, the tenant's adapter applies as a
    per-sample delta, and one batched matrix multiply produces the scores.

    Note:
        The serving path here is the measured one, but reranking itself was
        never load-tested end-to-end — the paper's throughput and latency
        numbers are taken on single-sequence classification and embedding
        workloads. Pair inputs consume roughly twice the sequence budget per
        request, so plan ``max_seq_len`` accordingly and measure before
        assuming the published figures transfer.

    Example:
        rr = MultiTenantReranker.from_pretrained("BAAI/bge-m3", max_seq_len=256)
        rr.register_adapter("acme", "adapters/acme")
        rr.heads.load("acme", coef, intercept)          # one logit

        rr.rerank("how do I renew?", ["...", "..."], "acme", top_k=5)
    """

    def __init__(self, *args, heads: HeadStore | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads if heads is not None else HeadStore(self.config)

    def score(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
        tenants: Sequence[str] | str,
    ) -> list[float]:
        """Relevance score for each ``(query, document)`` pair, in input order.

        Args:
            queries:   One query per pair.
            documents: One document per pair, parallel to ``queries``.
            tenants:   One tenant id per pair, or a single id for all of them.

        Returns:
            One float per pair. Higher is more relevant. These are raw logits,
            comparable within a tenant but not calibrated across tenants — each
            head was fitted separately.
        """
        if len(queries) != len(documents):
            raise ValueError(
                f"got {len(queries)} queries but {len(documents)} documents; "
                "score() takes parallel lists, one entry per pair"
            )
        if not queries:
            return []
        tenant_ids = (
            [tenants] * len(queries) if isinstance(tenants, str) else list(tenants)
        )

        pooled = self._encode(queries, tenant_ids, text_pairs=list(documents))
        batched = self.heads.assemble(tenant_ids)
        out = torch.zeros(
            pooled.shape[0], 1, batched.coef.shape[1],
            dtype=self.config.dtype, device=self.config.device,
        )
        LRHeadOps.predict_proba(
            pooled.unsqueeze(1), batched.coef, batched.intercept, out
        )
        # Multi-label heads are allowed through: column 0 is the relevance
        # logit by convention, and a one-column head (the usual case) makes that
        # the only column.
        return out[:, 0, 0].tolist()

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        tenant: str,
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rank ``documents`` against one ``query`` for one tenant.

        Args:
            top_k: Keep only the best ``k``. None (default) returns all.

        Returns:
            :class:`RerankResult` list, highest score first. ``index`` is the
            document's position in the input list, so the caller can map back to
            whatever metadata it holds alongside.
        """
        scores = self.score([query] * len(documents), documents, tenant)
        ranked = sorted(
            (
                RerankResult(index=i, document=doc, score=score)
                for i, (doc, score) in enumerate(zip(documents, scores))
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked if top_k is None else ranked[:top_k]
