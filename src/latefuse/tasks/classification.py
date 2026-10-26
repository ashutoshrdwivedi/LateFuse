"""Multi-tenant sequence classification."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from latefuse.ops.head import LRHeadOps
from latefuse.tasks.base import TaskPipeline
from latefuse.tasks.heads import HeadStore


class MultiTenantClassifier(TaskPipeline):
    """Classify texts under many tenants' adapters *and* heads in one forward pass.

    Each tenant brings two things: a LoRA adapter that shapes the encoder, and a
    classification head fitted on their own label set. Both are applied per
    sample inside a single mixed batch — the adapters as batched low-rank
    deltas, the heads as one batched matrix multiply.

    Tenants need not agree on how many labels they have. The batch is
    zero-padded to the widest head present and each sample is sliced back to its
    own count before softmax, so a 3-class tenant never sees probability mass
    leak into a 10-class tenant's unused lanes.

    Example:
        clf = MultiTenantClassifier.from_pretrained("BAAI/bge-m3")
        clf.register_adapter("acme", "adapters/acme")
        clf.heads.load_from_sklearn("acme", fitted_logreg)

        clf.predict(["renew my policy"], ["acme"])        # -> [2]
        clf.predict_proba(["renew my policy"], ["acme"])  # -> [[0.1, 0.2, 0.7]]
    """

    def __init__(self, *args, heads: HeadStore | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads if heads is not None else HeadStore(self.config)

    def logits(
        self, texts: Sequence[str], tenants: Sequence[str]
    ) -> list[list[float]]:
        """Raw per-tenant logits, ragged: row ``i`` has that tenant's label count.

        Returns a list of lists rather than a tensor precisely because the rows
        are ragged — a dense tensor would have to keep the zero padding, and the
        padded lanes are not classes.
        """
        if len(texts) != len(tenants):
            raise ValueError(
                f"got {len(texts)} texts but {len(tenants)} tenant ids"
            )
        if not texts:
            return []

        pooled = self._encode(texts, tenants)              # (N, H)
        batched = self.heads.assemble(list(tenants))       # padded to max_labels
        out = torch.zeros(
            pooled.shape[0], 1, batched.coef.shape[1],
            dtype=self.config.dtype, device=self.config.device,
        )
        LRHeadOps.predict_proba(
            pooled.unsqueeze(1), batched.coef, batched.intercept, out
        )
        scores = out[:, 0]                                  # (N, max_labels)
        return [
            scores[i, :n].tolist() for i, n in enumerate(batched.num_labels)
        ]

    def predict_proba(
        self, texts: Sequence[str], tenants: Sequence[str]
    ) -> list[list[float]]:
        """Per-tenant class probabilities, ragged, each row summing to 1.

        Softmax is taken over each tenant's own labels only. Single-logit heads
        (a binary scikit-learn classifier keeps one row) get a sigmoid instead,
        since a softmax over one class is the constant 1.
        """
        return [
            _sigmoid(row) if len(row) == 1 else _softmax(row)
            for row in self.logits(texts, tenants)
        ]

    def predict(self, texts: Sequence[str], tenants: Sequence[str]) -> list[int]:
        """Highest-scoring label index per text, in each tenant's own label space.

        A single-logit head returns 1 above zero and 0 below, matching the
        binary convention its probabilities follow.
        """
        out = []
        for row in self.logits(texts, tenants):
            if len(row) == 1:
                out.append(int(row[0] > 0.0))
            else:
                out.append(max(range(len(row)), key=row.__getitem__))
        return out

    def encode(self, texts: Sequence[str], tenants: Sequence[str]) -> Tensor:
        """Pooled ``(N, H)`` representations, skipping the heads.

        The features the heads are applied to — useful for fitting a new
        tenant's head on the adapted encoder without a second forward pass.
        """
        return self._encode(texts, tenants)


def _softmax(row: list[float]) -> list[float]:
    # Shift by the max before exponentiating: these are raw logits and a large
    # positive one overflows exp() to inf, turning the row into NaN.
    top = max(row)
    exps = [math.exp(v - top) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def _sigmoid(row: list[float]) -> list[float]:
    return [1.0 / (1.0 + math.exp(-row[0]))]
