"""GPU-resident cache of per-tenant classification heads.

The counterpart to :class:`~latefuse.weights.store.AdapterStore`, and
deliberately shaped like it (``get`` / ``evict`` / ``tenant_ids`` / ``__len__`` /
``memory_bytes``): a tenant is served by an adapter *and* a head, and the two
should be managed the same way.

Until now nothing owned these. The reference server manufactured random heads
per request and the benchmarks used ``make_synthetic_lr_weights`` — fine for
measuring the forward pass, useless for serving a real tenant whose head was
fitted on their own labels. Tenants may carry different label counts; the batch
is zero-padded to the widest head present and each sample's own count travels
with it, so callers never read a padded lane as a class.
"""

from __future__ import annotations

import torch
from torch import Tensor

from latefuse.config import LoraServingConfig
from latefuse.weights.batch import BatchedLRWeights


class LRHead:
    """One tenant's logistic-regression head.

    Attributes:
        coef:      ``(num_labels, H)``
        intercept: ``(num_labels,)``
    """

    __slots__ = ("coef", "intercept")

    def __init__(self, coef: Tensor, intercept: Tensor):
        self.coef = coef
        self.intercept = intercept

    @property
    def num_labels(self) -> int:
        return self.coef.shape[0]


class HeadStore:
    """Pre-loaded per-tenant heads, keyed by the same tenant id as the adapter.

    Example:
        heads = HeadStore(config)
        heads.load("tenant_42", coef, intercept)     # torch or numpy
        heads.load_from_sklearn("tenant_7", clf)     # fitted LogisticRegression
        heads.get("tenant_42").num_labels
    """

    def __init__(self, config: LoraServingConfig):
        self._config = config
        self._store: dict[str, LRHead] = {}

    def load(self, tenant_id: str, coef, intercept) -> None:
        """Cache a head from any array-like ``coef`` / ``intercept``.

        Args:
            tenant_id: Key this head is served under — must match the tenant's
                       adapter id in the :class:`AdapterStore`.
            coef:      ``(num_labels, H)``, or ``(H,)`` for a single output
                       (binary, or a reranker's one logit).
            intercept: ``(num_labels,)``, or a scalar alongside a ``(H,)`` coef.

        Raises:
            ValueError: if the hidden size disagrees with the serving config, or
                        coef and intercept disagree on the label count.
        """
        cfg = self._config
        coef_t = torch.as_tensor(coef, dtype=cfg.dtype, device=cfg.device)
        intercept_t = torch.as_tensor(intercept, dtype=cfg.dtype, device=cfg.device)
        if coef_t.ndim == 1:
            coef_t = coef_t.unsqueeze(0)
        if intercept_t.ndim == 0:
            intercept_t = intercept_t.unsqueeze(0)
        if coef_t.ndim != 2:
            raise ValueError(
                f"coef for '{tenant_id}' must be (num_labels, H) or (H,), got "
                f"{tuple(coef_t.shape)}"
            )
        if coef_t.shape[1] != cfg.hidden_size:
            raise ValueError(
                f"coef for '{tenant_id}' has hidden size {coef_t.shape[1]}, but "
                f"{cfg.model_name} pools to {cfg.hidden_size}. A head fitted on "
                "a different backbone cannot be served by this one."
            )
        if intercept_t.shape != (coef_t.shape[0],):
            raise ValueError(
                f"intercept for '{tenant_id}' has shape "
                f"{tuple(intercept_t.shape)}, expected {(coef_t.shape[0],)} to "
                "match coef"
            )
        self._store[tenant_id] = LRHead(coef_t.contiguous(), intercept_t.contiguous())

    def load_from_sklearn(self, tenant_id: str, classifier) -> None:
        """Cache a head from a fitted scikit-learn linear classifier.

        Reads ``coef_`` / ``intercept_``, which is how SetFit and the accuracy
        benchmarks in this repo produce heads. A binary scikit-learn classifier
        stores a single ``(1, H)`` row for the positive class; that is kept
        as-is, so such a tenant has one logit rather than two.
        """
        for attr in ("coef_", "intercept_"):
            if not hasattr(classifier, attr):
                raise ValueError(
                    f"{type(classifier).__name__} has no '{attr}'; expected a "
                    "fitted linear classifier (e.g. sklearn LogisticRegression)"
                )
        self.load(tenant_id, classifier.coef_, classifier.intercept_)

    def load_from_file(self, tenant_id: str, path: str) -> None:
        """Cache a head from a torch file holding ``{"coef": ..., "intercept": ...}``."""
        state = torch.load(path, map_location=self._config.device, weights_only=True)
        missing = {"coef", "intercept"} - set(state)
        if missing:
            raise KeyError(
                f"{path} is missing {sorted(missing)}; expected a dict with "
                "'coef' and 'intercept'"
            )
        self.load(tenant_id, state["coef"], state["intercept"])

    def assemble(self, tenant_ids: list[str]) -> BatchedLRWeights:
        """Stack one batch of heads, zero-padded to the widest one present.

        Lives here rather than on a batch assembler because it depends only on
        the heads: it must work identically whichever LoRA assembler the
        pipeline is configured with, and ``IndexSelectBatchAssembler`` has no
        head path at all.

        Returns:
            BatchedLRWeights with ``coef`` ``(B, max_labels, H)``, ``intercept``
            ``(B, max_labels)``, and ``num_labels`` carrying each sample's real
            count so callers can slice the padding off before softmax.
        """
        heads = [self.get(t) for t in tenant_ids]
        num_labels = [h.num_labels for h in heads]
        max_labels = max(num_labels)

        coef = torch.zeros(
            len(heads), max_labels, self._config.hidden_size,
            dtype=self._config.dtype, device=self._config.device,
        )
        intercept = torch.zeros(
            len(heads), max_labels,
            dtype=self._config.dtype, device=self._config.device,
        )
        for i, head in enumerate(heads):
            coef[i, : head.num_labels] = head.coef
            intercept[i, : head.num_labels] = head.intercept

        batched = BatchedLRWeights()
        batched.coef = coef
        batched.intercept = intercept
        batched.num_labels = num_labels
        return batched

    def get(self, tenant_id: str) -> LRHead:
        if tenant_id not in self._store:
            raise KeyError(
                f"No head for tenant '{tenant_id}'. Call load(), "
                "load_from_sklearn() or load_from_file() first."
            )
        return self._store[tenant_id]

    def evict(self, tenant_id: str) -> None:
        if tenant_id not in self._store:
            raise KeyError(f"Head for tenant '{tenant_id}' is not resident.")
        del self._store[tenant_id]

    def tenant_ids(self) -> list[str]:
        """All loaded tenant ids, in insertion order."""
        return list(self._store)

    def num_labels(self, tenant_id: str) -> int:
        return self.get(tenant_id).num_labels

    def __contains__(self, tenant_id: object) -> bool:
        return tenant_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def memory_bytes(self) -> int:
        """Total GPU memory held by all cached heads."""
        return sum(h.coef.nbytes + h.intercept.nbytes for h in self._store.values())
