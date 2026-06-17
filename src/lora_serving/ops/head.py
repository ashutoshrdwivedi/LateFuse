import torch
from torch import Tensor


class LRHeadOps:
    """Batched logistic regression inference for a mixed-tenant batch.

    Each sample in the batch has its own LR head (different coef/intercept).
    All heads are applied in a single BMM — no per-sample loops.

    Usage:
        output = LRHeadOps.predict_proba(pooled, coef, intercept, out)
    """

    @staticmethod
    def predict_proba(
        pooled: Tensor,
        coef: Tensor,
        intercept: Tensor,
        out: Tensor,
    ) -> Tensor:
        """Compute logistic regression scores for a batch of samples.

        Args:
            pooled:    (B, 1, H)          — pooled encoder output (CLS token)
            coef:      (B, max_labels, H) — per-sample LR weight matrix
            intercept: (B, max_labels)    — per-sample LR bias
            out:       (B, 1, max_labels) — pre-allocated output buffer

        Returns:
            out: filled with raw logit scores (B, 1, max_labels)

        Note:
            Tenants with fewer than max_labels classes have zero-padded coef/intercept.
            Callers should slice [:num_labels] per sample before applying softmax.
        """
        assert pooled.is_cuda, "pooled must be on GPU"
        assert coef.is_cuda, "coef must be on GPU"
        torch.bmm(pooled, coef.transpose(1, 2), out=out)
        torch.add(out, intercept.unsqueeze(1), out=out)
        return out
