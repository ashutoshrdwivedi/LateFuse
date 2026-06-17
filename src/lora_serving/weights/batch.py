from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore


class LayerwiseBatchedWeights:
    """LoRA weights for all samples in a batch, for one transformer layer.

    Each list contains one tensor per sample (unsqueezed to (1, H, R) or (1, R, H)).
    Before passing to LoraOps, call torch.cat() on each list to get (B, H, R).
    """

    def __init__(self):
        # Per target_module: list of (1, H, R) A tensors and (1, R, H) B tensors
        self.a: dict[str, list[Tensor]] = {}  # module_name → list of A tensors
        self.b: dict[str, list[Tensor]] = {}  # module_name → list of B tensors


class BatchedLRWeights:
    """Logistic regression weights for all samples in a batch, zero-padded to max_labels."""

    coef: Tensor       # (B, max_labels, H)
    intercept: Tensor  # (B, max_labels)
    num_labels: list[int]  # actual label count per sample (before padding)


class BatchAssembler:
    """Assembles per-sample adapter weights into batch tensors for a single forward pass.

    Usage:
        assembler = BatchAssembler(store, config)
        lora_weights, lr_weights = assembler.assemble(adapter_ids, lr_store)
    """

    def __init__(self, store: AdapterStore, config: LoraServingConfig):
        self._store = store
        self._config = config

    def assemble_lora(
        self,
        adapter_ids: list[str],
    ) -> list[LayerwiseBatchedWeights]:
        """Stack per-sample LoRA weights into batch tensors.

        Args:
            adapter_ids: One adapter ID per sample in the batch.

        Returns:
            List of LayerwiseBatchedWeights, one per transformer layer.
        """
        cfg = self._config
        lora_weights = [LayerwiseBatchedWeights() for _ in range(cfg.num_layers)]

        for module in cfg.target_modules:
            for layer_idx in range(cfg.num_layers):
                lora_weights[layer_idx].a[module] = []
                lora_weights[layer_idx].b[module] = []

        for adapter_id in adapter_ids:
            weight = self._store.get(adapter_id)
            for layer_idx in range(cfg.num_layers):
                for module in cfg.target_modules:
                    lora_weights[layer_idx].a[module].append(weight.wa[module][layer_idx].unsqueeze(0))
                    lora_weights[layer_idx].b[module].append(weight.wb[module][layer_idx].unsqueeze(0))

        return lora_weights

    def assemble_lr(
        self,
        lr_coefs: list[Tensor],
        lr_intercepts: list[Tensor],
    ) -> BatchedLRWeights:
        """Stack per-sample LR head weights into a single padded batch tensor.

        Args:
            lr_coefs:      Per-sample LR coef tensors, shape (1, num_labels, H).
            lr_intercepts: Per-sample LR intercept tensors, shape (1, num_labels).

        Returns:
            BatchedLRWeights with zero-padded coef/intercept.
        """
        num_labels = [c.shape[1] for c in lr_coefs]
        max_labels = max(num_labels)

        padded_coefs = [
            F.pad(c, (0, 0, 0, max_labels - c.shape[1])) for c in lr_coefs
        ]
        padded_intercepts = [
            F.pad(ic, (0, max_labels - ic.shape[1])) for ic in lr_intercepts
        ]

        lr_weights = BatchedLRWeights()
        lr_weights.coef = torch.cat(padded_coefs, dim=0)           # (B, max_labels, H)
        lr_weights.intercept = torch.cat(padded_intercepts, dim=0)  # (B, max_labels)
        lr_weights.num_labels = num_labels

        return lr_weights

    def assemble(
        self,
        adapter_ids: list[str],
        lr_coefs: list[Tensor],
        lr_intercepts: list[Tensor],
    ) -> tuple[list[LayerwiseBatchedWeights], BatchedLRWeights]:
        """Stack per-sample adapter weights into batch tensors.

        Convenience method that calls :meth:`assemble_lora` and
        :meth:`assemble_lr` together.

        Args:
            adapter_ids:   One adapter ID per sample in the batch.
            lr_coefs:      Per-sample LR coef tensors, shape (1, num_labels, H).
            lr_intercepts: Per-sample LR intercept tensors, shape (1, num_labels).

        Returns:
            lora_weights: List of LayerwiseBatchedWeights, one per transformer layer.
            lr_weights:   BatchedLRWeights with zero-padded coef/intercept.
        """
        return self.assemble_lora(adapter_ids), self.assemble_lr(lr_coefs, lr_intercepts)
