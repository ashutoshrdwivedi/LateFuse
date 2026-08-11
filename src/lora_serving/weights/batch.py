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


class IndexSelectBatchAssembler:
    """Device-resident batch assembler: one ``index_select`` per target module.

    The baseline :class:`BatchAssembler` gathers LoRA weights with a Python loop
    over samples × layers × modules — O(B·L·|M|) iterations building per-layer
    lists of ``(1,H,R)``/``(1,R,H)`` views that the encoder later ``torch.cat``s.
    That host-side loop is the dominant per-request overhead and the source of
    the tail-latency spikes (Finding 5 in the paper): the work scales with the
    batch, and the Python/GC churn is unpredictable.

    This assembler removes the loop. At construction it packs the whole
    :class:`AdapterStore` into one contiguous tensor per module::

        wa[module]: (N, L, H, R)        wb[module]: (N, L, R, H)

    where ``N`` is the number of cached adapters. Assembling a batch is then
    ``|M|`` ``index_select`` calls (one per module), each gathering **all L
    layers for the whole batch** in a single device-resident kernel. The only
    remaining Python work is the O(B) adapter-ID -> row-index lookup, which is
    independent of L and |M|. Everything is pure PyTorch, so the gather is also
    amenable to ``torch.compile`` fusion (see
    ``benchmarks/profiling/assembly_bench.py``).

    The packed tensors hold the same bytes as the per-adapter store, just in one
    allocation instead of N. The store must be fully loaded before construction;
    adapters added afterwards are not reflected (matches the pre-loaded serving
    design where all tenants are resident at startup).
    """

    def __init__(self, store: AdapterStore, config: LoraServingConfig):
        self._config = config
        ids = store.adapter_ids()
        self._row_of: dict[str, int] = {aid: i for i, aid in enumerate(ids)}
        # Pack each module's per-adapter slices into one contiguous tensor.
        # torch.stack copies into a fresh (N, L, H, R) / (N, L, R, H) buffer;
        # the contiguous leading adapter axis is what index_select gathers over.
        self.wa: dict[str, Tensor] = {
            m: torch.stack([store.get(aid).wa[m] for aid in ids])
            for m in config.target_modules
        }
        self.wb: dict[str, Tensor] = {
            m: torch.stack([store.get(aid).wb[m] for aid in ids])
            for m in config.target_modules
        }

    def batch_index(self, adapter_ids: list[str]) -> Tensor:
        """Map adapter IDs to a ``(B,)`` int64 row-index tensor on the store device.

        This is the sole remaining host-side loop: O(B), independent of layer
        count and target-module count. It is deliberately kept outside the
        gather so that ``torch.compile`` (which would graph-break on the dict
        lookup) can still fuse the tensor ops in :meth:`assemble_lora`.
        """
        return torch.tensor(
            [self._row_of[aid] for aid in adapter_ids],
            dtype=torch.long,
            device=self._config.device,
        )

    def assemble_lora(
        self, adapter_ids: list[str]
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Gather batched A/B for every module via ``index_select``.

        Args:
            adapter_ids: One adapter ID per sample in the batch.

        Returns:
            module -> (A, B) where ``A`` is ``(B, L, H, R)`` and ``B`` is
            ``(B, L, R, H)``. Slice ``A[:, layer_idx]`` -> ``(B, H, R)`` for that
            layer's shrink, ``B[:, layer_idx]`` -> ``(B, R, H)`` for its expand.
        """
        idx = self.batch_index(adapter_ids)
        return {
            m: (self.wa[m].index_select(0, idx), self.wb[m].index_select(0, idx))
            for m in self._config.target_modules
        }

    def to_layerwise(self, adapter_ids: list[str]) -> list[LayerwiseBatchedWeights]:
        """Assemble into the per-layer structure the existing encoder consumes.

        Drop-in replacement for :meth:`BatchAssembler.assemble_lora`: each
        module's batched ``(B, H, R)`` layer slice is wrapped in a single-element
        list so the encoder's ``torch.cat(..., dim=0)`` is a no-op concatenation
        of one tensor. The wrapping loop is O(L·|M|) over views — the O(B) factor
        of the baseline's append loop is gone.
        """
        batched = self.assemble_lora(adapter_ids)
        layers = [LayerwiseBatchedWeights() for _ in range(self._config.num_layers)]
        for m, (a, b) in batched.items():
            for layer_idx in range(self._config.num_layers):
                layers[layer_idx].a[m] = [a[:, layer_idx]]
                layers[layer_idx].b[m] = [b[:, layer_idx]]
        return layers
