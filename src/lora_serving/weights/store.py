from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from lora_serving.config import LoraServingConfig

# Maps (layer_idx, module_name) → tensor key in the adapter .bin file.
# Callers provide this to make loading model-agnostic.
# Example for SetFit/adapters-library format:
#   key_fn = lambda i, mod: f"encoder.layer.{i}.attention.self.{mod}.loras.setfit_<id>.lora_A"
KeyResolverFn = Callable[[int, str], tuple[str, str]]  # returns (key_A, key_B)


class LoraWeight:
    """Stores the A and B low-rank matrices for one adapter, per target module per layer.

    Shapes (one entry per target module):
        wa[module]: (num_layers, hidden_size, lora_rank)  — A matrix (shrink)
        wb[module]: (num_layers, lora_rank, hidden_size)  — B matrix (expand)
    """

    def __init__(self, config: LoraServingConfig):
        H, R, L = config.hidden_size, config.lora_rank, config.num_layers
        self.wa: dict[str, Tensor] = {
            m: torch.zeros(L, H, R, dtype=config.dtype, device=config.device)
            for m in config.target_modules
        }
        self.wb: dict[str, Tensor] = {
            m: torch.zeros(L, R, H, dtype=config.dtype, device=config.device)
            for m in config.target_modules
        }

    def copy_from_tensors(
        self,
        a_by_module: dict[str, Tensor],
        b_by_module: dict[str, Tensor],
        scale: float = 1.0,
    ) -> None:
        """Load from column-major tensors (as stored in adapter .bin files).

        Args:
            a_by_module: module → (num_layers, lora_rank, hidden_size)  — A in column-major order
            b_by_module: module → (num_layers, hidden_size, lora_rank)  — B in column-major order
            scale:       LoRA scaling factor alpha/rank, folded into B so the
                         forward pass computes delta = (alpha/r)·B·A·x without
                         a per-call multiply. Use 1.0 when alpha == rank.
        """
        for m, a in a_by_module.items():
            self.wa[m].copy_(a.to(self.wa[m].device, self.wa[m].dtype).transpose(1, 2))
        for m, b in b_by_module.items():
            self.wb[m].copy_(b.to(self.wb[m].device, self.wb[m].dtype).transpose(1, 2))
            if scale != 1.0:
                self.wb[m].mul_(scale)


class AdapterStore:
    """Pre-loaded GPU cache of tenant LoRA adapters.

    Load all adapters at startup via load_from_file() or load_synthetic(),
    then retrieve them by ID during inference via get().

    Example:
        store = AdapterStore(config)
        store.load_from_file("tenant_42", "/path/to/adapter.bin", my_key_fn)
        store.load_synthetic("tenant_99")  # random weights for benchmarking
        weight = store.get("tenant_42")
    """

    def __init__(self, config: LoraServingConfig):
        self._config = config
        self._store: dict[str, LoraWeight] = {}

    def load_from_file(
        self,
        adapter_id: str,
        path: str,
        key_fn: KeyResolverFn,
        lora_alpha: float | None = None,
    ) -> None:
        """Load a LoRA adapter from a .bin file and cache it on GPU.

        Args:
            adapter_id: Unique identifier for this adapter (used in get()).
            path:       Path to the pytorch_adapter.bin file.
            key_fn:     Maps (layer_idx, module_name) → (key_A, key_B) in the state dict.
            lora_alpha: The adapter's lora_alpha hyperparameter. The serving
                        forward pass applies no scaling factor, so the standard
                        delta = (alpha/r)·B·A·x is honored by folding alpha/r
                        into B at load time. None assumes alpha == lora_rank
                        (scale 1.0) — only correct if the adapter was trained
                        that way; pass the real alpha for adapters where it
                        differs from the rank.
        """
        cfg = self._config
        scale = 1.0 if lora_alpha is None else lora_alpha / cfg.lora_rank
        state_dict = torch.load(path, map_location=cfg.device, weights_only=True)
        weight = LoraWeight(cfg)

        a_by_module: dict[str, Tensor] = {}
        b_by_module: dict[str, Tensor] = {}
        for module in cfg.target_modules:
            a_tensors, b_tensors = [], []
            for i in range(cfg.num_layers):
                key_a, key_b = key_fn(i, module)
                a_tensors.append(state_dict[key_a])
                b_tensors.append(state_dict[key_b])
            a_by_module[module] = torch.stack(a_tensors)
            b_by_module[module] = torch.stack(b_tensors)

        weight.copy_from_tensors(a_by_module, b_by_module, scale=scale)
        self._store[adapter_id] = weight

    def load_synthetic(self, adapter_id: str, seed: int | None = None) -> None:
        """Load a random LoRA adapter — useful for benchmarking without real files."""
        cfg = self._config
        if seed is not None:
            torch.manual_seed(seed)
        weight = LoraWeight(cfg)
        for m in cfg.target_modules:
            torch.nn.init.normal_(weight.wa[m], std=0.02)
            torch.nn.init.zeros_(weight.wb[m])  # B=0 means delta=0, safe for correctness tests
        self._store[adapter_id] = weight

    def get(self, adapter_id: str) -> LoraWeight:
        """Retrieve a pre-loaded adapter by ID."""
        if adapter_id not in self._store:
            raise KeyError(f"Adapter '{adapter_id}' not found. Call load_from_file() or load_synthetic() first.")
        return self._store[adapter_id]

    def evict(self, adapter_id: str) -> None:
        """Remove a resident adapter so its GPU tensors can be released."""
        if adapter_id not in self._store:
            raise KeyError(f"Adapter '{adapter_id}' is not resident.")
        del self._store[adapter_id]

    def adapter_ids(self) -> list[str]:
        """All loaded adapter IDs in insertion order.

        The order is stable for the lifetime of the store and defines the row
        index each adapter occupies in a packed tensor (see
        :class:`~lora_serving.weights.batch.IndexSelectBatchAssembler`).
        """
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def memory_bytes(self) -> int:
        """Total GPU memory used by all cached adapters."""
        total = 0
        for w in self._store.values():
            for m in w.wa:
                total += w.wa[m].nbytes + w.wb[m].nbytes
        return total

    def memory_gb(self) -> float:
        return self.memory_bytes() / 1e9
