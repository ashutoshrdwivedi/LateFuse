"""Synthetic data generation for benchmarking without real adapter files."""

import torch

from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore


def make_synthetic_adapters(store: AdapterStore, n: int, seed: int = 42) -> None:
    """Add n randomly-initialised LoRA adapters to the store.

    Adapter IDs are "adapter_0", "adapter_1", ..., "adapter_{n-1}".
    A matrices are drawn from N(0, 0.02); B matrices are zero (so delta=0,
    safe for correctness checks).

    Args:
        store: AdapterStore to populate.
        n:     Number of adapters to generate.
        seed:  RNG seed for reproducibility.
    """
    for i in range(n):
        store.load_synthetic(f"adapter_{i}", seed=seed + i)


def make_synthetic_inputs(config: LoraServingConfig, batch_size: int) -> dict:
    """Generate random tokenized inputs for a batch.

    Returns a dict with:
        input_ids:      (batch_size, max_seq_len) int64
        attention_mask: (batch_size, max_seq_len) int64  (all ones — no padding)
    """
    return {
        "input_ids": torch.randint(
            low=1,
            high=30000,  # safe for most vocab sizes
            size=(batch_size, config.max_seq_len),
            device=config.device,
        ),
        "attention_mask": torch.ones(
            batch_size, config.max_seq_len,
            dtype=torch.long,
            device=config.device,
        ),
    }


def make_synthetic_lr_weights(
    config: LoraServingConfig,
    num_labels: int = 10,
) -> tuple:
    """Generate random LR coef and intercept tensors for one tenant.

    Returns:
        coef:      (1, num_labels, hidden_size)
        intercept: (1, num_labels)
    """
    coef = torch.randn(1, num_labels, config.hidden_size, dtype=config.dtype, device=config.device)
    intercept = torch.zeros(1, num_labels, dtype=config.dtype, device=config.device)
    return coef, intercept
