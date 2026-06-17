"""Numerical equivalence against scikit-learn's LogisticRegression.

SetFit's default classification head is sklearn LogisticRegression. This test
verifies that our batched LRHeadOps.predict_proba produces the same raw logits
as sklearn's decision_function for the same coef/intercept weights.

Two scenarios:
  1. Single-tenant — all samples share one LR head.
  2. Multi-tenant  — each sample has a different LR head (different num_labels),
     exercising zero-padding in assemble_lr.

Skipped if CUDA or scikit-learn is unavailable.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.weights.batch import BatchAssembler, BatchedLRWeights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LR head equivalence tests require CUDA",
)

sklearn = pytest.importorskip(
    "sklearn", reason="scikit-learn not installed (install with `uv add --dev scikit-learn`)"
)
from sklearn.linear_model import LogisticRegression
import numpy as np


MODEL_NAME = "intfloat/multilingual-e5-small"
LORA_RANK = 8
BATCH_SIZE = 4
SEQ_LEN = 16
ATOL = 1e-5
RTOL = 1e-5


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def serving_config() -> LoraServingConfig:
    return LoraServingConfig(
        model_name=MODEL_NAME,
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        target_modules=["query", "value"],
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )


# ---------- helpers ----------

def _make_sklearn_lr(
    num_classes: int,
    hidden_size: int,
    seed: int,
) -> LogisticRegression:
    """Create a LogisticRegression with random coef/intercept (no training).

    sklearn requires classes_ and certain attributes to call decision_function
    without fitting. We set them directly.
    """
    rng = np.random.default_rng(seed)
    lr = LogisticRegression()
    lr.coef_ = rng.standard_normal((num_classes, hidden_size)).astype(np.float32) * 0.02
    lr.intercept_ = rng.standard_normal((num_classes,)).astype(np.float32) * 0.01
    lr.classes_ = np.arange(num_classes)
    return lr


def _sklearn_to_tensors(
    lr: LogisticRegression,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Convert sklearn LR weights to our per-sample tensor format.

    Returns:
        coef:      (1, num_labels, H)
        intercept: (1, num_labels)
    """
    coef = torch.from_numpy(lr.coef_).to(device=device, dtype=dtype).unsqueeze(0)
    intercept = torch.from_numpy(lr.intercept_).to(device=device, dtype=dtype).unsqueeze(0)
    return coef, intercept


def _sklearn_decision_function(
    lr: LogisticRegression,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute raw logits the same way sklearn does: X @ coef.T + intercept.

    We reimplement this directly rather than calling lr.decision_function()
    to avoid sklearn's internal input validation (which expects a fitted model
    with full metadata).
    """
    return embeddings @ lr.coef_.T + lr.intercept_


# ---------- tests ----------

class TestLRHeadEquivalence:
    """Verify LRHeadOps.predict_proba matches sklearn LogisticRegression logits."""

    def test_single_tenant(self, serving_config):
        """All samples share one LR head — our batched output should match sklearn."""
        H = serving_config.hidden_size
        num_labels = 5
        device = serving_config.device
        dtype = serving_config.dtype

        # Build sklearn LR with known weights
        lr = _make_sklearn_lr(num_labels, H, seed=42)

        # Random pooled embeddings (as if they came from the encoder)
        torch.manual_seed(0)
        pooled_torch = torch.randn(BATCH_SIZE, H, device=device, dtype=dtype)
        pooled_np = pooled_torch.cpu().numpy()

        # sklearn logits: (B, num_labels)
        expected = _sklearn_decision_function(lr, pooled_np)
        expected_tensor = torch.from_numpy(expected).to(device=device, dtype=dtype)

        # Our implementation: assemble_lr then LRHeadOps
        coef, intercept = _sklearn_to_tensors(lr, device, dtype)
        # Repeat for all samples in the batch (single-tenant)
        lr_coefs = [coef] * BATCH_SIZE
        lr_intercepts = [intercept] * BATCH_SIZE

        # Use a dummy store — assemble_lr doesn't need it
        from lora_serving.weights.store import AdapterStore
        store = AdapterStore(serving_config)
        assembler = BatchAssembler(store, serving_config)
        lr_weights = assembler.assemble_lr(lr_coefs, lr_intercepts)

        out = torch.zeros(BATCH_SIZE, 1, num_labels, device=device, dtype=dtype)
        LRHeadOps.predict_proba(
            pooled_torch.unsqueeze(1),  # (B, 1, H)
            lr_weights.coef,
            lr_weights.intercept,
            out,
        )
        # out is (B, 1, num_labels) → squeeze to (B, num_labels)
        our_logits = out.squeeze(1)

        assert torch.allclose(our_logits, expected_tensor, atol=ATOL, rtol=RTOL), (
            f"max abs diff = {(our_logits - expected_tensor).abs().max().item()}"
        )

    def test_multi_tenant_with_padding(self, serving_config):
        """Each sample has a different LR head with varying num_labels.

        Verifies that zero-padding in assemble_lr doesn't corrupt the real logits.
        """
        H = serving_config.hidden_size
        device = serving_config.device
        dtype = serving_config.dtype

        # Different number of classes per tenant
        labels_per_tenant = [3, 5, 2, 7]
        assert len(labels_per_tenant) == BATCH_SIZE
        max_labels = max(labels_per_tenant)

        # Build one sklearn LR per tenant
        lrs = [_make_sklearn_lr(n, H, seed=100 + i) for i, n in enumerate(labels_per_tenant)]

        # Random pooled embeddings
        torch.manual_seed(1)
        pooled_torch = torch.randn(BATCH_SIZE, H, device=device, dtype=dtype)
        pooled_np = pooled_torch.cpu().numpy()

        # Compute expected logits per-sample via sklearn
        expected_per_sample = [
            _sklearn_decision_function(lr, pooled_np[i:i+1])
            for i, lr in enumerate(lrs)
        ]

        # Assemble via our pipeline
        lr_coefs = []
        lr_intercepts = []
        for lr in lrs:
            coef, intercept = _sklearn_to_tensors(lr, device, dtype)
            lr_coefs.append(coef)
            lr_intercepts.append(intercept)

        from lora_serving.weights.store import AdapterStore
        store = AdapterStore(serving_config)
        assembler = BatchAssembler(store, serving_config)
        lr_weights = assembler.assemble_lr(lr_coefs, lr_intercepts)

        out = torch.zeros(BATCH_SIZE, 1, max_labels, device=device, dtype=dtype)
        LRHeadOps.predict_proba(
            pooled_torch.unsqueeze(1),  # (B, 1, H)
            lr_weights.coef,
            lr_weights.intercept,
            out,
        )
        our_logits = out.squeeze(1)  # (B, max_labels)

        # Check each sample's real logits (up to num_labels), ignoring padding
        for i, (n_labels, expected) in enumerate(zip(labels_per_tenant, expected_per_sample)):
            expected_tensor = torch.from_numpy(expected).to(device=device, dtype=dtype).squeeze(0)
            our_sample = our_logits[i, :n_labels]

            assert torch.allclose(our_sample, expected_tensor, atol=ATOL, rtol=RTOL), (
                f"Sample {i} (num_labels={n_labels}): "
                f"max abs diff = {(our_sample - expected_tensor).abs().max().item()}"
            )

        # Also verify padded entries are just intercept (coef is zero-padded → logit = 0 + 0 = 0
        # for the padded rows, but intercept is also zero-padded so padded logits should be 0)
        for i, n_labels in enumerate(labels_per_tenant):
            if n_labels < max_labels:
                padded = our_logits[i, n_labels:]
                assert torch.allclose(padded, torch.zeros_like(padded), atol=ATOL), (
                    f"Sample {i}: padded logits should be zero, got max={padded.abs().max().item()}"
                )
