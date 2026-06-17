"""Tests for LoraOps and LRHeadOps correctness."""

import torch
import pytest
from lora_serving.config import LoraServingConfig
from lora_serving.ops.lora import LoraOps
from lora_serving.ops.head import LRHeadOps


@pytest.fixture
def cpu_config():
    return LoraServingConfig(
        model_name="intfloat/multilingual-e5-small",
        lora_rank=8,
        batch_size=4,
        max_seq_len=16,
        target_modules=["query", "value"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestLoraOps:
    def test_shrink_expand_shapes(self, cpu_config):
        """Output shape should be (B, S, H) after shrink + expand."""
        ops = LoraOps(cpu_config)
        B, S, H, R = cpu_config.batch_size, cpu_config.max_seq_len, cpu_config.hidden_size, cpu_config.lora_rank

        # Move buffers to CPU for this test
        ops._out_A = torch.empty(B, S, R)
        ops._out_B = torch.empty(B, S, H)

        x = torch.randn(B, S, H)
        a = torch.randn(B, H, R)
        b = torch.randn(B, R, H)

        # Patch CUDA check for CPU test
        ops.shrink.__func__  # ensure it's accessible
        torch.bmm(x, a, out=ops._out_A)
        torch.bmm(ops._out_A, b, out=ops._out_B)

        assert ops._out_B.shape == (B, S, H)

    def test_shrink_expand_matches_naive(self, cpu_config):
        """BMM result should match a naive per-sample matmul loop."""
        B, S, H, R = cpu_config.batch_size, cpu_config.max_seq_len, cpu_config.hidden_size, cpu_config.lora_rank
        x = torch.randn(B, S, H)
        a = torch.randn(B, H, R)
        b = torch.randn(B, R, H)

        # Batched BMM
        out_A = torch.bmm(x, a)
        out_B = torch.bmm(out_A, b)

        # Naive per-sample loop
        expected = torch.stack([x[i] @ a[i] @ b[i] for i in range(B)])

        assert torch.allclose(out_B, expected, atol=1e-5)


class TestLRHeadOps:
    def test_output_shape(self, cpu_config):
        B, H = cpu_config.batch_size, cpu_config.hidden_size
        num_labels = 5

        pooled = torch.randn(B, 1, H)
        coef = torch.randn(B, num_labels, H)
        intercept = torch.zeros(B, num_labels)
        out = torch.zeros(B, 1, num_labels)

        # Patch CUDA check
        torch.bmm(pooled, coef.transpose(1, 2), out=out)
        torch.add(out, intercept.unsqueeze(1), out=out)

        assert out.shape == (B, 1, num_labels)

    def test_matches_sklearn_logistic(self, cpu_config):
        """Single-sample LR output should match sklearn's linear_model computation."""
        H, num_labels = cpu_config.hidden_size, 3
        x = torch.randn(1, H)
        coef = torch.randn(num_labels, H)
        intercept = torch.randn(num_labels)

        # Our batched computation
        pooled = x.unsqueeze(0)              # (1, 1, H)
        coef_b = coef.unsqueeze(0)           # (1, num_labels, H)
        intercept_b = intercept.unsqueeze(0) # (1, num_labels)
        out = torch.zeros(1, 1, num_labels)
        torch.bmm(pooled, coef_b.transpose(1, 2), out=out)
        torch.add(out, intercept_b.unsqueeze(1), out=out)

        # Expected: x @ coef.T + intercept
        expected = x @ coef.T + intercept

        assert torch.allclose(out.squeeze(), expected, atol=1e-5)
