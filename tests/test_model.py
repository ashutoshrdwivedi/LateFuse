"""End-to-end model tests: forward pass shape and correctness."""

import torch
import pytest
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.store import AdapterStore
from lora_serving.weights.batch import BatchAssembler
from lora_serving.benchmark.synthetic import make_synthetic_lr_weights


# Skip all tests in this module if no GPU available — model tests require GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Model tests require a CUDA GPU"
)


@pytest.fixture(scope="module")
def serving_config():
    return LoraServingConfig(
        model_name="intfloat/multilingual-e5-small",
        lora_rank=8,
        batch_size=4,
        max_seq_len=32,
        target_modules=["query", "value"],
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def model(serving_config):
    m = EncoderWithLora.from_pretrained_serving(serving_config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def store(serving_config):
    s = AdapterStore(serving_config)
    for i in range(8):
        s.load_synthetic(f"adapter_{i}", seed=i)
    return s


class TestEncoderWithLora:
    def test_forward_output_shape(self, serving_config, model, store):
        """Output should be (B, 1, num_labels)."""
        B = serving_config.batch_size
        num_labels = 5
        device = serving_config.device

        assembler = BatchAssembler(store, serving_config)
        ids = [f"adapter_{i % 8}" for i in range(B)]
        coefs = [make_synthetic_lr_weights(serving_config, num_labels)[0] for _ in ids]
        intercepts = [make_synthetic_lr_weights(serving_config, num_labels)[1] for _ in ids]

        lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
        output_lr = torch.zeros(B, 1, num_labels, dtype=serving_config.dtype, device=device)

        input_ids = torch.randint(1, 30000, (B, serving_config.max_seq_len), device=device)
        attention_mask = torch.ones(B, serving_config.max_seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids, attention_mask, lora_w, lr_w, output_lr)

        assert out.shape == (B, 1, num_labels)

    def test_forward_is_deterministic(self, serving_config, model, store):
        """The full model pipeline (encoder + pooler + LR head) is deterministic in eval mode.

        Two calls with identical inputs must produce bit-identical outputs. Catches
        accidental state leaks across forward calls (e.g., a pre-allocated LoraOps
        buffer being read before being written, or dropout left enabled).

        The stronger claim — "zero-LoRA output equals the unwrapped base HF model" —
        is verified in test_peft_equivalence.py::test_zero_lora_matches_base_hf.
        """
        B = serving_config.batch_size
        num_labels = 3
        device = serving_config.device

        # Synthetic adapters have B=0 by default — delta is zero
        assembler = BatchAssembler(store, serving_config)
        ids = ["adapter_0"] * B
        coefs = [make_synthetic_lr_weights(serving_config, num_labels)[0] for _ in ids]
        intercepts = [make_synthetic_lr_weights(serving_config, num_labels)[1] for _ in ids]

        lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
        output_lr = torch.zeros(B, 1, num_labels, dtype=serving_config.dtype, device=device)

        input_ids = torch.randint(1, 30000, (B, serving_config.max_seq_len), device=device)
        attention_mask = torch.ones(B, serving_config.max_seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            out1 = model(input_ids, attention_mask, lora_w, lr_w, output_lr.clone())
            out2 = model(input_ids, attention_mask, lora_w, lr_w, output_lr.clone())

        assert torch.allclose(out1, out2, atol=1e-6)
