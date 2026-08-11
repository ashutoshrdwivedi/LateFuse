"""Tests for AdapterStore and BatchAssembler."""

import torch
import pytest
from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore, LoraWeight
from lora_serving.weights.batch import BatchAssembler, IndexSelectBatchAssembler


@pytest.fixture
def config():
    return LoraServingConfig(
        model_name="intfloat/multilingual-e5-small",
        lora_rank=8,
        batch_size=4,
        max_seq_len=16,
        target_modules=["query", "value"],
        device=torch.device("cpu"),  # these tests don't need GPU.  
        dtype=torch.float32,
    )


class TestAdapterStore:
    def test_load_synthetic(self, config):
        store = AdapterStore(config)
        store.load_synthetic("adapter_0", seed=0)
        assert len(store) == 1

    def test_get_returns_correct_shapes(self, config):
        store = AdapterStore(config)
        store.load_synthetic("adapter_0")
        w = store.get("adapter_0")
        L, H, R = config.num_layers, config.hidden_size, config.lora_rank
        for m in config.target_modules:
            assert w.wa[m].shape == (L, H, R)
            assert w.wb[m].shape == (L, R, H)

    def test_get_unknown_raises(self, config):
        store = AdapterStore(config)
        with pytest.raises(KeyError, match="not found"):
            store.get("nonexistent")

    def test_memory_bytes(self, config):
        store = AdapterStore(config)
        store.load_synthetic("a0")
        store.load_synthetic("a1")
        # Per adapter: M target modules × 2 matrices (wa, wb) × L × H × R × 4 bytes
        L, H, R = config.num_layers, config.hidden_size, config.lora_rank
        M = len(config.target_modules)
        expected = M * 2 * (L * H * R * 4) * 2  # 2 adapters
        assert store.memory_bytes() == expected

    def test_multiple_adapters_are_independent(self, config):
        store = AdapterStore(config)
        store.load_synthetic("a0", seed=0)
        store.load_synthetic("a1", seed=1)
        w0 = store.get("a0")
        w1 = store.get("a1")
        # Different seeds should give different weights
        m = config.target_modules[0]
        assert not torch.equal(w0.wa[m], w1.wa[m])


class TestBatchAssembler:
    def test_assemble_shapes(self, config):
        store = AdapterStore(config)
        for i in range(4):
            store.load_synthetic(f"adapter_{i}")

        assembler = BatchAssembler(store, config)
        ids = [f"adapter_{i}" for i in range(config.batch_size)]

        H, num_labels = config.hidden_size, 5
        lr_coefs = [torch.randn(1, num_labels, H) for _ in ids]
        lr_intercepts = [torch.zeros(1, num_labels) for _ in ids]

        lora_w, lr_w = assembler.assemble(ids, lr_coefs, lr_intercepts)

        assert len(lora_w) == config.num_layers
        for layer in lora_w:
            for module in config.target_modules:
                # Each list should have batch_size tensors of shape (1, H, R)
                assert len(layer.a[module]) == config.batch_size
                assert len(layer.b[module]) == config.batch_size

        B = config.batch_size
        assert lr_w.coef.shape == (B, num_labels, H)
        assert lr_w.intercept.shape == (B, num_labels)

    def test_assemble_pads_variable_labels(self, config):
        """Batch with different num_labels per sample should be padded to max.

        Verifies that requests with different target classification label counts
        (e.g., 3 labels vs 7 labels) are dynamically zero-padded along the class
        dimension to the maximum label size in the batch, enabling uniform stacking,
        while maintaining the original (unpadded) label counts as metadata.
        """
        store = AdapterStore(config)
        for i in range(2):
            store.load_synthetic(f"adapter_{i}")

        assembler = BatchAssembler(store, config)
        ids = ["adapter_0", "adapter_1"]
        H = config.hidden_size

        # Different label counts
        lr_coefs = [torch.randn(1, 3, H), torch.randn(1, 7, H)]
        lr_intercepts = [torch.zeros(1, 3), torch.zeros(1, 7)]

        _, lr_w = assembler.assemble(ids, lr_coefs, lr_intercepts)

        assert lr_w.coef.shape[1] == 7  # max labels
        assert lr_w.num_labels == [3, 7]


class TestIndexSelectBatchAssembler:
    """The index_select assembler must be a drop-in for BatchAssembler: the
    batched weights it produces have to be bit-identical, since the encoder
    forward is fully determined by those tensors."""

    def _store(self, config, n=6):
        store = AdapterStore(config)
        for i in range(n):
            # Distinct seeds so every adapter's A matrix differs — a wrong
            # row-index mapping would then surface as a mismatch.
            store.load_synthetic(f"adapter_{i}", seed=100 + i)
            # B is zero-initialised by load_synthetic; perturb it so wb is also
            # adapter-specific and the B-gather is exercised too.
            torch.nn.init.normal_(store.get(f"adapter_{i}").wb[config.target_modules[0]], std=0.02)
        return store

    def test_packed_shapes(self, config):
        store = self._store(config, n=5)
        asm = IndexSelectBatchAssembler(store, config)
        N, L, H, R = 5, config.num_layers, config.hidden_size, config.lora_rank
        for m in config.target_modules:
            assert asm.wa[m].shape == (N, L, H, R)
            assert asm.wb[m].shape == (N, L, R, H)

    def test_matches_baseline_assemble(self, config):
        store = self._store(config, n=6)
        baseline = BatchAssembler(store, config)
        fast = IndexSelectBatchAssembler(store, config)

        # Repeated and out-of-order IDs: catches any positional assumption.
        ids = ["adapter_3", "adapter_0", "adapter_3", "adapter_5"]

        ref = baseline.assemble_lora(ids)          # list[LayerwiseBatchedWeights]
        got = fast.to_layerwise(ids)               # list[LayerwiseBatchedWeights]

        assert len(got) == len(ref) == config.num_layers
        for layer_idx in range(config.num_layers):
            for m in config.target_modules:
                ref_a = torch.cat(ref[layer_idx].a[m], dim=0)   # (B, H, R)
                ref_b = torch.cat(ref[layer_idx].b[m], dim=0)   # (B, R, H)
                got_a = torch.cat(got[layer_idx].a[m], dim=0)
                got_b = torch.cat(got[layer_idx].b[m], dim=0)
                assert torch.equal(got_a, ref_a)
                assert torch.equal(got_b, ref_b)

    def test_batch_index_maps_ids_to_rows(self, config):
        store = self._store(config, n=4)
        fast = IndexSelectBatchAssembler(store, config)
        idx = fast.batch_index(["adapter_2", "adapter_0", "adapter_2"])
        assert idx.tolist() == [2, 0, 2]
        assert idx.dtype == torch.long

    def test_unknown_id_raises(self, config):
        store = self._store(config, n=2)
        fast = IndexSelectBatchAssembler(store, config)
        with pytest.raises(KeyError):
            fast.batch_index(["nonexistent"])
