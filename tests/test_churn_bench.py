"""Tests for the registration-churn microbenchmark (lora_serving.benchmark.churn).

The point of the benchmark is that the timed path is the *production* one, so
the tests that matter are the ones establishing that the corpus files really do
ingest through ``AdapterStore.load_from_file`` and that the staged arm produces
bit-identical weights to the plain arm.  If either of those were false the
measured comparison would be between two different things.
"""

import numpy as np
import pytest
import torch

from lora_serving.benchmark.churn import (
    Registrar,
    _zipf_probabilities,
    build_corpus,
    ci95,
    corpus_key_fn,
)
from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore, LoraWeight


@pytest.fixture
def config():
    return LoraServingConfig(
        model_name="intfloat/multilingual-e5-small",
        lora_rank=8,
        batch_size=4,
        max_seq_len=16,
        target_modules=["query", "value"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestCorpus:
    def test_writes_files_with_expected_geometry(self, config, tmp_path):
        paths = build_corpus(config, tmp_path, 3)
        assert len(paths) == 3
        assert all(p.exists() for p in paths)

        sd = torch.load(paths[0], map_location="cpu", weights_only=True)
        H, R, L = config.hidden_size, config.lora_rank, config.num_layers
        assert len(sd) == 2 * L * len(config.target_modules)
        for layer in range(L):
            for module in config.target_modules:
                key_a, key_b = corpus_key_fn(layer, module)
                # Column-major on disk: A is (r, H), B is (H, r). load_from_file
                # transposes on ingest, so getting this backwards would be
                # caught here rather than as a silent shape error at run time.
                assert sd[key_a].shape == (R, H)
                assert sd[key_b].shape == (H, R)

    def test_reuses_existing_files(self, config, tmp_path):
        first = build_corpus(config, tmp_path, 2)
        mtimes = [p.stat().st_mtime_ns for p in first]
        second = build_corpus(config, tmp_path, 2)
        assert [p.stat().st_mtime_ns for p in second] == mtimes

    def test_file_ingests_through_production_path(self, config, tmp_path):
        """The corpus must load via load_from_file -- the path the server uses."""
        paths = build_corpus(config, tmp_path, 1)
        store = AdapterStore(config)
        store.load_from_file("tenant_0", str(paths[0]), corpus_key_fn)

        w = store.get("tenant_0")
        H, R, L = config.hidden_size, config.lora_rank, config.num_layers
        for module in config.target_modules:
            assert w.wa[module].shape == (L, H, R)
            assert w.wb[module].shape == (L, R, H)

        # And the values are the on-disk ones transposed, not zeros.
        sd = torch.load(paths[0], map_location="cpu", weights_only=True)
        key_a, _ = corpus_key_fn(0, config.target_modules[0])
        torch.testing.assert_close(
            w.wa[config.target_modules[0]][0], sd[key_a].T.to(config.dtype)
        )


class TestStoreInsert:
    def test_insert_makes_weight_resident(self, config):
        store = AdapterStore(config)
        store.insert("tenant_0", LoraWeight(config))
        assert len(store) == 1
        assert "tenant_0" in store.adapter_ids()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
class TestPinnedArmEquivalence:
    def test_staged_registration_matches_plain(self, tmp_path):
        """file and file_pinned must yield identical weights, or the arms differ."""
        cfg = LoraServingConfig(
            model_name="intfloat/multilingual-e5-small",
            lora_rank=8,
            batch_size=4,
            max_seq_len=16,
            target_modules=["query", "value"],
            device=torch.device("cuda:0"),
            dtype=torch.float16,
        )
        paths = build_corpus(cfg, tmp_path, 1)

        plain = AdapterStore(cfg)
        plain.load_from_file("t", str(paths[0]), corpus_key_fn)

        staged_store = AdapterStore(cfg)
        staged = Registrar(staged_store, cfg, paths, "file_pinned", 4, {}, {})
        staged.register("t", 0)
        torch.cuda.synchronize()

        for module in cfg.target_modules:
            torch.testing.assert_close(
                plain.get("t").wa[module], staged_store.get("t").wa[module]
            )
            torch.testing.assert_close(
                plain.get("t").wb[module], staged_store.get("t").wb[module]
            )


class TestZipf:
    def test_normalized_and_decreasing(self):
        p = _zipf_probabilities(100, 1.1)
        assert p.shape == (100,)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(np.diff(p) < 0)

    def test_alpha_controls_skew(self):
        flat = _zipf_probabilities(100, 0.5)
        steep = _zipf_probabilities(100, 2.0)
        assert steep[0] > flat[0]


class TestCI:
    def test_single_value_has_no_interval(self):
        mean, half = ci95([4.0])
        assert mean == 4.0
        assert np.isnan(half)

    def test_identical_values_give_zero_width(self):
        mean, half = ci95([2.0, 2.0, 2.0])
        assert mean == 2.0
        assert half == 0.0

    def test_interval_widens_with_spread(self):
        _, tight = ci95([1.0, 1.1, 0.9])
        _, wide = ci95([1.0, 5.0, -3.0])
        assert wide > tight
