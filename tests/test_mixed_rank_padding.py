"""Tests for the mixed-rank padding path used by the mixed-rank benchmark.

The mixed-rank result rests on one identity: zero-padding a rank-r adapter into
a rank-r_max slot changes the batch's SHAPE but not its VALUE, so a mixed-rank
batch is the same computation as a uniform r_max batch. The benchmark asserts
that on-device before it times anything; these tests pin the same identity on
CPU, where it is exact in fp32 and costs no GPU time to check.

They also cover the two ways the padding can silently go wrong:
  - drawing weights into a strided slice instead of a contiguous tensor, which
    would make "the same adapter at rank r" mean different weights in different
    arms (test_padded_store_matches_native_draw);
  - reusing a batch buffer without re-padding, which leaves the previous
    occupant's high-rank columns behind (test_no_stale_weights_across_batches).
"""

import pytest
import torch

from benchmarks.profiling.mixed_rank_bench import (
    PackedPaddedAssembler,
    PadToMaxAssembler,
    assign_ranks,
    bucket_bytes,
    build_native_buckets,
    build_packed_padded,
    build_padded_store,
    draw_native,
    parse_mix,
)
from lora_serving.config import LoraServingConfig
from lora_serving.weights.batch import IndexSelectBatchAssembler

R_MIN, R_MAX = 4, 16


@pytest.fixture
def config():
    """r_max-wide slots on CPU in fp32 — the padding identity is exact there."""
    return LoraServingConfig(
        model_name="intfloat/multilingual-e5-small",
        lora_rank=R_MAX,
        batch_size=4,
        max_seq_len=16,
        target_modules=["query", "value"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


@pytest.fixture
def mix():
    return parse_mix(f"{R_MIN},{R_MAX}")


class TestPaddingIdentity:
    def test_zero_padding_preserves_the_delta(self):
        """B·A·x is unchanged by zero-padding A's columns and B's rows.

        This is the whole argument in three lines: the padded components of the
        shrink output multiply zero rows of B, so they contribute nothing, while
        the bmm shapes become those of a uniform r_max batch.
        """
        torch.manual_seed(0)
        B, S, H = 4, 16, 32
        x = torch.randn(B, S, H)
        a = torch.randn(B, H, R_MIN)
        b = torch.randn(B, R_MIN, H)

        native = torch.bmm(torch.bmm(x, a), b)

        a_pad = torch.zeros(B, H, R_MAX)
        b_pad = torch.zeros(B, R_MAX, H)
        a_pad[:, :, :R_MIN] = a
        b_pad[:, :R_MIN, :] = b
        shrink_pad = torch.bmm(x, a_pad)
        padded = torch.bmm(shrink_pad, b_pad)

        # The exact part of the claim, and the only part that can be asserted
        # bitwise: the padded columns of the shrink output are identically zero,
        # so the padded rows of B are multiplied by nothing and the r_max bmm
        # sums the same R_MIN products the native one does.
        assert not shrink_pad[:, :, R_MIN:].any()

        # The values themselves are only equal in exact arithmetic. Both bmms
        # reduce over the same K, but the padded one has a 4x wider output, and
        # BLAS picks its blocking from the output shape — so the FMAs are
        # re-associated and the last bits move. Observed worst case here is
        # ~4e-5 relative (a few fp32 ULPs accumulated over a 32-term dot product
        # and a second reduction); asserting torch.equal makes this test a
        # report on the host's GEMM kernel selection rather than on padding.
        # The benchmark's on-device gate takes the same view at EXACTNESS_TOL.
        torch.testing.assert_close(native, padded, rtol=1e-4, atol=1e-5)

    def test_padded_store_matches_native_draw(self, config):
        """A padded adapter holds its native draw, and zeros beyond it.

        Guards the reason draw_native allocates a contiguous tensor instead of
        drawing straight into `wa[:, :, :r]`: a strided view consumes the RNG
        stream in an order PyTorch does not promise to keep stable, so the same
        (adapter, rank) could differ between the padded and native arms and the
        benchmark's exactness gate would be comparing unrelated tensors.
        """
        store = build_padded_store(config, [R_MIN, R_MAX])
        weight = store.get("adapter_0")
        expect_a, expect_b = draw_native(config, R_MIN, seed=42)

        for m in config.target_modules:
            assert torch.equal(weight.wa[m][:, :, :R_MIN], expect_a[m])
            assert torch.equal(weight.wb[m][:, :R_MIN, :], expect_b[m])
            assert not weight.wa[m][:, :, R_MIN:].any()
            assert not weight.wb[m][:, R_MIN:, :].any()

    def test_full_rank_adapter_is_not_padded(self, config):
        """An r_max tenant fills its slot — nothing is zeroed."""
        store = build_padded_store(config, [R_MAX])
        weight = store.get("adapter_0")
        for m in config.target_modules:
            assert weight.wa[m].shape[-1] == R_MAX
            assert weight.wa[m].abs().sum() > 0


class TestPackedPaddedAssembler:
    def test_packed_matches_index_select(self, config, mix):
        """The in-place packed builder is the shipped assembler's layout.

        build_packed_padded fills the (N,L,H,R) buffer adapter by adapter to
        avoid IndexSelectBatchAssembler's transient 2x (store + stacked copy
        resident together), and PackedPaddedAssembler reproduces its gather.
        Both stand in for repo code the benchmark's headline arm runs on, so
        they are pinned to it here rather than trusted to stay in step.
        """
        native_ranks = assign_ranks(8, mix)
        ids = [f"adapter_{i}" for i in range(config.batch_size)]

        shipped = IndexSelectBatchAssembler(
            build_padded_store(config, native_ranks), config
        )
        wa, wb = build_packed_padded(config, native_ranks)
        packed = PackedPaddedAssembler(wa, wb, config)

        for m in config.target_modules:
            assert torch.equal(wa[m], shipped.wa[m])
            assert torch.equal(wb[m], shipped.wb[m])

        want, got = shipped.to_layerwise(ids), packed.to_layerwise(ids)
        for layer in range(config.num_layers):
            for m in config.target_modules:
                assert torch.equal(got[layer].a[m][0], want[layer].a[m][0])
                assert torch.equal(got[layer].b[m][0], want[layer].b[m][0])


class TestPadToMaxAssembler:
    def test_matches_prepadded_store(self, config, mix):
        """Pad-at-gather and pad-at-load produce the same batch tensors.

        These are the two deployment layouts the benchmark compares, so if they
        disagreed the latency arms would not be measuring one system.
        """
        n = 8
        native_ranks = assign_ranks(n, mix)
        ids = [f"adapter_{i}" for i in range(config.batch_size)]

        prepadded = IndexSelectBatchAssembler(
            build_padded_store(config, native_ranks), config
        )
        buckets, row_of = build_native_buckets(config, native_ranks)
        pad_at_gather = PadToMaxAssembler(buckets, row_of, config, config.batch_size)

        want = prepadded.to_layerwise(ids)
        got = pad_at_gather.to_layerwise(ids)

        assert len(got) == config.num_layers
        for layer in range(config.num_layers):
            for m in config.target_modules:
                assert torch.equal(got[layer].a[m][0], want[layer].a[m][0])
                assert torch.equal(got[layer].b[m][0], want[layer].b[m][0])

    def test_no_stale_weights_across_batches(self, config, mix):
        """A low-rank tenant landing in a slot a high-rank tenant just used
        must not inherit the padded columns.

        The buffer is reused across batches, so this is the failure mode of
        zeroing once at construction instead of re-padding every batch: the
        stale columns would be multiplied by real activations and the tenant
        would be served a blend of two adapters.
        """
        native_ranks = assign_ranks(16, mix)
        buckets, row_of = build_native_buckets(config, native_ranks)
        assembler = PadToMaxAssembler(buckets, row_of, config, config.batch_size)

        high = [i for i, r in enumerate(native_ranks) if r == R_MAX]
        low = [i for i, r in enumerate(native_ranks) if r == R_MIN]
        assert high and low, "fixture mix must produce both ranks"

        assembler.to_layerwise([f"adapter_{high[0]}"] * config.batch_size)
        out_a, out_b = assembler.assemble_lora(
            [f"adapter_{low[0]}"] * config.batch_size
        )

        for m in config.target_modules:
            assert not out_a[m][:, :, :, R_MIN:].any()
            assert not out_b[m][:, :, R_MIN:, :].any()

    def test_native_store_is_smaller_than_padded(self, config, mix):
        """Padding costs compute, not resident memory.

        A 50/50 r=4 / r=16 fleet stored natively occupies (4+16)/(16+16) = 62.5%
        of the pre-padded layout, so a mixed fleet's tenant ceiling follows mean
        rank rather than max rank.
        """
        native_ranks = assign_ranks(64, mix)
        padded = build_padded_store(config, native_ranks).memory_bytes()
        buckets, _ = build_native_buckets(config, native_ranks)
        assert bucket_bytes(buckets) == pytest.approx(padded * 0.625)


class TestRankAssignment:
    def test_respects_fractions(self):
        ranks = assign_ranks(100, parse_mix("4:0.75,16:0.25"))
        assert ranks.count(4) == 75
        assert ranks.count(16) == 25

    def test_is_deterministic(self, mix):
        """The padded and native arms must describe the same fleet."""
        assert assign_ranks(64, mix) == assign_ranks(64, mix)

    def test_covers_every_adapter(self):
        for n in (1, 7, 64, 1000):
            ranks = assign_ranks(n, parse_mix("4,8,16,32"))
            assert len(ranks) == n
            assert set(ranks) <= {4, 8, 16, 32}
