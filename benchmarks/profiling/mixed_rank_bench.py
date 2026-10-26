"""Does Finding 1's flat-in-N latency survive a MIXED-RANK fleet?

Fig 3 shows latency flat as N scales — does this hold under mixed-rank
deployments (e.g. some adapters r=4, others r=16 in the same batch)? And does
padding every adapter to the max rank in the batch waste enough compute on the
low-rank adapters for the overhead to matter?

The paper's serving path is uniform-rank: `LoraServingConfig.lora_rank` is a
scalar, so every adapter in the store — and every slot in the (B, H, r) /
(B, r, H) batch tensors — has the same r. Serving a heterogeneous fleet means
zero-padding each tenant's A columns / B rows out to the batch's maximum rank.
That padding is *numerically exact* (the padded components multiply zero rows,
contributing nothing) and it makes the batch tensors bit-identically shaped to a
uniform-r_max batch, so the padded mixed batch runs the same kernels with the
same FLOPs as the uniform max-rank batch the paper already measures. This
benchmark is the empirical version of that argument.

Arms, at each (N, B):

  uniform_r{min}      every tenant at the fleet's LOWEST rank, no padding.
                      The "what if nobody forced padding" reference — the cost
                      a low-rank tenant would pay in a homogeneous deployment.
  uniform_r{max}      every tenant at the fleet's HIGHEST rank. The cost of the
                      padded batch's SHAPE.
  mixed_*_padded      heterogeneous native ranks, stored pre-padded to r_max,
                      gathered with the ordinary index_select assembler. The
                      simplest deployment: pad once at load time.
  mixed_*_native      the same fleet stored at NATIVE rank (one packed tensor
                      per rank bucket), zero-padded into the batch buffer at
                      gather time. Memory-optimal: the store pays for mean rank,
                      only the transient batch tensor pays for max rank.

Two ratios come out of that, and they are the answer:

  padding tax        mixed / uniform_r{min} — what a low-rank tenant actually
                     pays for sharing a batch with high-rank tenants.
  shape overhead     mixed / uniform_r{max} — whether a mixed batch costs
                     anything BEYOND its padded shape. Should be ~0; a non-zero
                     value means the gather, not the math, is the cost.

Before any timing, an exactness gate asserts that a low-rank tenant served
zero-padded inside an r_max batch produces the same logits as the same tenant
served natively in an r_min batch, and that the two mixed implementations agree.
The reported max|Δlogit| is the parity number to quote.

Unlike `latefuse.benchmark.synthetic`, adapters here are drawn with a
NON-ZERO B matrix. The shared helper zeroes B (making delta=0, which is safe for
its own correctness checks but would make this benchmark's exactness gate
vacuous — every arm would agree on a delta of zero). Shapes and kernels are
identical either way, so timing is unaffected.

Run on the GPU node (house 50/200 protocol, matching run_sxm80.sh):
    uv run python -m benchmarks.profiling.mixed_rank_bench \\
        --model BAAI/bge-m3 --dtype fp16 \\
        --adapters 100 1000 5000 10000 --batch-size 32 \\
        --rank-mix 4,16 4,8,16,32 --seq-len 128 \\
        --warmup 50 --iters 200 --seeds 1 2 3 4 5 \\
        --out benchmarks/results/mixed_rank/mixed_rank_bench.txt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Reuse the environment-capture and seed-aggregation helpers rather than copying
# them: cgroup_cpu_quota()'s container-vs-host core accounting is subtle enough
# that a second copy would drift, and both benchmarks must report provenance the
# same way for their numbers to compose in the fact sheet.
from benchmarks.profiling.assembly_bench import aggregate, cpu_line, env_metadata, fmt, stats
from latefuse.benchmark.synthetic import make_synthetic_inputs, make_synthetic_lr_weights
from latefuse.config import LoraServingConfig
from latefuse.model.encoder import EncoderWithLora
from latefuse.model.hf_wrapper import HFEncoderWithLora
from latefuse.weights.batch import (
    BatchAssembler,
    IndexSelectBatchAssembler,
    LayerwiseBatchedWeights,
)
from latefuse.weights.store import AdapterStore, LoraWeight

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

# Std of the synthetic A/B draws. Matches make_synthetic_adapters' A init; the
# resulting per-layer delta is ~5% of the base projection at r=16, i.e. the same
# order as a trained adapter and comfortably inside fp16 range over 24 layers.
INIT_STD = 0.02
# Per-adapter weight seed, offset by adapter index. Shared by every arm so that
# adapter i drawn at rank r is bit-identical wherever it appears — which is what
# makes the padded-vs-native exactness gate meaningful rather than a comparison
# of two unrelated random tensors.
WEIGHT_SEED = 42
# Seed for the native-rank ASSIGNMENT (which tenant gets which rank). Fixed, so
# the padded and native arms of one mix describe the same fleet.
FLEET_SEED = 0
LR_SEED = 7
# fp16 logits through 24 layers: the padded and native paths do the same math
# but with different bmm reduction widths, so cuBLAS may pick different
# algorithms. Same tolerance assembly_bench.py gates its equivalence check on.
EXACTNESS_TOL = 1e-3


# --------------------------------------------------------------- rank mixes


@dataclass
class Mix:
    """A fleet composition: which native ranks, in what proportion."""

    ranks: list[int]
    fracs: list[float]

    @property
    def label(self) -> str:
        return "+".join(str(r) for r in self.ranks)

    @property
    def r_min(self) -> int:
        return min(self.ranks)

    @property
    def r_max(self) -> int:
        return max(self.ranks)


def parse_mix(spec: str) -> Mix:
    """'4,16' -> equal split; '4:0.75,16:0.25' -> explicit fractions.

    Fractions need not sum to 1; they are normalised. Ranks are sorted so a mix
    is named the same however it was typed.
    """
    ranks: list[int] = []
    weights: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            r_str, w_str = part.split(":", 1)
            ranks.append(int(r_str))
            weights.append(float(w_str))
        else:
            ranks.append(int(part))
            weights.append(1.0)
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"--rank-mix {spec!r} repeats a rank")
    if len(ranks) < 2:
        raise ValueError(
            f"--rank-mix {spec!r} names {len(ranks)} rank(s); a mixed-rank arm "
            "needs at least two, otherwise it is just the uniform rank sweep "
            "the paper already reports"
        )
    if any(w <= 0 for w in weights):
        raise ValueError(f"--rank-mix {spec!r} has a non-positive fraction")
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    total = sum(weights)
    return Mix(
        ranks=[ranks[i] for i in order],
        fracs=[weights[i] / total for i in order],
    )


def assign_ranks(n: int, mix: Mix) -> list[int]:
    """Native rank per adapter index, shuffled under a fixed seed.

    Shuffled rather than blocked so that a batch drawn uniformly from the pool
    is a random draw from the fleet composition, not a run of consecutive
    same-rank ids.
    """
    counts = [int(round(f * n)) for f in mix.fracs[:-1]]
    counts.append(n - sum(counts))
    if counts[-1] < 0:
        raise ValueError(
            f"rank fractions {mix.fracs} do not fit {n} adapters: the "
            f"leading buckets already claim {sum(counts[:-1])}"
        )
    out = [r for r, c in zip(mix.ranks, counts) for _ in range(c)]
    random.Random(FLEET_SEED).shuffle(out)
    return out


# ------------------------------------------------------------ weight drawing


def draw_native(
    config: LoraServingConfig, rank: int, seed: int
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """One adapter's A/B at its NATIVE rank: wa (L,H,r), wb (L,r,H).

    Drawn into freshly allocated contiguous tensors, never into a padded slice.
    A strided view consumes the RNG stream in an order PyTorch does not promise
    to keep stable, so drawing straight into `wa[:, :, :r]` of an r_max slot
    would not reliably reproduce the same adapter as drawing at native rank —
    and the exactness gate would then be comparing two different adapters.
    """
    torch.manual_seed(seed)
    L, H = config.num_layers, config.hidden_size
    kw = dict(dtype=config.dtype, device=config.device)
    a = {
        m: torch.empty(L, H, rank, **kw).normal_(0.0, INIT_STD)
        for m in config.target_modules
    }
    b = {
        m: torch.empty(L, rank, H, **kw).normal_(0.0, INIT_STD)
        for m in config.target_modules
    }
    return a, b


def build_padded_store(
    config: LoraServingConfig, native_ranks: list[int]
) -> AdapterStore:
    """Store every adapter in a `config.lora_rank`-wide slot, zero-padding.

    `LoraWeight` allocates with torch.zeros, so the padded tail is already zero;
    only the leading `native_rank` columns/rows are written.

    Weights are inserted straight into the store's dict because AdapterStore has
    no public "adopt this prepared LoraWeight" entry point — load_from_file
    reads a checkpoint and load_synthetic draws its own (B=0) weights, and
    neither can express "this adapter is natively rank 4 living in a rank-16
    slot", which is the whole subject of this benchmark.
    """
    store = AdapterStore(config)
    for i, r in enumerate(native_ranks):
        a, b = draw_native(config, r, WEIGHT_SEED + i)
        weight = LoraWeight(config)
        for m in config.target_modules:
            weight.wa[m][:, :, :r].copy_(a[m])
            weight.wb[m][:, :r, :].copy_(b[m])
        store._store[f"adapter_{i}"] = weight
    return store


def build_packed_padded(
    config: LoraServingConfig, native_ranks: list[int]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Pre-padded packed tensors — wa (N,L,H,r_max), wb (N,L,r_max,H) — filled
    adapter by adapter.

    This is the layout :class:`IndexSelectBatchAssembler` builds, reached
    without its transient cost: it torch.stacks an AdapterStore, so the N
    per-adapter tensors and the stacked copy are resident at once, a 2x spike
    that halves the pool this benchmark can reach at r=32 (a 10,000-adapter
    r=32 fleet is 63 GB packed, so the spike alone would not fit an 80 GB card).
    Filling the buffer in place peaks at 1x plus one adapter.
    `tests/test_mixed_rank_padding.py` pins the two layouts to the same bytes so
    this cannot drift from the assembler it stands in for.
    """
    n, L, H = len(native_ranks), config.num_layers, config.hidden_size
    r_max = config.lora_rank
    kw = dict(dtype=config.dtype, device=config.device)
    wa = {m: torch.zeros(n, L, H, r_max, **kw) for m in config.target_modules}
    wb = {m: torch.zeros(n, L, r_max, H, **kw) for m in config.target_modules}
    for i, r in enumerate(native_ranks):
        a, b = draw_native(config, r, WEIGHT_SEED + i)
        for m in config.target_modules:
            wa[m][i, :, :, :r].copy_(a[m])
            wb[m][i, :, :r, :].copy_(b[m])
    return wa, wb


def build_native_buckets(
    config: LoraServingConfig, native_ranks: list[int]
) -> tuple[dict[int, dict[str, dict[str, Tensor]]], dict[str, tuple[int, int]]]:
    """Pack the fleet into one contiguous tensor per (rank, module).

    Returns:
        buckets: rank -> {"a": {module: (N_r, L, H, r)},
                          "b": {module: (N_r, L, r, H)}}
        row_of:  adapter_id -> (native_rank, row index within that bucket)

    This is the memory-optimal layout: an r=4 tenant occupies four columns, not
    r_max. Padding happens per batch in PadToMaxAssembler, on the transient
    (B, L, H, r_max) buffer only.
    """
    per_rank: dict[int, list[int]] = {}
    for i, r in enumerate(native_ranks):
        per_rank.setdefault(r, []).append(i)

    buckets: dict[int, dict[str, dict[str, Tensor]]] = {}
    row_of: dict[str, tuple[int, int]] = {}
    for r, indices in per_rank.items():
        a_rows: dict[str, list[Tensor]] = {m: [] for m in config.target_modules}
        b_rows: dict[str, list[Tensor]] = {m: [] for m in config.target_modules}
        for row, i in enumerate(indices):
            a, b = draw_native(config, r, WEIGHT_SEED + i)
            for m in config.target_modules:
                a_rows[m].append(a[m])
                b_rows[m].append(b[m])
            row_of[f"adapter_{i}"] = (r, row)
        buckets[r] = {
            "a": {m: torch.stack(a_rows[m]) for m in config.target_modules},
            "b": {m: torch.stack(b_rows[m]) for m in config.target_modules},
        }
    return buckets, row_of


def bucket_bytes(buckets: dict[int, dict[str, dict[str, Tensor]]]) -> int:
    return sum(
        t.nbytes
        for bucket in buckets.values()
        for side in bucket.values()
        for t in side.values()
    )


# -------------------------------------------------------- padding assembler


class PackedPaddedAssembler:
    """:class:`IndexSelectBatchAssembler`'s gather over pre-built packed tensors.

    Performs the same two operations the shipped assembler does — one
    ``index_select`` per target module over the leading adapter axis, then
    per-layer views of the result — and exists only because that class builds
    its packed tensors in its constructor and cannot be handed tensors that were
    filled in place (see :func:`build_packed_padded` for why that matters).
    Nothing about the gather differs, and
    ``tests/test_mixed_rank_padding.py::test_packed_matches_index_select``
    asserts the two produce identical batch tensors.
    """

    def __init__(
        self, wa: dict[str, Tensor], wb: dict[str, Tensor], config: LoraServingConfig
    ):
        self.wa = wa
        self.wb = wb
        self._cfg = config
        n = next(iter(wa.values())).shape[0]
        self._row_of = {f"adapter_{i}": i for i in range(n)}

    def to_layerwise(self, adapter_ids: list[str]) -> list[LayerwiseBatchedWeights]:
        idx = torch.tensor(
            [self._row_of[a] for a in adapter_ids],
            dtype=torch.long,
            device=self._cfg.device,
        )
        layers = [LayerwiseBatchedWeights() for _ in range(self._cfg.num_layers)]
        for m in self._cfg.target_modules:
            a = self.wa[m].index_select(0, idx)
            b = self.wb[m].index_select(0, idx)
            for layer_idx in range(self._cfg.num_layers):
                layers[layer_idx].a[m] = [a[:, layer_idx]]
                layers[layer_idx].b[m] = [b[:, layer_idx]]
        return layers


class PadToMaxAssembler:
    """Gathers a mixed-rank batch from native-rank buckets, padding to r_max.

    Per batch, for each rank bucket present: one ``index_select`` pulls that
    bucket's rows, ``F.pad`` zero-extends them to r_max, and ``index_copy_``
    scatters them into the pre-allocated batch buffer at their batch positions.
    Work is O(number of distinct ranks in the batch), not O(B) — the same
    device-resident shape as :class:`IndexSelectBatchAssembler`, which is what
    keeps this arm comparable to the pre-padded one.

    The pad is re-applied every batch rather than the buffer being zeroed once
    at construction: a slot that held an r=32 tenant last batch and an r=4
    tenant this batch would otherwise keep the previous occupant's weights in
    columns 4:32 and silently serve a blend of two tenants.
    """

    def __init__(
        self,
        buckets: dict[int, dict[str, dict[str, Tensor]]],
        row_of: dict[str, tuple[int, int]],
        config: LoraServingConfig,
        batch_size: int,
    ):
        self._cfg = config
        self._buckets = buckets
        self._row_of = row_of
        self._rmax = config.lora_rank
        L, H = config.num_layers, config.hidden_size
        kw = dict(dtype=config.dtype, device=config.device)
        self._out_a = {
            m: torch.zeros(batch_size, L, H, self._rmax, **kw)
            for m in config.target_modules
        }
        self._out_b = {
            m: torch.zeros(batch_size, L, self._rmax, H, **kw)
            for m in config.target_modules
        }

    def assemble_lora(self, adapter_ids: list[str]) -> tuple[dict, dict]:
        by_rank: dict[int, tuple[list[int], list[int]]] = {}
        for pos, aid in enumerate(adapter_ids):
            r, row = self._row_of[aid]
            slot = by_rank.setdefault(r, ([], []))
            slot[0].append(pos)
            slot[1].append(row)

        device = self._cfg.device
        for r, (positions, rows) in by_rank.items():
            pos_t = torch.tensor(positions, dtype=torch.long, device=device)
            row_t = torch.tensor(rows, dtype=torch.long, device=device)
            pad = self._rmax - r
            for m in self._cfg.target_modules:
                a = self._buckets[r]["a"][m].index_select(0, row_t)
                b = self._buckets[r]["b"][m].index_select(0, row_t)
                if pad:
                    a = F.pad(a, (0, pad))           # (b_r, L, H, r) -> r_max
                    b = F.pad(b, (0, 0, 0, pad))     # (b_r, L, r, H) -> r_max
                self._out_a[m].index_copy_(0, pos_t, a)
                self._out_b[m].index_copy_(0, pos_t, b)
        return self._out_a, self._out_b

    def to_layerwise(self, adapter_ids: list[str]) -> list[LayerwiseBatchedWeights]:
        out_a, out_b = self.assemble_lora(adapter_ids)
        layers = [LayerwiseBatchedWeights() for _ in range(self._cfg.num_layers)]
        for m in self._cfg.target_modules:
            for layer_idx in range(self._cfg.num_layers):
                layers[layer_idx].a[m] = [out_a[m][:, layer_idx]]
                layers[layer_idx].b[m] = [out_b[m][:, layer_idx]]
        return layers


# ------------------------------------------------------------------- arms


@dataclass
class Arm:
    """One measured configuration: a fleet composition plus a storage layout."""

    name: str
    kind: str            # "padded" (uniform or pre-padded mixed) | "native"
    mix: Mix
    slot_rank: int       # the rank the ops/model run at = r_max of the mix


def uniform_arm(rank: int) -> Arm:
    return Arm(
        name=f"uniform_r{rank}",
        kind="padded",
        mix=Mix(ranks=[rank], fracs=[1.0]),
        slot_rank=rank,
    )


def mix_arm_names(mix: Mix) -> tuple[str, str, str, str]:
    """The four arm names a mix's verdict is computed from, in report order."""
    return (
        f"uniform_r{mix.r_min}",
        f"uniform_r{mix.r_max}",
        f"mixed_{mix.label}_padded",
        f"mixed_{mix.label}_native",
    )


def build_arms(mixes: list[Mix]) -> list[Arm]:
    """Arms for a set of mixes, de-duplicated (mixes often share r_min)."""
    arms: dict[str, Arm] = {}
    for mix in mixes:
        for arm in (
            uniform_arm(mix.r_min),
            uniform_arm(mix.r_max),
            Arm(f"mixed_{mix.label}_padded", "padded", mix, mix.r_max),
            Arm(f"mixed_{mix.label}_native", "native", mix, mix.r_max),
        ):
            arms.setdefault(arm.name, arm)
    return list(arms.values())


def arm_config(
    args, arm: Arm, device: torch.device, dtype: torch.dtype
) -> LoraServingConfig:
    return LoraServingConfig(
        model_name=args.model,
        lora_rank=arm.slot_rank,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        target_modules=args.target_modules,
        device=device,
        dtype=dtype,
    )


def build_arm(args, arm: Arm, n: int, config: LoraServingConfig):
    """Materialise an arm's store and assembler.

    Returns (assemble_fn, resident_bytes, owner). `owner` is the object graph
    holding the packed weights; the caller drops it to free the arm before the
    next one allocates.
    """
    native_ranks = assign_ranks(n, arm.mix)
    if arm.kind == "native":
        buckets, row_of = build_native_buckets(config, native_ranks)
        assembler = PadToMaxAssembler(buckets, row_of, config, args.batch_size)
        return assembler.to_layerwise, bucket_bytes(buckets), (buckets, assembler)

    if args.assembler == "baseline":
        store = build_padded_store(config, native_ranks)
        baseline = BatchAssembler(store, config)
        return baseline.assemble_lora, store.memory_bytes(), (store, baseline)

    wa, wb = build_packed_padded(config, native_ranks)
    packed = PackedPaddedAssembler(wa, wb, config)
    packed_bytes = sum(t.nbytes for side in (wa, wb) for t in side.values())
    return packed.to_layerwise, packed_bytes, (wa, wb, packed)


# ---------------------------------------------------------------- timing


def time_arm(
    model,
    assemble_fn,
    adapter_ids: list[str],
    native_of: dict[str, int],
    r_max: int,
    lr_for,
    inputs: dict,
    output_lr: Tensor,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict:
    """Warm up, then time `iters` batches. Returns flat per-seed scalars."""

    def sample_ids() -> list[str]:
        return random.choices(adapter_ids, k=batch_size)

    for _ in range(warmup):
        ids = sample_ids()
        lora_w = assemble_fn(ids)
        with torch.no_grad():
            model(inputs["input_ids"], inputs["attention_mask"], lora_w,
                  lr_for(ids), output_lr)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    asm_ms = np.empty(iters)
    total_ms = np.empty(iters)
    at_fleet_max = 0
    for i in range(iters):
        ids = sample_ids()
        if max(native_of[a] for a in ids) == r_max:
            at_fleet_max += 1
        lr_w = lr_for(ids)

        t0 = time.perf_counter()
        lora_w = assemble_fn(ids)
        # Sync so a device-resident gather is charged to assembly rather than
        # leaking into the forward timed straight after.
        torch.cuda.synchronize(device)
        asm_ms[i] = (time.perf_counter() - t0) * 1000

        with torch.no_grad():
            model(inputs["input_ids"], inputs["attention_mask"], lora_w,
                  lr_w, output_lr)
        torch.cuda.synchronize(device)
        total_ms[i] = (time.perf_counter() - t0) * 1000

    fwd_ms = total_ms - asm_ms
    t, a, f = stats(total_ms), stats(asm_ms), stats(fwd_ms)
    return {
        "total_mean_ms": t["mean_ms"],
        "total_p50_ms": t["p50_ms"],
        "total_p99_ms": t["p99_ms"],
        "p99_over_p50": t["p99_over_p50"],
        "assemble_mean_ms": a["mean_ms"],
        "forward_mean_ms": f["mean_ms"],
        "forward_p50_ms": f["p50_ms"],
        "assemble_share_pct": 100.0 * a["mean_ms"] / t["mean_ms"],
        "throughput_samples_sec": batch_size / (t["mean_ms"] / 1000.0),
        # Includes every model in the rank cache, not just this arm's store —
        # `store_gb`, computed from the tensors themselves, is the figure the
        # memory claim rests on.
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "batches_at_fleet_max_rank_pct": 100.0 * at_fleet_max / iters,
    }


# ------------------------------------------------------------- exactness


def _padding_identity_fp32(
    args, mix: Mix, device, gate_dtype, n: int, native_ranks, ids_low
) -> float:
    """max|delta logit| for a low-rank tenant served natively vs zero-padded.

    Self-contained at `gate_dtype`: its own models, LR heads and stores, all
    dropped before returning, so the serving-dtype objects the timed arms reuse
    are untouched. Two extra checkpoint loads at a pool of `--check-adapters`
    (64 by default) is a few seconds once per mix.
    """
    cfg_min = arm_config(args, uniform_arm(mix.r_min), device, gate_dtype)
    cfg_max = arm_config(args, uniform_arm(mix.r_max), device, gate_dtype)

    model_min = EncoderWithLora.from_pretrained_serving(cfg_min).eval()
    model_max = EncoderWithLora.from_pretrained_serving(cfg_max).eval()

    # Same draw as the serving-dtype path (same seed, same order), just widened
    # — LR heads are rank-independent, so both slot ranks see the same head.
    torch.manual_seed(LR_SEED)
    lr_store = {
        f"adapter_{i}": make_synthetic_lr_weights(cfg_max, args.num_labels)
        for i in range(n)
    }
    lr_assembler = BatchAssembler(AdapterStore(cfg_max), cfg_max)

    def lr_for(ids: list[str]):
        return lr_assembler.assemble_lr(
            [lr_store[a][0] for a in ids], [lr_store[a][1] for a in ids]
        )

    wa_min, wb_min = build_packed_padded(cfg_min, [mix.r_min] * n)
    asm_min = PackedPaddedAssembler(wa_min, wb_min, cfg_min)
    wa_pad, wb_pad = build_packed_padded(cfg_max, native_ranks)
    asm_pad = PackedPaddedAssembler(wa_pad, wb_pad, cfg_max)

    inputs = make_synthetic_inputs(cfg_max, args.batch_size)
    out = torch.zeros(
        args.batch_size, 1, args.num_labels, dtype=gate_dtype, device=device
    )

    def run(model, lora_w):
        with torch.no_grad():
            return model(inputs["input_ids"], inputs["attention_mask"], lora_w,
                         lr_for(ids_low), out.clone()).clone()

    d = (run(model_min, asm_min.to_layerwise(ids_low))
         - run(model_max, asm_pad.to_layerwise(ids_low))).abs().max().item()

    del (model_min, model_max, wa_min, wb_min, asm_min, wa_pad, wb_pad,
         asm_pad, lr_store, lr_assembler)
    torch.cuda.empty_cache()
    return d


def check_exactness(args, mix: Mix, device, dtype, model_for, lr_for, n: int) -> dict:
    """Gate: padding must not change what a tenant is served.

    Check 1 (the load-bearing one): take tenants whose NATIVE rank is r_min and
    serve them two ways — natively in an r_min batch, and zero-padded inside an
    r_max batch. Same weights by construction (same seed, same native shape), so
    identical logits is the claim that "pad-to-max is exact" rests on.

    Check 1 runs in fp32, NOT in the run's serving dtype, and that distinction
    is the whole reason this docstring is long. The identity being asserted is
    algebraic — the padded components multiply zero rows, so they contribute
    nothing — and in fp32 it holds bit-exactly on-device (measured 0.0e+00, not
    merely inside a tolerance). In fp16 it does not, and not because padding is
    wrong: the native check runs an (H x 4) GEMM where the padded one runs an
    (H x 16), cuBLAS picks its tiling and split-k from that shape, and the two
    therefore accumulate the same products in different orders. One layer's
    delta moves by ~3e-5 against a delta of scale ~1e-2 (0.3%, i.e. fp16
    rounding), but bge-m3 restacks that through 24 layers of attention and the
    logits end up ~7% apart — 8e-1 absolute against a mean |logit| near 11.
    Gating THAT at 1e-3 measures the stability of fp16 accumulation across two
    GEMM shapes, which is not what this benchmark is about and which no correct
    implementation could pass. The serving-dtype divergence is still measured
    and reported below (`..._serving_dtype`), because it is a real property of
    a mixed-rank fleet that should be quoted rather than hidden: padding a
    tenant to r_max in fp16 perturbs its logits at the same order as any other
    change of batch composition or cuBLAS version. It does not touch the timing
    arms, whose shapes, kernels and FLOP counts are identical either way.

    Checks 2 and 3 stay in the serving dtype. They compare LAYOUTS at one fixed
    slot rank — pre-padded store vs pad-at-gather, and the in-place packed
    builder vs the shipped IndexSelectBatchAssembler — so they are pure data
    movement into identically shaped buffers, with no GEMM shape change to
    re-associate anything. They are exact in fp16 and are asserted as such.
    """
    gate_dtype = torch.float32
    cfg_min = arm_config(args, uniform_arm(mix.r_min), device, dtype)
    cfg_max = arm_config(args, uniform_arm(mix.r_max), device, dtype)
    native_ranks = assign_ranks(n, mix)

    low_ids = [f"adapter_{i}" for i, r in enumerate(native_ranks) if r == mix.r_min]
    if not low_ids:
        raise ValueError(
            f"mix {mix.label} put no adapter at r={mix.r_min} in a pool of {n}: "
            f"its fraction ({mix.fracs[0]}) rounds to zero. Raise "
            "--check-adapters so the low-rank bucket is non-empty."
        )

    # Built exactly as the timed arms build them, so the gate certifies the
    # objects that actually get measured. Every adapter at r_min in an r_min
    # slot: adapter i's draw here is the same tensor as its draw in the padded
    # layout, because draw_native keys on (native rank, seed) and both use the
    # fleet's assignment for i.
    wa_min, wb_min = build_packed_padded(cfg_min, [mix.r_min] * n)
    asm_min = PackedPaddedAssembler(wa_min, wb_min, cfg_min)

    wa_pad, wb_pad = build_packed_padded(cfg_max, native_ranks)
    asm_pad = PackedPaddedAssembler(wa_pad, wb_pad, cfg_max)

    buckets, row_of = build_native_buckets(cfg_max, native_ranks)
    asm_nat = PadToMaxAssembler(buckets, row_of, cfg_max, args.batch_size)

    store_shipped = build_padded_store(cfg_max, native_ranks)
    asm_shipped = IndexSelectBatchAssembler(store_shipped, cfg_max)

    model_min = model_for(mix.r_min)
    model_max = model_for(mix.r_max)
    inputs = make_synthetic_inputs(cfg_max, args.batch_size)
    out = torch.zeros(args.batch_size, 1, args.num_labels, dtype=dtype, device=device)

    rng = random.Random(FLEET_SEED)
    ids_low = rng.choices(low_ids, k=args.batch_size)
    ids_any = rng.choices([f"adapter_{i}" for i in range(n)], k=args.batch_size)

    def run(model, lora_w, ids):
        with torch.no_grad():
            return model(inputs["input_ids"], inputs["attention_mask"], lora_w,
                         lr_for(ids), out.clone()).clone()

    # Informational only — the serving dtype's answer to check 1, kept so the
    # fp16 number can be quoted rather than discovered by a reader re-running
    # this. See the docstring for why it is not the gated quantity.
    native_logits = run(model_min, asm_min.to_layerwise(ids_low), ids_low)
    padded_logits = run(model_max, asm_pad.to_layerwise(ids_low), ids_low)
    d_pad_serving = (native_logits - padded_logits).abs().max().item()
    logit_scale = native_logits.abs().mean().item()

    pad_mixed = run(model_max, asm_pad.to_layerwise(ids_any), ids_any)
    nat_mixed = run(model_max, asm_nat.to_layerwise(ids_any), ids_any)
    d_layout = (pad_mixed - nat_mixed).abs().max().item()

    shipped_mixed = run(model_max, asm_shipped.to_layerwise(ids_any), ids_any)
    d_shipped = (pad_mixed - shipped_mixed).abs().max().item()

    del (wa_min, wb_min, asm_min, wa_pad, wb_pad, asm_pad, buckets, asm_nat,
         store_shipped, asm_shipped)
    torch.cuda.empty_cache()

    # Check 1 proper, in fp32. Built and torn down here rather than through
    # `model_for` because that cache is keyed on slot rank alone and holds the
    # serving-dtype models the timed arms reuse; a gate must not evict them.
    d_pad = _padding_identity_fp32(args, mix, device, gate_dtype, n,
                                   native_ranks, ids_low)

    if d_pad > EXACTNESS_TOL:
        raise AssertionError(
            f"mix {mix.label}: zero-padding changed the served result in "
            f"fp32, where the identity is algebraic — max|delta logit| "
            f"padded-vs-native={d_pad:.3e}, tol={EXACTNESS_TOL:.0e}. This is a "
            "real padding bug, not an accumulation-order artefact; the timing "
            "arms below would be comparing different computations."
        )
    worst_layout = max(d_layout, d_shipped)
    if worst_layout > EXACTNESS_TOL:
        raise AssertionError(
            f"mix {mix.label}: two layouts of the SAME slot rank disagree in "
            f"{args.dtype} — pre-padded-vs-pad-at-gather={d_layout:.3e}, "
            f"packed-vs-shipped-assembler={d_shipped:.3e}, "
            f"tol={EXACTNESS_TOL:.0e}. No GEMM shape changes between these, so "
            "this is a gather bug, not rounding."
        )
    return {
        "mix": mix.label,
        "checked_adapters": n,
        "max_abs_delta_padded_vs_native": d_pad,
        "max_abs_delta_padded_vs_native_serving_dtype": d_pad_serving,
        "serving_dtype_mean_abs_logit": logit_scale,
        "gate_dtype": "fp32",
        "max_abs_delta_layouts": d_layout,
        "max_abs_delta_vs_shipped_assembler": d_shipped,
        "tolerance": EXACTNESS_TOL,
    }


# ------------------------------------------------------------------ report


def derived(agg_by_arm: dict, mix: Mix) -> dict:
    """The ratios that answer the question, for one (N, mix)."""

    def total(name: str) -> float:
        return agg_by_arm[name]["total_mean_ms"]["mean"]

    def fwd(name: str) -> float:
        return agg_by_arm[name]["forward_mean_ms"]["mean"]

    lo, hi, _, _ = mix_arm_names(mix)
    out = {"uniform_rank_delta_pct": 100.0 * (total(hi) - total(lo)) / total(lo)}
    for layout in ("padded", "native"):
        arm = f"mixed_{mix.label}_{layout}"
        if arm not in agg_by_arm:
            continue
        out[f"{layout}_padding_tax_vs_rmin_pct"] = 100.0 * (total(arm) - total(lo)) / total(lo)
        out[f"{layout}_padding_tax_forward_pct"] = 100.0 * (fwd(arm) - fwd(lo)) / fwd(lo)
        out[f"{layout}_overhead_vs_rmax_pct"] = 100.0 * (total(arm) - total(hi)) / total(hi)
        out[f"{layout}_store_gb"] = agg_by_arm[arm]["store_gb"]["mean"]
        out[f"{layout}_batches_at_fleet_max_pct"] = agg_by_arm[arm][
            "batches_at_fleet_max_rank_pct"
        ]["mean"]
    if "padded_store_gb" in out and "native_store_gb" in out:
        out["native_store_vs_padded"] = out["native_store_gb"] / out["padded_store_gb"]
    return out


def write_report(path: Path, args, env, seeds, mixes, results, exactness) -> None:
    """results[(n, arm_name)] = aggregated {metric: {mean, std}}."""
    multi = len(seeds) > 1
    ns = args.adapters

    def column(metric: str, prec: int):
        def cell(n: int, name: str) -> str:
            r = results.get((n, name))
            return fmt(r[metric], prec=prec, multi=multi) if r else "OOM"
        return cell

    with path.open("w") as f:
        f.write(f"# Mixed-rank serving — {args.model}\n")
        f.write(f"# GPU: {env['gpu']}\n")
        f.write(f"# CPU: {cpu_line(env)}\n")
        f.write(f"# torch {env['torch_version']} / CUDA {env['cuda_version']}\n")
        seed_str = ",".join(str(s) for s in seeds) if multi else "single run"
        f.write(
            f"# B={args.batch_size} seq={args.seq_len} dtype={args.dtype} "
            f"engine={args.engine} assembler={args.assembler} "
            f"warmup={args.warmup} iters={args.iters} seeds={seed_str}\n"
        )
        f.write(f"# targets={'+'.join(args.target_modules)}\n")
        if multi:
            f.write("# Values are mean±s.d. across seeds.\n")
        f.write("#\n# Padding exactness, asserted before any timing:\n")
        for e in exactness:
            f.write(
                f"#   mix {e['mix']:<12} max|delta logit|  "
                f"padded-vs-native={e['max_abs_delta_padded_vs_native']:.2e} "
                f"[{e.get('gate_dtype', 'fp32')}]  "
                f"layouts={e['max_abs_delta_layouts']:.2e}  "
                f"vs-shipped-assembler={e['max_abs_delta_vs_shipped_assembler']:.2e}  "
                f"(tol {e['tolerance']:.0e})\n"
            )
        # Reported, not gated. The identity is algebraic and holds bit-exactly
        # in fp32; in fp16 the native and padded paths run different GEMM
        # shapes, so they accumulate in different orders and bge-m3 restacks
        # that over 24 layers. Quote it as the cost of padding in fp16, not as
        # a failure — no shape, kernel or FLOP count differs between the arms.
        f.write(f"#\n# Same check in the serving dtype ({args.dtype}), reported "
                "not gated —\n# accumulation order, not padding:\n")
        for e in exactness:
            d = e.get("max_abs_delta_padded_vs_native_serving_dtype")
            if d is None:
                continue
            scale = e.get("serving_dtype_mean_abs_logit") or float("nan")
            f.write(
                f"#   mix {e['mix']:<12} max|delta logit|={d:.2e}  "
                f"mean|logit|={scale:.2e}  ({100.0 * d / scale:.2f}% of scale)\n"
            )

        for mix in mixes:
            names = mix_arm_names(mix)
            f.write(
                f"\n\n## Mix {mix.label}  (r_min={mix.r_min}, r_max={mix.r_max}, "
                f"fractions {[round(x, 3) for x in mix.fracs]})\n"
            )
            for title, metric, prec in (
                ("End-to-end p50 latency (assemble + forward), ms", "total_p50_ms", 2),
                ("Forward-only mean, ms (the rank-sensitive component)", "forward_mean_ms", 2),
                ("Assembly mean, ms", "assemble_mean_ms", 3),
                ("Adapter store, GB", "store_gb", 2),
            ):
                cell = column(metric, prec)
                f.write(f"\n### {title}\n")
                f.write(f"{'N':>8}  " + "  ".join(f"{n:>21}" for n in names) + "\n")
                for n in ns:
                    row = "  ".join(f"{cell(n, name):>21}" for name in names)
                    f.write(f"{n:>8}  {row}\n")

            f.write("\n### Verdict (per N)\n")
            f.write(
                f"{'N':>8}  {'pad tax vs r_min':>17}  {'(forward only)':>16}  "
                f"{'vs uniform r_max':>17}  {'store vs padded':>16}  "
                f"{'batches @ r_max':>16}\n"
            )
            for n in ns:
                have = {name: results[(n, name)] for name in names if (n, name) in results}
                if len(have) < len(names):
                    missing = ", ".join(sorted(set(names) - set(have)))
                    f.write(f"{n:>8}  incomplete — no result for: {missing}\n")
                    continue
                d = derived(have, mix)
                f.write(
                    f"{n:>8}  {d['padded_padding_tax_vs_rmin_pct']:>+16.2f}%  "
                    f"{d['padded_padding_tax_forward_pct']:>+15.2f}%  "
                    f"{d['padded_overhead_vs_rmax_pct']:>+16.2f}%  "
                    f"{d['native_store_vs_padded']:>15.2f}x  "
                    f"{d['padded_batches_at_fleet_max_pct']:>15.1f}%\n"
                )

        f.write("\n\nInterpretation:\n")
        f.write("- 'pad tax vs r_min' is the cost a LOW-rank tenant pays for sharing a\n")
        f.write("  batch with high-rank tenants: the mixed batch against a batch where\n")
        f.write("  every tenant runs at the fleet's lowest rank and nothing is padded.\n")
        f.write("  This is the quantity the uniform-rank limitation is about.\n")
        f.write("- 'vs uniform r_max' isolates whether a mixed batch costs anything\n")
        f.write("  BEYOND its padded shape. Zero-padding makes the batch tensors\n")
        f.write("  bit-identically shaped to a uniform r_max batch, so this should sit\n")
        f.write("  at ~0; a positive value is gather cost, not arithmetic.\n")
        f.write("- 'store vs padded' is the native-rank layout's footprint relative to\n")
        f.write("  pre-padding every adapter to r_max. Below 1.0x means padding can be\n")
        f.write("  confined to the transient batch tensor, so a mixed fleet's tenant\n")
        f.write("  ceiling is set by MEAN rank, not max.\n")
        f.write("- 'batches @ r_max' is the share of timed batches whose maximum native\n")
        f.write("  rank equals the fleet maximum. Near 100% means padding to the batch\n")
        f.write("  max is, in practice, padding to the fleet max — the pessimistic\n")
        f.write("  reading of these numbers is the realistic one, and rank-bucketed\n")
        f.write("  routing is the only thing that would change it.\n")
        f.write("- Latency flat down each N column is Finding 1 surviving a mixed-rank\n")
        f.write("  fleet: pool size enters no tensor shape, rank enters the batch\n")
        f.write("  tensors only, and the two axes stay independent.\n")


# -------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Mixed-rank multi-tenant LoRA serving benchmark"
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument(
        "--adapters", nargs="+", type=int, default=[100, 1000, 5000, 10000],
        help="Pool sizes to sweep. This is the Fig-3 axis: the question is "
             "whether flat-in-N survives a mixed-rank fleet, so N must vary.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Single batch size (Finding 3's operating point). Arms, not batch "
             "sizes, are this benchmark's axis.",
    )
    parser.add_argument(
        "--rank-mix", nargs="+", default=["4,16", "4,8,16,32"], metavar="SPEC",
        help="Fleet compositions. 'r1,r2' splits evenly; 'r1:f1,r2:f2' gives "
             "explicit fractions (normalised). The default covers the "
             "even (4,16) split and the full-spread worst case (4,8,16,32).",
    )
    parser.add_argument("--engine", choices=["custom", "hf"], default="custom")
    parser.add_argument(
        "--assembler", choices=["indexsel", "baseline"], default="indexsel",
        help="Gather strategy for the uniform and pre-padded arms. 'indexsel' "
             "is the device-resident path; 'baseline' is the paper's CPU-loop "
             "BatchAssembler, for reproducing under the assembler the published "
             "figures used. The native-rank arm always pads at gather time — "
             "there is no CPU-loop analogue of it.",
    )
    parser.add_argument("--target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--check-adapters", type=int, default=64,
        help="Pool size for the pre-timing exactness gate. Small on purpose: "
             "the padding identity does not depend on N.",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/results/mixed_rank/mixed_rank_bench.txt",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Need CUDA. Aborting.")
        return

    device = torch.device("cuda:0")
    dtype = DTYPE_MAP[args.dtype]
    env = env_metadata()
    print(f"GPU: {env['gpu']}")
    print(f"CPU: {cpu_line(env)}")

    mixes = [parse_mix(s) for s in args.rank_mix]
    arms = build_arms(mixes)
    print(f"\nMixes: {[m.label for m in mixes]}")
    print(f"Arms:  {[a.name for a in arms]}")

    # Model cache keyed by slot rank: LoraOps buffers are sized (B, S, r) at
    # construction, so one model per distinct rank. Built once and reused across
    # arms and pool sizes — the base weights are stateless w.r.t. the adapters,
    # and reloading a multi-GB checkpoint per cell would dominate the runtime.
    models: dict[int, object] = {}

    def model_for(rank: int):
        if rank not in models:
            cfg = arm_config(args, uniform_arm(rank), device, dtype)
            model = (
                HFEncoderWithLora.from_pretrained_serving(cfg)
                if args.engine == "hf"
                else EncoderWithLora.from_pretrained_serving(cfg)
            )
            model.eval()
            models[rank] = model
        return models[rank]

    scratch = arm_config(args, uniform_arm(max(m.r_max for m in mixes)), device, dtype)
    elem = torch.empty([], dtype=dtype).element_size()
    per_rank_bytes = 2 * scratch.num_layers * scratch.hidden_size * len(
        args.target_modules
    ) * elem
    print("\nPredicted store size (pre-padded layout, the largest of the arms):")
    for n in args.adapters:
        sizes = "  ".join(
            f"r={rank}: {n * per_rank_bytes * rank / 1e9:5.1f} GB"
            for rank in sorted({a.slot_rank for a in arms})
        )
        print(f"  N={n:>7,}   {sizes}")

    # LR heads are rank-independent and shared by every arm, so the exactness
    # gate compares adapters rather than two different classification heads.
    torch.manual_seed(LR_SEED)
    lr_store = {
        f"adapter_{i}": make_synthetic_lr_weights(scratch, args.num_labels)
        for i in range(max(args.adapters))
    }
    # Only assemble_lr is used, and it reads neither the store nor the config;
    # an empty store keeps this from being mistaken for an arm's adapter source.
    lr_assembler = BatchAssembler(AdapterStore(scratch), scratch)

    def lr_for(ids: list[str]):
        return lr_assembler.assemble_lr(
            [lr_store[a][0] for a in ids], [lr_store[a][1] for a in ids]
        )

    print("\n=== Exactness gate (padding must not change the served result) ===")
    exactness = []
    for mix in mixes:
        e = check_exactness(args, mix, device, dtype, model_for, lr_for,
                            args.check_adapters)
        print(f"  mix {mix.label:<12} padded-vs-native "
              f"max|delta|={e['max_abs_delta_padded_vs_native']:.2e} [fp32]  "
              f"layouts={e['max_abs_delta_layouts']:.2e}  OK "
              f"({args.dtype} padded-vs-native, reported only: "
              f"{e['max_abs_delta_padded_vs_native_serving_dtype']:.2e})")
        exactness.append(e)

    seeds: list[int | None] = args.seeds if args.seeds else [None]
    per_seed: dict[tuple[int, str], list[dict]] = {}
    oomed: list[tuple[int, str, object]] = []
    total_cells = len(seeds) * len(args.adapters) * len(arms)
    done = 0

    inputs = make_synthetic_inputs(scratch, args.batch_size)
    output_lr = torch.zeros(
        args.batch_size, 1, args.num_labels, dtype=dtype, device=device
    )

    # Seed outermost, matching latefuse.benchmark.run: each pass is a full
    # sweep, so between-seed spread also absorbs thermal and clock drift rather
    # than only per-iteration jitter.
    for seed in seeds:
        for n in args.adapters:
            for arm in arms:
                done += 1
                print(f"\n[{done}/{total_cells}] N={n} arm={arm.name} seed={seed}")
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                config = arm_config(args, arm, device, dtype)
                native_of = {
                    f"adapter_{i}": r
                    for i, r in enumerate(assign_ranks(n, arm.mix))
                }
                adapter_ids = [f"adapter_{i}" for i in range(n)]
                assemble_fn = owner = row = None
                try:
                    assemble_fn, store_bytes, owner = build_arm(args, arm, n, config)
                    row = time_arm(
                        model_for(arm.slot_rank), assemble_fn, adapter_ids,
                        native_of, arm.mix.r_max, lr_for, inputs, output_lr,
                        device, args.batch_size, args.warmup, args.iters,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    print(f"  OOM: {exc}")
                    oomed.append((n, arm.name, seed))
                finally:
                    # assemble_fn is a bound method and keeps the assembler (and
                    # so the packed weights) alive on its own; dropping `owner`
                    # alone would free nothing before the next arm allocates.
                    assemble_fn = owner = None
                    torch.cuda.empty_cache()
                if row is None:
                    continue
                row["store_gb"] = store_bytes / 1e9
                per_seed.setdefault((n, arm.name), []).append(row)
                print(f"  p50={row['total_p50_ms']:.2f}ms  "
                      f"fwd={row['forward_mean_ms']:.2f}ms  "
                      f"asm={row['assemble_mean_ms']:.3f}ms "
                      f"({row['assemble_share_pct']:.1f}%)  "
                      f"tput={row['throughput_samples_sec']:.0f}/s  "
                      f"store={row['store_gb']:.2f}GB")

    if not per_seed:
        print("\nNo cell completed (everything OOM'd?). Nothing written.")
        raise SystemExit(1)

    # aggregate() takes a list of per-seed {variant: metrics}; there is one
    # "variant" per (N, arm) here, so each key is aggregated on its own.
    results = {
        key: aggregate([{"arm": row} for row in rows])["arm"]
        for key, rows in per_seed.items()
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, args, env, seeds, mixes, results, exactness)

    complete = {
        (n, m.label)
        for m in mixes
        for n in args.adapters
        if all((n, name) in results for name in mix_arm_names(m))
    }
    payload = {
        "gpu": env["gpu"],
        "env": env,
        "config": vars(args),
        "seeds": seeds,
        "mixes": [
            {"label": m.label, "ranks": m.ranks, "fracs": m.fracs,
             "r_min": m.r_min, "r_max": m.r_max}
            for m in mixes
        ],
        "exactness": exactness,
        "oom_cells": [{"n": n, "arm": a, "seed": s} for n, a, s in oomed],
        "aggregated": {
            f"{n}|{name}": metrics for (n, name), metrics in results.items()
        },
        "derived": {
            f"{n}|{label}": derived(
                {name: results[(n, name)] for name in mix_arm_names(m)}, m
            )
            for m in mixes
            for n in args.adapters
            for label in [m.label]
            if (n, label) in complete
        },
        "per_seed": {f"{n}|{name}": rows for (n, name), rows in per_seed.items()},
    }
    json_path = out.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    if oomed:
        print(f"\n{len(oomed)} cell(s) OOM'd and are absent from the report:")
        for n, name, seed in oomed:
            print(f"    N={n} arm={name} seed={seed}")

    print(f"\nWrote {out} and {json_path}")


if __name__ == "__main__":
    main()
