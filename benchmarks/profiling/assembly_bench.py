"""Does a GPU-resident index_select gather (and torch.compile) fix the
CPU-side batch-assembly bottleneck of Finding 5?

The paper's serving path assembles a mixed-tenant batch with a Python loop over
samples x layers x modules (:class:`BatchAssembler`), which is the residual
bottleneck and the driver of tail latency. This benchmark measures three
assembly strategies under an otherwise identical forward pass:

  baseline   - BatchAssembler: O(B*L*|M|) Python loop building per-layer lists
  indexsel   - IndexSelectBatchAssembler: |M| index_select calls over a packed
               (N, L, H, R) adapter tensor; only an O(B) id->row lookup is Python
  indexsel+compile - same gather wrapped in torch.compile (the id->row lookup is
               kept eager, since Dynamo graph-breaks on the dict access)

For each batch size it reports (a) end-to-end latency (assemble + encoder
forward + LR head), the methodology-agnostic headline, (b) the assembly-only
split that connects to Finding 5, and (c) result scatter (the per-tenant slice +
softmax that unbatches logits back to each caller). Equivalence
(identical logits) is asserted before any timing, so the speedups are for a
numerically identical computation.

Run on the GPU node (matches benchmarks/run_sxm80.sh conventions):
    uv run python -m benchmarks.profiling.assembly_bench \\
        --model BAAI/bge-m3 --dtype fp16 --num-adapters 2000 \\
        --batch-sizes 8 16 32 64 128 --lora-rank 8 --seq-len 128 \\
        --warmup 50 --iters 200 --seeds 1 2 3 4 5 \\
        --out benchmarks/results/assembly_bench.txt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import (
    BatchAssembler,
    IndexSelectBatchAssembler,
    LayerwiseBatchedWeights,
)
from lora_serving.weights.store import AdapterStore

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def make_compiled_gather():
    """torch.compile'd index_select gather for one module.

    Only the tensor ops are compiled. The adapter-id -> row lookup that produces
    `idx` stays in eager Python on purpose: Dynamo graph-breaks on the dict
    access, so compiling around it would buy nothing and only obscures what
    torch.compile actually fuses (the gather)."""

    @torch.compile
    def gather(wa_m: torch.Tensor, wb_m: torch.Tensor, idx: torch.Tensor):
        return wa_m.index_select(0, idx), wb_m.index_select(0, idx)

    return gather


def to_layerwise_compiled(fast, gather, adapter_ids, config):
    """`IndexSelectBatchAssembler.to_layerwise` routed through a compiled gather."""
    idx = fast.batch_index(adapter_ids)
    layers = [LayerwiseBatchedWeights() for _ in range(config.num_layers)]
    for m in config.target_modules:
        a, b = gather(fast.wa[m], fast.wb[m], idx)
        for layer_idx in range(config.num_layers):
            layers[layer_idx].a[m] = [a[:, layer_idx]]
            layers[layer_idx].b[m] = [b[:, layer_idx]]
    return layers


def stats(latencies_ms: np.ndarray) -> dict:
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "p99_over_p50": float(
            np.percentile(latencies_ms, 99) / np.percentile(latencies_ms, 50)
        ),
    }


def assert_equivalent(model, inputs, lr_w, output_lr, lora_a, lora_b):
    """Run two assembled weight sets through the forward; assert identical logits."""
    with torch.no_grad():
        out_a = model(inputs["input_ids"], inputs["attention_mask"], lora_a, lr_w,
                      output_lr.clone()).clone()
        out_b = model(inputs["input_ids"], inputs["attention_mask"], lora_b, lr_w,
                      output_lr.clone()).clone()
    if not torch.allclose(out_a, out_b, atol=1e-3, rtol=1e-3):
        max_abs = (out_a - out_b).abs().max().item()
        raise AssertionError(
            f"index_select assembly diverged from baseline: max|delta|={max_abs:.3e}"
        )


def scatter_results(output_lr: torch.Tensor, class_counts: list[int]) -> list[torch.Tensor]:
    """Result scatter: unbatch the padded logits back to each tenant.

    After the batched forward writes logits into `output_lr` of shape
    (B, 1, c_max), each sample must be sliced to its tenant's real class count
    and softmaxed independently -- the operation the paper describes in
    Batched Classification Heads ("callers slice [:, :, :c_i] per sample before
    softmax, so the padded entries never enter the normalization"). This is the
    mirror image of the input gather, done per sample in a Python loop, so its
    cost -- like the assembler's -- scales with batch size B, not model FLOPs.

    Scope: this times the *compute-path* scatter (per-sample slice + softmax) on
    device, with a single sync by the caller, consistent with how the forward
    measurement also leaves its output on device. Host-side transfer of the
    per-tenant results and serialization back to each request is a serving-layer
    cost (like the network) and is excluded here, matching the forward's
    boundary. To include the device->host copy instead, append `.cpu()` /
    `.tolist()` inside the comprehension -- flagged as a review decision.
    """
    return [
        torch.softmax(output_lr[i, 0, :c], dim=-1)
        for i, c in enumerate(class_counts)
    ]


def bench_one_batch_size(
    store, baseline, fast, gather, adapter_ids, lr_store, config, device,
    batch_size, num_labels, warmup, iters, dtype, seed, check,
):
    """Build the model at `batch_size`, optionally verify equivalence, time all
    variants for a single seed. Returns variant -> flat scalar metrics dict."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    cfg = LoraServingConfig(
        model_name=config.model_name, lora_rank=config.lora_rank,
        batch_size=batch_size, max_seq_len=config.max_seq_len,
        target_modules=config.target_modules, device=device, dtype=dtype,
    )
    model = EncoderWithLora.from_pretrained_serving(cfg)
    model.eval()

    inputs = make_synthetic_inputs(cfg, batch_size)
    output_lr = torch.zeros(batch_size, 1, num_labels, dtype=dtype, device=device)

    # Per-tenant class counts (variable c_i in [2, num_labels]) so result scatter
    # exercises real per-sample slicing to different widths rather than a uniform
    # c_max. Fixed per (seed, batch_size): the scatter cost is dominated by the
    # B-length Python loop, not the exact c_i values, so a representative draw is
    # enough. Uses the seeded RNG above, so it is deterministic across variants.
    class_counts = [random.randint(2, num_labels) for _ in range(batch_size)]

    def sample_ids():
        return random.choices(adapter_ids, k=batch_size)

    def lr_for(ids):
        coefs = [lr_store[a][0] for a in ids]
        intercepts = [lr_store[a][1] for a in ids]
        return baseline.assemble_lr(coefs, intercepts)

    # --- correctness gate: identical logits for baseline vs both fast paths.
    # Run once (first seed) since the mapping is seed-independent. ---
    if check:
        ids = sample_ids()
        lr_w = lr_for(ids)
        assert_equivalent(model, inputs, lr_w, output_lr,
                          baseline.assemble_lora(ids), fast.to_layerwise(ids))
        assert_equivalent(model, inputs, lr_w, output_lr,
                          baseline.assemble_lora(ids),
                          to_layerwise_compiled(fast, gather, ids, cfg))

    variants = {
        "baseline": lambda ids: baseline.assemble_lora(ids),
        "indexsel": lambda ids: fast.to_layerwise(ids),
        "indexsel_compile": lambda ids: to_layerwise_compiled(fast, gather, ids, cfg),
    }

    results = {}
    for name, assemble in variants.items():
        # Warmup (also triggers torch.compile for the compiled variant, and warms
        # the softmax kernels used by the scatter step).
        for _ in range(warmup):
            ids = sample_ids()
            lora_w = assemble(ids)
            with torch.no_grad():
                model(inputs["input_ids"], inputs["attention_mask"], lora_w,
                      lr_for(ids), output_lr)
            scatter_results(output_lr, class_counts)
        torch.cuda.synchronize(device)

        asm_ms = np.empty(iters)
        total_ms = np.empty(iters)    # assemble + forward (incl LR head)
        scatter_ms = np.empty(iters)  # result scatter (per-tenant slice + softmax)
        e2e_ms = np.empty(iters)      # assemble + forward + scatter
        for i in range(iters):
            ids = sample_ids()
            lr_w = lr_for(ids)

            # End-to-end wall-clock: assemble + forward, syncs between stages so
            # each component is attributed to the right stage.
            t0 = time.perf_counter()
            lora_w = assemble(ids)
            # Assembly split: sync so the index_select gather kernel (if any) is
            # attributed to assembly, not to the forward timed just after.
            torch.cuda.synchronize(device)
            asm_ms[i] = (time.perf_counter() - t0) * 1000

            with torch.no_grad():
                model(inputs["input_ids"], inputs["attention_mask"], lora_w,
                      lr_w, output_lr)
            torch.cuda.synchronize(device)
            total_ms[i] = (time.perf_counter() - t0) * 1000

            # Result scatter: per-tenant slice + softmax over the logits just
            # written to output_lr. Timed separately, then folded into e2e.
            ts = time.perf_counter()
            scatter_results(output_lr, class_counts)
            torch.cuda.synchronize(device)
            scatter_ms[i] = (time.perf_counter() - ts) * 1000
            e2e_ms[i] = (time.perf_counter() - t0) * 1000

        t, a = stats(total_ms), stats(asm_ms)
        sc, e = stats(scatter_ms), stats(e2e_ms)
        # Flat per-seed scalars so main() can aggregate mean +/- s.d. across seeds.
        results[name] = {
            "total_mean_ms": t["mean_ms"],
            "total_p50_ms": t["p50_ms"],
            "total_p99_ms": t["p99_ms"],
            "p99_over_p50": t["p99_over_p50"],
            "assemble_mean_ms": a["mean_ms"],
            "assemble_p50_ms": a["p50_ms"],
            "assemble_share_pct": 100.0 * a["mean_ms"] / t["mean_ms"],
            "scatter_mean_ms": sc["mean_ms"],
            "scatter_p50_ms": sc["p50_ms"],
            "e2e_mean_ms": e["mean_ms"],
            # scatter as a share of the full assemble+forward+scatter path.
            "scatter_share_pct": 100.0 * sc["mean_ms"] / e["mean_ms"],
        }

    del model
    torch.cuda.empty_cache()
    return results


def aggregate(per_seed: list[dict]) -> dict:
    """Aggregate a list of per-seed {variant: {metric: value}} dicts into
    {variant: {metric: {mean, std}}} across seeds (population s.d., matching a
    single-run report when only one seed is present)."""
    variants = per_seed[0].keys()
    metrics = next(iter(per_seed[0].values())).keys()
    agg = {}
    for v in variants:
        agg[v] = {}
        for m in metrics:
            vals = np.array([s[v][m] for s in per_seed], dtype=float)
            agg[v][m] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return agg


def fmt(cell: dict, unit: str = "", prec: int = 3, multi: bool = True) -> str:
    """mean±std (or just mean when single-seed) for a {mean,std} metric cell."""
    if multi:
        return f"{cell['mean']:.{prec}f}±{cell['std']:.{prec}f}{unit}"
    return f"{cell['mean']:.{prec}f}{unit}"


def cgroup_cpu_quota() -> float | None:
    """Effective CPU limit from the cgroup, in cores. None if uncapped.

    Container runtimes usually cap CPU with a CFS quota rather than by pinning
    cores, so sched_getaffinity happily reports every core on the host while the
    process is throttled to a fraction of them -- on a RunPod A100 pod that is
    192 visible versus ~20 actually granted. For a CPU-bound benchmark this gap
    is the difference between a reproducible number and an unexplainable one.
    """
    try:  # cgroup v2
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw[0] != "max":
            return int(raw[0]) / int(raw[1])
    except (OSError, ValueError, IndexError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def env_metadata() -> dict:
    """Hardware/software provenance for the report header and JSON.

    The CPU fields carry as much weight as the GPU here: the baseline assembler
    is CPU-bound by construction, so the throughput ceiling it reports is a
    property of this host. Inside a container the usable core count can sit well
    below the physical one, and it is the usable count that actually bounds the
    per-sample gather.
    """
    cpu = platform.processor() or platform.machine()
    try:
        # Linux gives the real model string; platform.processor() only says "x86_64".
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        usable = len(os.sched_getaffinity(0))  # respects the container cpuset
    except AttributeError:  # not Linux
        usable = os.cpu_count() or 0
    return {
        "gpu": torch.cuda.get_device_name(0),
        "cpu": cpu,
        "cpu_cores_total": os.cpu_count() or 0,
        "cpu_cores_usable": usable,
        "cpu_cgroup_quota": cgroup_cpu_quota(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "n/a",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def cpu_line(env: dict) -> str:
    """One-line CPU description. The quota, not the visible count, is the bound."""
    quota = env["cpu_cgroup_quota"]
    cap = f"cgroup quota {quota:.1f}" if quota else "uncapped"
    return (f"{env['cpu']} ({env['cpu_cores_usable']}/{env['cpu_cores_total']} "
            f"cores visible, {cap})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument("--num-adapters", type=int, default=2000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=10)
    # 50/200 is the house protocol, shared with lora_serving.benchmark.run and
    # every generality sweep, so assembly numbers compose with the model sweeps.
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Repeat the full batch-size sweep once per seed (seed is the "
             "outermost loop, matching lora_serving.benchmark.run, so "
             "between-seed variance also captures thermal/clock drift). Results "
             "are aggregated to mean +/- s.d. per (batch_size, variant). Default "
             "None: single unseeded run.",
    )
    parser.add_argument("--out", default="benchmarks/results/assembly_bench.txt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Need CUDA. Aborting.")
        return

    device = torch.device("cuda:0")
    dtype = DTYPE_MAP[args.dtype]
    env = env_metadata()
    print(f"GPU: {env['gpu']}")
    print(f"CPU: {cpu_line(env)}")

    # Config used only to build the store/assemblers (batch_size irrelevant here;
    # the per-batch-size model is rebuilt inside bench_one_batch_size).
    config = LoraServingConfig(
        model_name=args.model, lora_rank=args.lora_rank, batch_size=max(args.batch_sizes),
        max_seq_len=args.seq_len, target_modules=["query", "value"],
        device=device, dtype=dtype,
    )

    print(f"Generating {args.num_adapters} synthetic adapters (rank={args.lora_rank})...")
    store = AdapterStore(config)
    make_synthetic_adapters(store, args.num_adapters)
    adapter_ids = [f"adapter_{i}" for i in range(args.num_adapters)]
    lr_store = {a: make_synthetic_lr_weights(config, args.num_labels) for a in adapter_ids}

    baseline = BatchAssembler(store, config)
    print("Packing AdapterStore into contiguous (N, L, H, R) tensors...")
    t0 = time.perf_counter()
    fast = IndexSelectBatchAssembler(store, config)
    print(f"  packed in {time.perf_counter() - t0:.2f}s; "
          f"store={store.memory_gb():.2f} GB, packed adds ~same")
    gather = make_compiled_gather()

    seeds: list[int | None] = args.seeds if args.seeds else [None]
    multi = len(seeds) > 1
    # per_seed_results[bs] = list (one entry per seed) of {variant: metrics}
    per_seed_results: dict[int, list[dict]] = {bs: [] for bs in args.batch_sizes}

    # Seed outermost: each pass is a full sweep over batch sizes, so between-seed
    # variance captures slow drift in addition to per-iter jitter.
    for si, seed in enumerate(seeds):
        for bs in args.batch_sizes:
            print(f"\n=== seed={seed} batch_size={bs} ===")
            res = bench_one_batch_size(
                store, baseline, fast, gather, adapter_ids, lr_store, config, device,
                batch_size=bs, num_labels=args.num_labels,
                warmup=args.warmup, iters=args.iters, dtype=dtype,
                seed=seed, check=(si == 0),  # equivalence gate once is enough
            )
            per_seed_results[bs].append(res)
            for name, r in res.items():
                print(f"  {name:18s} total mean={r['total_mean_ms']:.3f}ms "
                      f"assemble mean={r['assemble_mean_ms']:.3f}ms "
                      f"({r['assemble_share_pct']:.1f}%) "
                      f"scatter mean={r['scatter_mean_ms']:.3f}ms "
                      f"({r['scatter_share_pct']:.1f}%) "
                      f"p99/p50={r['p99_over_p50']:.2f}x")

    agg = {bs: aggregate(per_seed_results[bs]) for bs in args.batch_sizes}

    # ---- report ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_str = ",".join(str(s) for s in seeds) if multi else "single run"
    with out.open("w") as f:
        f.write(f"# Batch-assembly strategies — {args.model}\n")
        f.write(f"# GPU: {env['gpu']}\n")
        # The baseline arm is CPU-bound, so the host CPU is part of the result.
        f.write(f"# CPU: {cpu_line(env)}\n")
        f.write(f"# torch {env['torch_version']} / CUDA {env['cuda_version']}\n")
        f.write(f"# N={args.num_adapters} r={args.lora_rank} seq={args.seq_len} "
                f"dtype={args.dtype} warmup={args.warmup} iters={args.iters} "
                f"seeds={seed_str}\n")
        f.write("# Values are mean±s.d. across seeds.\n\n" if multi else "\n")

        f.write("## End-to-end latency (assemble + forward + LR head), mean ms\n")
        f.write(f"{'batch':>6}  {'baseline':>16}  {'indexsel':>16}  {'idx+comp':>16}  "
                f"{'speedup':>8}\n")
        for bs in args.batch_sizes:
            r = agg[bs]
            b = r["baseline"]["total_mean_ms"]
            i = r["indexsel"]["total_mean_ms"]
            c = r["indexsel_compile"]["total_mean_ms"]
            f.write(f"{bs:>6}  {fmt(b, multi=multi):>16}  {fmt(i, multi=multi):>16}  "
                    f"{fmt(c, multi=multi):>16}  {b['mean'] / i['mean']:>7.2f}x\n")

        f.write("\n## Assembly-only time (Finding 5 quantity), mean ms\n")
        f.write(f"{'batch':>6}  {'baseline':>16}  {'indexsel':>16}  {'idx+comp':>16}\n")
        for bs in args.batch_sizes:
            r = agg[bs]
            f.write(f"{bs:>6}  {fmt(r['baseline']['assemble_mean_ms'], multi=multi):>16}  "
                    f"{fmt(r['indexsel']['assemble_mean_ms'], multi=multi):>16}  "
                    f"{fmt(r['indexsel_compile']['assemble_mean_ms'], multi=multi):>16}\n")

        f.write("\n## Assembly share of end-to-end (%)\n")
        f.write(f"{'batch':>6}  {'baseline':>16}  {'indexsel':>16}  {'idx+comp':>16}\n")
        for bs in args.batch_sizes:
            r = agg[bs]
            f.write(f"{bs:>6}  {fmt(r['baseline']['assemble_share_pct'], '%', 1, multi):>16}  "
                    f"{fmt(r['indexsel']['assemble_share_pct'], '%', 1, multi):>16}  "
                    f"{fmt(r['indexsel_compile']['assemble_share_pct'], '%', 1, multi):>16}\n")

        f.write("\n## Result scatter (per-tenant slice + softmax), mean ms\n")
        f.write(f"{'batch':>6}  {'baseline':>16}  {'indexsel':>16}  {'idx+comp':>16}\n")
        for bs in args.batch_sizes:
            r = agg[bs]
            f.write(f"{bs:>6}  {fmt(r['baseline']['scatter_mean_ms'], multi=multi):>16}  "
                    f"{fmt(r['indexsel']['scatter_mean_ms'], multi=multi):>16}  "
                    f"{fmt(r['indexsel_compile']['scatter_mean_ms'], multi=multi):>16}\n")

        f.write("\n## Scatter share of end-to-end incl. scatter (%)\n")
        f.write(f"{'batch':>6}  {'baseline':>16}  {'indexsel':>16}  {'idx+comp':>16}\n")
        for bs in args.batch_sizes:
            r = agg[bs]
            f.write(f"{bs:>6}  {fmt(r['baseline']['scatter_share_pct'], '%', 1, multi):>16}  "
                    f"{fmt(r['indexsel']['scatter_share_pct'], '%', 1, multi):>16}  "
                    f"{fmt(r['indexsel_compile']['scatter_share_pct'], '%', 1, multi):>16}\n")

        f.write("\n## Tail ratio p99/p50 (end-to-end)\n")
        f.write(f"{'batch':>6}  {'baseline':>16}  {'indexsel':>16}  {'idx+comp':>16}\n")
        for bs in args.batch_sizes:
            r = agg[bs]
            f.write(f"{bs:>6}  {fmt(r['baseline']['p99_over_p50'], 'x', 2, multi):>16}  "
                    f"{fmt(r['indexsel']['p99_over_p50'], 'x', 2, multi):>16}  "
                    f"{fmt(r['indexsel_compile']['p99_over_p50'], 'x', 2, multi):>16}\n")

        f.write("\nInterpretation:\n")
        f.write("- 'speedup' is baseline/indexsel end-to-end (ratio of seed means).\n")
        f.write("  >1 means the index_select gather reduced total latency.\n")
        f.write("- 'idx+comp' vs 'indexsel' isolates torch.compile's MARGINAL\n")
        f.write("  effect on the already-vectorized gather (the id->row Python\n")
        f.write("  lookup is eager in both, so this is the pure fusion upside).\n")
        f.write("- Assembly share collapsing toward 0% means the CPU-side\n")
        f.write("  bottleneck is gone; the forward then bounds latency.\n")
        f.write("- 'Result scatter' is the per-tenant slice + softmax that\n")
        f.write("  unbatches logits; it is variant-independent\n")
        f.write("  (same op regardless of assembler), timed on-device. Host-side\n")
        f.write("  transfer/serialization to callers is serving-layer, excluded\n")
        f.write("  here to match the forward's boundary.\n")

    json_path = out.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump({
            "gpu": env["gpu"],  # kept at top level for back-compat with earlier runs
            "env": env,
            "config": vars(args),
            "seeds": seeds,
            "aggregated": {str(bs): agg[bs] for bs in args.batch_sizes},
            "per_seed": {str(bs): per_seed_results[bs] for bs in args.batch_sizes},
        }, f, indent=2)

    print(f"\nWrote {out} and {json_path}")


if __name__ == "__main__":
    main()
