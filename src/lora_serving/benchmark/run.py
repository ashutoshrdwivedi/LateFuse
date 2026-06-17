"""Standalone benchmark for multi-tenant LoRA serving on a GPU.

Run on RunPod (or any GPU machine):
    pip install uv
    uv sync
    uv run python -m lora_serving.benchmark.run

Sweep options:
    uv run python -m lora_serving.benchmark.run \\
        --model BAAI/bge-m3 \\
        --dtype fp16 \\
        --adapters 1000 5000 10000 20000 \\
        --batch-sizes 16 32 64 128 \\
        --lora-ranks 8 16 32 \\
        --seq-len 128 \\
        --warmup 50 \\
        --iters 200 \\
        --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
import torch

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def per_adapter_bytes(config: LoraServingConfig) -> int:
    """Predicted on-device bytes for one adapter: M target modules × 2 (A,B) × L × H × R × dtype_size."""
    elem_size = torch.empty([], dtype=config.dtype).element_size()
    return (
        len(config.target_modules) * 2 * config.num_layers
        * config.hidden_size * config.lora_rank * elem_size
    )


def memory_ceiling(
    config: LoraServingConfig,
    vram_budget_gb: float,
    model_overhead_gb: float = 2.5,
) -> tuple[int, int]:
    """Max adapters that fit in (vram_budget - model_overhead). Returns (count, bytes_per_adapter)."""
    bpa = per_adapter_bytes(config)
    available_bytes = (vram_budget_gb - model_overhead_gb) * 1e9
    return max(0, int(available_bytes / bpa)), bpa


def print_memory_ceiling(model_name: str, dtype: torch.dtype, vram_gb: float) -> None:
    """At startup: predict adapter ceiling for a representative config so the run is self-documenting."""
    print(f"\n=== Memory ceiling (predicted) ===")
    print(f"  Model: {model_name}  dtype: {dtype}  VRAM budget: {vram_gb:.1f} GB")
    for rank in (4, 8, 16, 32):
        cfg = LoraServingConfig(
            model_name=model_name,
            lora_rank=rank,
            batch_size=1,
            max_seq_len=128,
            target_modules=["query", "value"],
            device=torch.device("cuda:0"),
            dtype=dtype,
        )
        max_n, bpa = memory_ceiling(cfg, vram_gb)
        print(f"  rank={rank:<3} → {bpa/1024:>7.1f} KB/adapter  →  ~{max_n:>7,} adapters fit")
    print()


def run_single_config(
    model_name: str,
    num_adapters: int,
    batch_size: int,
    lora_rank: int,
    seq_len: int,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    num_labels: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one benchmark configuration and return metrics (including assembly/forward split)."""
    device = torch.device("cuda:0")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    config = LoraServingConfig(
        model_name=model_name,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_seq_len=seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )

    print(f"  Loading base model ({model_name}, {dtype})...")
    model = EncoderWithLora.from_pretrained_serving(config)
    model.eval()

    print(f"  Generating {num_adapters} synthetic adapters (rank={lora_rank})...")
    store = AdapterStore(config)
    t0 = time.perf_counter()
    make_synthetic_adapters(store, num_adapters)
    adapter_load_total_s = time.perf_counter() - t0
    print(f"  AdapterStore preload x{num_adapters} took {adapter_load_total_s:.2f}s",
          flush=True)
    adapter_ids = [f"adapter_{i}" for i in range(num_adapters)]

    lr_coefs = {}
    lr_intercepts = {}
    for aid in adapter_ids:
        coef, intercept = make_synthetic_lr_weights(config, num_labels)
        lr_coefs[aid] = coef
        lr_intercepts[aid] = intercept

    assembler = BatchAssembler(store, config)
    inputs = make_synthetic_inputs(config, batch_size)
    output_lr = torch.zeros(batch_size, 1, num_labels, dtype=dtype, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Reusable CUDA events for forward timing (avoids per-iter allocation cost)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)

    def run_batch_timed() -> tuple[float, float]:
        """Returns (assemble_ms, forward_ms). Assembly is CPU wall-clock; forward is CUDA-event timed."""
        batch_ids = random.choices(adapter_ids, k=batch_size)
        coefs = [lr_coefs[aid] for aid in batch_ids]
        intercepts = [lr_intercepts[aid] for aid in batch_ids]

        t0 = time.perf_counter()
        lora_w, lr_w = assembler.assemble(batch_ids, coefs, intercepts)
        assemble_ms = (time.perf_counter() - t0) * 1000

        fwd_start.record()
        with torch.no_grad():
            model(
                inputs["input_ids"],
                inputs["attention_mask"],
                lora_w,
                lr_w,
                output_lr,
            )
        fwd_end.record()
        torch.cuda.synchronize(device)
        forward_ms = fwd_start.elapsed_time(fwd_end)
        return assemble_ms, forward_ms

    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        run_batch_timed()

    print(f"  Measuring ({iters} iters)...")
    assemble_latencies = np.empty(iters)
    forward_latencies = np.empty(iters)
    for i in range(iters):
        assemble_latencies[i], forward_latencies[i] = run_batch_timed()

    total_latencies = assemble_latencies + forward_latencies
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    adapter_mem_gb = store.memory_gb()
    throughput = batch_size / (np.mean(total_latencies) / 1000)

    return {
        "model": model_name,
        "dtype": str(dtype).replace("torch.", ""),
        "num_adapters": num_adapters,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
        "p50_ms": round(float(np.percentile(total_latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(total_latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(total_latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(total_latencies, 99)), 2),
        "mean_ms": round(float(np.mean(total_latencies)), 2),
        "assemble_p50_ms": round(float(np.percentile(assemble_latencies, 50)), 3),
        "assemble_mean_ms": round(float(np.mean(assemble_latencies)), 3),
        "forward_p50_ms": round(float(np.percentile(forward_latencies, 50)), 3),
        "forward_mean_ms": round(float(np.mean(forward_latencies)), 3),
        "assemble_share_pct": round(100 * np.mean(assemble_latencies) / np.mean(total_latencies), 1),
        "throughput_samples_sec": round(throughput, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "adapter_cache_gb": round(adapter_mem_gb, 3),
        "adapter_load_total_s": round(adapter_load_total_s, 2),
        "warmup": warmup,
        "iters": iters,
        "seed": seed if seed is not None else "",
    }


def print_table(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description="Multi-tenant LoRA serving benchmark")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="fp16")
    parser.add_argument(
        "--adapters", nargs="+", type=int,
        default=[100, 1000, 5000, 10000, 20000, 50000],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--extra-configs", nargs="*", default=None, metavar="N:B:r",
        help="Additional explicit num_adapters:batch_size:lora_rank configs to "
             "measure in the SAME run as the --adapters x --batch-sizes x "
             "--lora-ranks grid. Folds the rank sweep's off-axis points (e.g. "
             "the r!=8 operating-point cells) into the main sweep so the shared "
             "r=8 cell is measured exactly once and all ranks at an operating "
             "point share one thermal envelope. Example: "
             "--extra-configs 1000:32:4 1000:32:16 1000:32:32",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Repeat the full sweep once per seed (seeds RNG for batch "
             "sampling / inputs / torch per config) and write all repeats to "
             "one CSV with a 'seed' column, for between-run variance "
             "estimates. Default None: single unseeded run (original "
             "behaviour).",
    )
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Benchmark requires a GPU.")
        return

    dtype = DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print_memory_ceiling(args.model, dtype, vram_gb)

    seeds: list[int | None] = args.seeds if args.seeds else [None]

    # Build the explicit (N, B, r) config list: the cross product of the sweep
    # axes plus any --extra-configs, de-duplicated and sorted by (N, B, r).
    # Sorting groups every rank measured at a given (N, B) consecutively, so the
    # rank variations at an operating point are timed back-to-back within each
    # seed pass (one thermal envelope) rather than in a separate, later run.
    configs = {
        (n, b, r)
        for n in args.adapters
        for b in args.batch_sizes
        for r in args.lora_ranks
    }
    for spec in args.extra_configs or []:
        parts = spec.split(":")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            parser.error(f"--extra-configs entry must be N:B:r integers, got {spec!r}")
        configs.add(tuple(int(p) for p in parts))
    configs = sorted(configs)

    print(f"Sweep: {len(configs)} unique (N, B, r) configs × {len(seeds)} seed(s)\n")

    results = []
    total = len(configs) * len(seeds)
    done = 0

    # Seed is the outermost loop: each repeat is a full pass over the sweep,
    # so between-seed variance also captures slow drift (thermal, clocks).
    for seed in seeds:
        for num_adapters, batch_size, lora_rank in configs:
            done += 1
            print(f"[{done}/{total}] adapters={num_adapters} batch={batch_size} "
                  f"rank={lora_rank} seed={seed}")
            try:
                row = run_single_config(
                    model_name=args.model,
                    num_adapters=num_adapters,
                    batch_size=batch_size,
                    lora_rank=lora_rank,
                    seq_len=args.seq_len,
                    warmup=args.warmup,
                    iters=args.iters,
                    dtype=dtype,
                    num_labels=args.num_labels,
                    seed=seed,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM at adapters={num_adapters} batch={batch_size} rank={lora_rank}: {e}")
                torch.cuda.empty_cache()
                continue
            results.append(row)
            print(f"  p50={row['p50_ms']}ms  p95={row['p95_ms']}ms  p99={row['p99_ms']}ms  "
                  f"asm={row['assemble_mean_ms']}ms ({row['assemble_share_pct']}%)  "
                  f"fwd={row['forward_mean_ms']}ms  "
                  f"tput={row['throughput_samples_sec']} samples/s  "
                  f"peak_gpu={row['peak_gpu_mem_gb']}GB\n")

    if not results:
        print("No results collected (all configs OOM'd?)")
        return

    print("\n=== Results ===")
    print_table(results)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
