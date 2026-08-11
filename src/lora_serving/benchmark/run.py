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
import os
import random
import time
from collections import OrderedDict
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
from lora_serving.model.hf_wrapper import HFEncoderWithLora
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


def print_memory_ceiling(
    model_name: str,
    dtype: torch.dtype,
    vram_gb: float,
    target_modules: list[str],
) -> None:
    """At startup: predict adapter ceiling for a representative config so the run is self-documenting."""
    print(f"\n=== Memory ceiling (predicted) ===")
    print(f"  Model: {model_name}  dtype: {dtype}  VRAM budget: {vram_gb:.1f} GB")
    print(f"  Target modules: {target_modules}")
    for rank in (4, 8, 16, 32):
        cfg = LoraServingConfig(
            model_name=model_name,
            lora_rank=rank,
            batch_size=1,
            max_seq_len=128,
            target_modules=target_modules,
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
    engine: str = "custom",
    target_modules: list[str] | None = None,
    churn_num_tenants: int | None = None,
    churn_zipf_alpha: float = 1.1,
) -> dict:
    """Run one benchmark configuration and return assembly, forward, and churn metrics.

    ``num_adapters`` is the fixed resident AdapterStore capacity.  Setting
    ``churn_num_tenants`` larger than that capacity switches batch selection to
    a Zipf request stream over the larger tenant population.  A request miss
    evicts the least-recently-used resident adapter and registers the requested
    adapter before that batch is assembled and served.
    """
    device = torch.device("cuda:0")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    target_modules = target_modules or ["query", "value"]
    if churn_num_tenants is not None and churn_num_tenants < num_adapters:
        raise ValueError(
            "churn_num_tenants must be at least num_adapters (the resident capacity)."
        )
    if churn_num_tenants is not None and churn_zipf_alpha <= 0:
        raise ValueError("churn_zipf_alpha must be positive.")

    config = LoraServingConfig(
        model_name=model_name,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_seq_len=seq_len,
        target_modules=target_modules,
        device=device,
        dtype=dtype,
    )

    print(f"  Loading base model ({model_name}, {dtype}, engine={engine})...")
    if engine == "hf":
        model = HFEncoderWithLora.from_pretrained_serving(config)
    else:
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

    churn_enabled = churn_num_tenants is not None
    resident: OrderedDict[str, None] | None = None
    churn_rng: np.random.Generator | None = None
    churn_probabilities: np.ndarray | None = None
    if churn_enabled:
        resident = OrderedDict((adapter_id, None) for adapter_id in adapter_ids)
        churn_rng = np.random.default_rng(seed)
        ranks = np.arange(1, churn_num_tenants + 1, dtype=np.float64)
        churn_probabilities = 1.0 / np.power(ranks, churn_zipf_alpha)
        churn_probabilities /= churn_probabilities.sum()

    assembler = BatchAssembler(store, config)
    inputs = make_synthetic_inputs(config, batch_size)
    output_lr = torch.zeros(batch_size, 1, num_labels, dtype=dtype, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Reusable CUDA events for forward timing (avoids per-iter allocation cost)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)

    def run_batch_timed() -> tuple[float, float, float, float, int, int]:
        """Returns assembly, forward, registration, eviction, hits, and misses."""
        registration_ms = 0.0
        eviction_ms = 0.0
        hits = 0
        misses = 0
        if churn_enabled:
            assert resident is not None
            assert churn_rng is not None
            assert churn_probabilities is not None
            batch_indices = churn_rng.choice(
                churn_num_tenants, size=batch_size, p=churn_probabilities
            )
            batch_ids = [f"adapter_{int(i)}" for i in batch_indices]
            for adapter_id in batch_ids:
                if adapter_id in resident:
                    hits += 1
                    resident.move_to_end(adapter_id)
                    continue

                misses += 1
                if len(resident) == num_adapters:
                    victim, _ = resident.popitem(last=False)
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    store.evict(victim)
                    lr_coefs.pop(victim)
                    lr_intercepts.pop(victim)
                    torch.cuda.synchronize(device)
                    eviction_ms += (time.perf_counter() - t0) * 1000

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                tenant_index = int(adapter_id.removeprefix("adapter_"))
                store.load_synthetic(adapter_id, seed=(seed or 0) + tenant_index)
                coef, intercept = make_synthetic_lr_weights(config, num_labels)
                lr_coefs[adapter_id] = coef
                lr_intercepts[adapter_id] = intercept
                torch.cuda.synchronize(device)
                registration_ms += (time.perf_counter() - t0) * 1000
                resident[adapter_id] = None
        else:
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
        return assemble_ms, forward_ms, registration_ms, eviction_ms, hits, misses

    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        run_batch_timed()

    print(f"  Measuring ({iters} iters)...")
    assemble_latencies = np.empty(iters)
    forward_latencies = np.empty(iters)
    registration_latencies = np.empty(iters)
    eviction_latencies = np.empty(iters)
    churn_hits = 0
    churn_misses = 0
    for i in range(iters):
        (
            assemble_latencies[i], forward_latencies[i], registration_latencies[i],
            eviction_latencies[i], hits, misses,
        ) = run_batch_timed()
        churn_hits += hits
        churn_misses += misses

    total_latencies = assemble_latencies + forward_latencies
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    adapter_mem_gb = store.memory_gb()
    throughput = batch_size / (np.mean(total_latencies) / 1000)
    serving_total_ms = float(np.sum(total_latencies))
    registration_total_ms = float(np.sum(registration_latencies))
    eviction_total_ms = float(np.sum(eviction_latencies))
    adapter_store_total_ms = registration_total_ms + eviction_total_ms
    adapter_store_latencies = registration_latencies + eviction_latencies
    end_to_end_latencies = total_latencies + adapter_store_latencies
    end_to_end_total_ms = float(np.sum(end_to_end_latencies))
    end_to_end_throughput = batch_size / (np.mean(end_to_end_latencies) / 1000)
    adapter_store_share_pcts = np.divide(
        100 * adapter_store_latencies,
        end_to_end_latencies,
        out=np.zeros_like(end_to_end_latencies),
        where=end_to_end_latencies != 0,
    )
    adapter_store_share_pct = (
        100 * adapter_store_total_ms / end_to_end_total_ms
        if end_to_end_total_ms else 0.0
    )

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
        "serving_total_ms": round(serving_total_ms, 3),
        "assemble_p50_ms": round(float(np.percentile(assemble_latencies, 50)), 3),
        "assemble_mean_ms": round(float(np.mean(assemble_latencies)), 3),
        "forward_p50_ms": round(float(np.percentile(forward_latencies, 50)), 3),
        "forward_mean_ms": round(float(np.mean(forward_latencies)), 3),
        "assemble_share_pct": round(100 * np.mean(assemble_latencies) / np.mean(total_latencies), 1),
        "throughput_samples_sec": round(throughput, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "adapter_cache_gb": round(adapter_mem_gb, 3),
        "adapter_load_total_s": round(adapter_load_total_s, 2),
        "churn_enabled": churn_enabled,
        "churn_num_tenants": churn_num_tenants if churn_enabled else "",
        "churn_zipf_alpha": churn_zipf_alpha if churn_enabled else "",
        "churn_request_hits": churn_hits if churn_enabled else "",
        "churn_request_misses": churn_misses if churn_enabled else "",
        "churn_admissions": churn_misses if churn_enabled else "",
        "churn_evictions": churn_misses if churn_enabled else "",
        "churn_registration_p50_ms": round(float(np.percentile(registration_latencies, 50)), 3) if churn_enabled else "",
        "churn_registration_mean_ms": round(float(np.mean(registration_latencies)), 3) if churn_enabled else "",
        "churn_registration_total_ms": round(registration_total_ms, 3) if churn_enabled else "",
        "churn_eviction_p50_ms": round(float(np.percentile(eviction_latencies, 50)), 3) if churn_enabled else "",
        "churn_eviction_mean_ms": round(float(np.mean(eviction_latencies)), 3) if churn_enabled else "",
        "churn_eviction_total_ms": round(eviction_total_ms, 3) if churn_enabled else "",
        "churn_adapter_store_total_ms": round(adapter_store_total_ms, 3) if churn_enabled else "",
        "churn_adapter_store_overall_share_pct": round(adapter_store_share_pct, 3) if churn_enabled else "",
        "churn_adapter_store_share_p50_pct": round(float(np.percentile(adapter_store_share_pcts, 50)), 3) if churn_enabled else "",
        "churn_adapter_store_share_p90_pct": round(float(np.percentile(adapter_store_share_pcts, 90)), 3) if churn_enabled else "",
        "churn_adapter_store_share_p95_pct": round(float(np.percentile(adapter_store_share_pcts, 95)), 3) if churn_enabled else "",
        "churn_adapter_store_share_p99_pct": round(float(np.percentile(adapter_store_share_pcts, 99)), 3) if churn_enabled else "",
        "churn_adapter_store_share_mean_pct": round(float(np.mean(adapter_store_share_pcts)), 3) if churn_enabled else "",
        "churn_end_to_end_p50_ms": round(float(np.percentile(end_to_end_latencies, 50)), 3) if churn_enabled else "",
        "churn_end_to_end_p90_ms": round(float(np.percentile(end_to_end_latencies, 90)), 3) if churn_enabled else "",
        "churn_end_to_end_p95_ms": round(float(np.percentile(end_to_end_latencies, 95)), 3) if churn_enabled else "",
        "churn_end_to_end_p99_ms": round(float(np.percentile(end_to_end_latencies, 99)), 3) if churn_enabled else "",
        "churn_end_to_end_mean_ms": round(float(np.mean(end_to_end_latencies)), 3) if churn_enabled else "",
        "churn_end_to_end_total_ms": round(end_to_end_total_ms, 3) if churn_enabled else "",
        "churn_end_to_end_throughput_samples_sec": round(end_to_end_throughput, 1) if churn_enabled else "",
        "warmup": warmup,
        "iters": iters,
        "seed": seed if seed is not None else "",
        "engine": engine,
        "target_modules": "+".join(target_modules),
        # Allocator provenance, recorded per row rather than inferred from the
        # filename. expandable_segments changes the achievable adapter ceiling
        # (the L40S row: 26k default vs 28k with it, +7.7% at identical p50), so
        # a capacity number is only interpretable alongside the setting it was
        # measured under. Without this column the two are told apart only by
        # which file they landed in, which is exactly how the L40S results ended
        # up split across four CSVs whose provenance lived in a comment.
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "status": "ok",
    }


# Metric columns, in CSV order. A measured row fills them; an OOM row leaves
# them blank. Both rows carry the same keys so the two write one shared header.
METRIC_COLUMNS = (
    "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "serving_total_ms",
    "assemble_p50_ms", "assemble_mean_ms", "forward_p50_ms", "forward_mean_ms",
    "assemble_share_pct", "throughput_samples_sec", "peak_gpu_mem_gb",
    "adapter_cache_gb", "adapter_load_total_s",
)


def oom_row(
    model_name: str,
    dtype,
    num_adapters: int,
    batch_size: int,
    lora_rank: int,
    seq_len: int,
    warmup: int,
    iters: int,
    seed,
    engine: str,
    target_modules,
    churn_num_tenants: int | None = None,
    churn_zipf_alpha: float = 1.1,
) -> dict:
    """A row recording that this config was attempted and ran out of memory.

    Written instead of skipping, because the absence of a row is ambiguous: it
    cannot distinguish "OOM'd here" from "never asked for". That ambiguity is
    why a capacity probe's ceiling had to be inferred from a max() over
    whatever happened to survive, and why the committed probe CSVs could not be
    reconciled against the grids their scripts requested.

    Metric columns are blank. Consumers must skip status != "ok" before
    aggregating -- a blank p50 is not a zero.
    """
    row = {
        "model": model_name,
        "dtype": str(dtype).replace("torch.", ""),
        "num_adapters": num_adapters,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
    }
    row.update(dict.fromkeys(METRIC_COLUMNS, ""))
    row.update({
        "warmup": warmup,
        "iters": iters,
        "seed": seed if seed is not None else "",
        "engine": engine,
        "target_modules": "+".join(target_modules),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "status": "oom",
        "churn_enabled": churn_num_tenants is not None,
        "churn_num_tenants": churn_num_tenants if churn_num_tenants is not None else "",
        "churn_zipf_alpha": churn_zipf_alpha if churn_num_tenants is not None else "",
        "churn_request_hits": "",
        "churn_request_misses": "",
        "churn_admissions": "",
        "churn_evictions": "",
        "churn_registration_p50_ms": "",
        "churn_registration_mean_ms": "",
        "churn_registration_total_ms": "",
        "churn_eviction_p50_ms": "",
        "churn_eviction_mean_ms": "",
        "churn_eviction_total_ms": "",
        "churn_adapter_store_total_ms": "",
        "churn_adapter_store_overall_share_pct": "",
        "churn_adapter_store_share_p50_pct": "",
        "churn_adapter_store_share_p90_pct": "",
        "churn_adapter_store_share_p95_pct": "",
        "churn_adapter_store_share_p99_pct": "",
        "churn_adapter_store_share_mean_pct": "",
        "churn_end_to_end_p50_ms": "",
        "churn_end_to_end_p90_ms": "",
        "churn_end_to_end_p95_ms": "",
        "churn_end_to_end_p99_ms": "",
        "churn_end_to_end_mean_ms": "",
        "churn_end_to_end_total_ms": "",
        "churn_end_to_end_throughput_samples_sec": "",
    })
    return row


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
    parser.add_argument(
        "--engine", choices=["custom", "hf"], default="hf",
        help="Base-model forward: 'custom' = the repo's EncoderWithLora "
             "(BERT-family reimplementation, strict weight load); 'hf' = stock "
             "HuggingFace AutoModel with LoRA injected via forward hooks "
             "(HFEncoderWithLora) — required for architectures the custom "
             "encoder cannot load (e.g. DeBERTa-v2's disentangled attention).",
    )
    parser.add_argument(
        "--target-modules", nargs="+", default=["query", "value"],
        help="Attention projections to attach LoRA to. Names are the logical "
             "'query'/'key'/'value' regardless of the checkpoint's own naming "
             "(the HF engine maps them per architecture, e.g. DeBERTa-v2's "
             "query_proj/value_proj). Per-adapter memory — and hence the "
             "tenant ceiling — scales with the number of modules.",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--churn",
        action="store_true",
        help="Enable the Zipf/LRU tenant-churn path. Requires "
        "--churn-num-tenants; leave unset for the normal resident-store benchmark.",
    )
    parser.add_argument(
        "--churn-num-tenants", type=int,
        help="Zipf tenant population used by --churn. Each --adapters value "
        "remains the fixed resident-store capacity.",
    )
    parser.add_argument(
        "--churn-zipf-alpha", type=float, default=1.1,
        help="Zipf popularity exponent used when --churn-num-tenants is set.",
    )
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
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of truncating it, writing the header only "
             "if the file is new or empty. Lets one script emit several arms "
             "into a single CSV -- the capacity probe in "
             "benchmarks/run_l40s.sh runs once per allocator and needs "
             "both arms in one file, told apart by the 'alloc_conf' column. "
             "Use it only for arms of the same logical result: the point is one "
             "file per artifact, not a scratch pad that survives a rerun.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if any config OOM'd, i.e. if the CSV is missing "
             "cells the grid asked for. Use it on sweeps, whose grids are "
             "chosen to fit -- a gap there is a fault worth failing on. Leave "
             "it off for capacity probes, which run past the ceiling on "
             "purpose and where an OOM is the result being measured.",
    )
    args = parser.parse_args()

    if args.churn and args.churn_num_tenants is None:
        parser.error("--churn requires --churn-num-tenants")
    if args.churn_num_tenants is not None and not args.churn:
        parser.error("--churn-num-tenants requires --churn")

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Benchmark requires a GPU.")
        return

    dtype = DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print_memory_ceiling(args.model, dtype, vram_gb, args.target_modules)

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
    oomed: list[tuple] = []
    total = len(configs) * len(seeds)
    done = 0
    # The CSV is written incrementally, one row per completed config, flushed
    # immediately. A long sweep (the largest here is ~6 h) that dies partway --
    # OOM at a high N, pod eviction, exhausted balance -- then still leaves
    # every config it did finish on disk, instead of losing the whole run.
    csv_file = None
    writer = None

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
                    engine=args.engine,
                    target_modules=args.target_modules,
                    churn_num_tenants=args.churn_num_tenants if args.churn else None,
                    churn_zipf_alpha=args.churn_zipf_alpha,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM at adapters={num_adapters} batch={batch_size} rank={lora_rank}: {e}")
                torch.cuda.empty_cache()
                row = oom_row(
                    model_name=args.model,
                    dtype=dtype,
                    num_adapters=num_adapters,
                    batch_size=batch_size,
                    lora_rank=lora_rank,
                    seq_len=args.seq_len,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=seed,
                    engine=args.engine,
                    target_modules=args.target_modules,
                    churn_num_tenants=args.churn_num_tenants if args.churn else None,
                    churn_zipf_alpha=args.churn_zipf_alpha,
                )
                oomed.append((num_adapters, batch_size, lora_rank, seed))
            else:
                results.append(row)
            if writer is None:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                # In append mode the header goes in only when starting a fresh
                # file, so a second arm concatenates cleanly. Checked before the
                # open() because "a" creates the file on the spot.
                out_path = Path(args.out)
                write_header = not (
                    args.append and out_path.exists() and out_path.stat().st_size > 0
                )
                csv_file = open(args.out, "a" if args.append else "w", newline="")
                writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            os.fsync(csv_file.fileno())
            if row["status"] == "ok":
                print(f"  p50={row['p50_ms']}ms  p95={row['p95_ms']}ms  p99={row['p99_ms']}ms  "
                      f"asm={row['assemble_mean_ms']}ms ({row['assemble_share_pct']}%)  "
                      f"fwd={row['forward_mean_ms']}ms  "
                      f"tput={row['throughput_samples_sec']} samples/s  "
                      f"peak_gpu={row['peak_gpu_mem_gb']}GB")
                if row["churn_enabled"]:
                    print(f"  churn e2e: p50={row['churn_end_to_end_p50_ms']}ms  "
                          f"p95={row['churn_end_to_end_p95_ms']}ms  "
                          f"p99={row['churn_end_to_end_p99_ms']}ms  "
                          f"mean={row['churn_end_to_end_mean_ms']}ms  "
                          f"tput={row['churn_end_to_end_throughput_samples_sec']} samples/s")
                    print(f"  churn: misses={row['churn_request_misses']}  "
                          f"register=p50 {row['churn_registration_p50_ms']}ms, "
                          f"mean {row['churn_registration_mean_ms']}ms, "
                          f"total {row['churn_registration_total_ms']}ms  "
                          f"evict=p50 {row['churn_eviction_p50_ms']}ms, "
                          f"mean {row['churn_eviction_mean_ms']}ms, "
                          f"total {row['churn_eviction_total_ms']}ms  "
                          f"store_total={row['churn_adapter_store_total_ms']}ms  "
                          f"store_share_overall={row['churn_adapter_store_overall_share_pct']}%\n")
                    print(f"  per-batch store-share: p50={row['churn_adapter_store_share_p50_pct']}%  "
                          f"p95={row['churn_adapter_store_share_p95_pct']}%  "
                          f"p99={row['churn_adapter_store_share_p99_pct']}%  "
                          f"mean={row['churn_adapter_store_share_mean_pct']}%\n")
                else:
                    print()
            else:
                print("  recorded as status=oom\n")

    if csv_file is not None:
        csv_file.close()

    if oomed:
        print(f"\n{len(oomed)} of {total} config(s) OOM'd (recorded with status=oom):")
        for n, b, r, s in oomed:
            print(f"    adapters={n} batch={b} rank={r} seed={s}")

    if not results:
        print("No results collected (all configs OOM'd?)")
        if args.require_complete:
            raise SystemExit("FAILED --require-complete: every config OOM'd.")
        return

    print("\n=== Results ===")
    print_table(results)
    print(f"\nSaved to {args.out}  ({len(results)}/{total} configs completed)")

    # A sweep's grid is chosen to fit; a missing cell means something went wrong
    # (fragmentation, eviction, a ceiling estimate that was too optimistic), so
    # --require-complete turns that into a non-zero exit. Capacity probes leave
    # the flag off: they deliberately run past the ceiling, and an OOM there is
    # the measurement, not a failure.
    if args.require_complete and len(results) != total:
        raise SystemExit(
            f"FAILED --require-complete: {len(results)}/{total} configs "
            f"completed, {len(oomed)} OOM'd. The CSV is missing cells -- see the "
            "status=oom rows above. Do not patch the gap with a second run; fix "
            "the cause (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "recovers allocator fragmentation) and re-run the sweep."
        )


if __name__ == "__main__":
    main()
