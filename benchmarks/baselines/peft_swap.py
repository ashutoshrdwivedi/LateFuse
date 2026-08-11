"""PEFT per-request swap baseline.

Measures latency / throughput for a mixed-tenant batch served via HuggingFace PEFT.
This is the worst-case-but-realistic baseline that our multi-tenant impl is compared against.

Five modes:
  - sequential:  each sample uses set_adapter() + forward(B=1). Worst case (no batching).
  - grouped:     bucket a mixed (uniformly sampled) batch by adapter_id, one batched
                 forward per bucket. With N >> B a batch is nearly all-distinct, so this
                 degrades to per-sample swapping.
  - homogeneous: every sample in the batch shares one (randomly drawn) adapter, so PEFT
                 pays exactly one set_adapter() + one batched forward per batch. This is
                 PEFT's friendliest batching (single-tenant batches); the swap cost is
                 O(N) in attached adapters, so it still bites at large N.
  - mixed:       PEFT's native mixed-batch API: one forward over the whole mixed batch
                 with adapter_names= listing each sample's adapter. No set_adapter();
                 PEFT shares the base path and loops over the distinct adapters inside
                 every LoRA layer, and validates adapter_names against all registered
                 adapters on every call (O(N*layers) scan, uncached as of peft 0.19.1).
                 Cannot route per-tenant heads (huggingface/peft#1960).
  - base:        no adapters; one batched forward of the bare base model. Single-tenant
                 throughput ceiling (N-independent), the upper bound any multi-tenant
                 serving system can aspire to. Run once per batch size (use --adapters 1).

Output CSV is schema-compatible with `lora_serving.benchmark.run` so the two can be
merged for plotting. Extra columns: `harness=peft_<mode>`.

Run :
    uv run python -m benchmarks.baselines.peft_swap \\
        --adapters 100 1000 5000 --batch-sizes 8 32 128 --mode sequential
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def build_base_model(model_name: str, dtype: torch.dtype, device: torch.device):
    """Load the bare base model (no PEFT, no adapters) for the single-tenant ceiling."""
    base = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    base.to(device=device, dtype=dtype)
    base.eval()
    return base


def build_peft_model(
    model_name: str,
    lora_rank: int,
    dtype: torch.dtype,
    device: torch.device,
    target_modules: list[str] | None = None,
):
    """Load base + wrap with PEFT, returning model with adapter named 'adapter_0'."""
    base = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=target_modules or ["query", "value"],
        lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(base, cfg, adapter_name="adapter_0")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


def add_n_adapters(model, lora_cfg: LoraConfig, n: int, device: torch.device, dtype: torch.dtype):
    """Add adapter_1..adapter_{n-1} to the model (adapter_0 already exists)."""
    for i in range(1, n):
        model.add_adapter(f"adapter_{i}", lora_cfg)
    model.to(device=device, dtype=dtype)


def sample_batch(adapter_ids: list[str], batch_size: int, mode: str) -> list[str]:
    """Per-iteration adapter assignment for a batch.

    homogeneous -> one random adapter replicated across the batch (single-tenant batch).
    everything else -> uniform draw with replacement (mixed-tenant batch).
    """
    if mode == "homogeneous":
        return [random.choice(adapter_ids)] * batch_size
    return random.choices(adapter_ids, k=batch_size)


def make_inputs(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(1, 30000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return input_ids, attention_mask


def run_sequential(model, adapter_ids: list[str], inputs: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    """One forward per sample with set_adapter() between each. Returns total ms (event-timed)."""
    input_ids, attention_mask = inputs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i, aid in enumerate(adapter_ids):
            model.set_adapter(aid)
            model(input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def run_base(model, adapter_ids: list[str], inputs: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    """One batched forward of the bare base model (adapter_ids ignored). Returns total ms."""
    input_ids, attention_mask = inputs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def run_mixed(model, adapter_ids: list[str], inputs: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    """One forward over the whole mixed batch via PEFT's adapter_names API. Returns total ms."""
    input_ids, attention_mask = inputs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask, adapter_names=adapter_ids)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def run_grouped(model, adapter_ids: list[str], inputs: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    """Bucket samples by adapter_id, one batched forward per bucket. Returns total ms."""
    input_ids, attention_mask = inputs
    buckets: dict[str, list[int]] = {}
    for i, aid in enumerate(adapter_ids):
        buckets.setdefault(aid, []).append(i)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for aid, idxs in buckets.items():
            model.set_adapter(aid)
            idx_t = torch.tensor(idxs, device=device)
            model(input_ids[idx_t], attention_mask=attention_mask[idx_t])
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def build_for_config(
    model_name: str,
    num_adapters: int,
    lora_rank: int,
    dtype: torch.dtype,
    device: torch.device,
    mode: str,
    target_modules: list[str] | None = None,
):
    """Build the model + adapters once. Returns (model, adapter_ids, runner, add_time_s).

    For grouped/sequential this pays the O(N) PEFT add_adapter() registration cost; callers
    reuse the returned model across all batch sizes so it is paid once per (N, rank), not per batch.
    """
    if mode == "base":
        print(f"  Building bare base model (no adapters)...", flush=True)
        model = build_base_model(model_name, dtype, device)
        return model, ["adapter_0"], run_base, 0.0

    print(f"  Building PEFT model + adding {num_adapters} adapters...", flush=True)
    model, lora_cfg = build_peft_model(model_name, lora_rank, dtype, device, target_modules)
    t0 = time.perf_counter()
    add_n_adapters(model, lora_cfg, num_adapters, device, dtype)
    add_time_s = time.perf_counter() - t0
    print(f"  add_adapter() x{num_adapters - 1} took {add_time_s:.1f}s", flush=True)
    adapter_ids = [f"adapter_{i}" for i in range(num_adapters)]
    # homogeneous reuses run_grouped: a single-adapter batch forms one bucket, so it
    # naturally becomes one set_adapter() + one batched forward.
    runners = {"sequential": run_sequential, "mixed": run_mixed}
    runner = runners.get(mode, run_grouped)
    return model, adapter_ids, runner, add_time_s


def measure_config(
    model,
    model_name: str,
    runner,
    adapter_ids: list[str],
    num_adapters: int,
    batch_size: int,
    lora_rank: int,
    seq_len: int,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    mode: str,
    add_time_s: float,
    device: torch.device,
) -> dict:
    """Warm up + measure one (already-built) model at a given batch size."""
    inputs = make_inputs(batch_size, seq_len, device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    print(f"  Warming up ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        batch_ids = sample_batch(adapter_ids, batch_size, mode)
        runner(model, batch_ids, inputs, device)

    print(f"  Measuring ({iters} iters)...", flush=True)
    latencies = np.empty(iters)
    for i in range(iters):
        batch_ids = sample_batch(adapter_ids, batch_size, mode)
        latencies[i] = runner(model, batch_ids, inputs, device)

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    throughput = batch_size / (np.mean(latencies) / 1000)

    return {
        "harness": f"peft_{mode}",
        "model": model_name,
        "dtype": str(dtype).replace("torch.", ""),
        "num_adapters": num_adapters,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
        "throughput_samples_sec": round(throughput, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "add_adapter_total_s": round(add_time_s, 2),
        "warmup": warmup,
        "iters": iters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="fp16")
    parser.add_argument("--adapters", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[8])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=["sequential", "grouped", "homogeneous", "mixed", "base"], default="sequential")
    parser.add_argument(
        "--target-modules", nargs="+", default=["query", "value"],
        help="Projection module names for LoRA — model-specific: BERT/RoBERTa "
             "family uses 'query value'; DeBERTa-v2 uses 'query_proj value_proj'.",
    )
    parser.add_argument("--out", default="peft_baseline.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA. Aborting.")
        return

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Mode: {args.mode}\n", flush=True)

    total = len(args.adapters) * len(args.batch_sizes) * len(args.lora_ranks)
    done = 0
    results = []
    # Build the model ONCE per (num_adapters, lora_rank) and reuse across all batch sizes.
    # PEFT add_adapter() is O(N) single-threaded registration; building per-batch wasted ~22 min/config.
    for num_adapters in args.adapters:
        for lora_rank in args.lora_ranks:
            try:
                model, adapter_ids, runner, add_time_s = build_for_config(
                    args.model, num_adapters, lora_rank, dtype, device, args.mode,
                    args.target_modules,
                )
            except Exception as e:
                print(f"  BUILD ERROR ({type(e).__name__}) at adapters={num_adapters} rank={lora_rank}: {e}", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue

            for batch_size in args.batch_sizes:
                done += 1
                print(f"[{done}/{total}] adapters={num_adapters} batch={batch_size} rank={lora_rank}", flush=True)
                try:
                    row = measure_config(
                        model=model,
                        model_name=args.model,
                        runner=runner,
                        adapter_ids=adapter_ids,
                        num_adapters=num_adapters,
                        batch_size=batch_size,
                        lora_rank=lora_rank,
                        seq_len=args.seq_len,
                        warmup=args.warmup,
                        iters=args.iters,
                        dtype=dtype,
                        mode=args.mode,
                        add_time_s=add_time_s,
                        device=device,
                    )
                    results.append(row)
                    print(f"  p50={row['p50_ms']}ms  p99={row['p99_ms']}ms  "
                          f"tput={row['throughput_samples_sec']} samples/s  "
                          f"peak={row['peak_gpu_mem_gb']}GB\n", flush=True)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  OOM: {e}", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"  ERROR ({type(e).__name__}): {e}", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

            # Free the model before building the next (num_adapters, rank) variant.
            del model
            torch.cuda.empty_cache()
            gc.collect()

    if not results:
        print("No results collected.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
