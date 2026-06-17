"""Empirically establish that the LoRA path is a small fraction of forward time.

Produces evidence for the paper claim "custom kernels (Punica MBGMV / S-LoRA SGMV)
would not move the needle for this serving setting" by:

  1. PROFILER DUMP — PyTorch profiler over N representative forward iterations,
     sorted by SELF CUDA time (children excluded, so totals are not
     double-counted across the parent/child op hierarchy).

     NOTE: aten::bmm is emitted by the LoRA shrink/expand, by attention
     (matmul over 4-D tensors dispatches to bmm), and by the LR head, so the
     raw bmm row is NOT LoRA-only. We isolate the LoRA contribution by diffing
     two profiler runs (apply_lora=True vs False) op-by-op.

  2. NO-LORA ABLATION — identical model and inputs, but the forward is run with
     apply_lora=False, which skips the entire LoRA delta path (cat/shrink/
     expand/add) rather than zeroing operands. The wall-clock difference is the
     true cost of the LoRA path. (Zeroing operands measures nothing: bmm cost is
     independent of operand values, and synthetic adapters already have B=0.)

  3. ANALYTIC FLOP RATIO — printed for reference: base encoder FLOPs vs
     LoRA shrink/expand FLOPs per layer per batch.

Run on the e2e/A100 node:
    uv run python -m benchmarks.profiling.forward_breakdown \\
        --model BAAI/bge-m3 --batch-size 32 --num-adapters 1000 \\
        --lora-rank 8 --warmup 20 --profiler-iters 10 --ablation-iters 100 \\
        --out benchmarks/results/forward_breakdown.txt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

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

# Op-name buckets for categorizing profiler rows
BUCKETS = {
    "base_linear": ("aten::addmm", "aten::linear", "aten::matmul"),
    "lora_bmm": ("aten::bmm", "aten::baddbmm"),
    "attention": ("aten::softmax", "aten::_softmax", "aten::scaled_dot_product_attention", "aten::masked_fill"),
    "layernorm_residual": ("aten::layer_norm", "aten::add", "aten::add_"),
    "elementwise": ("aten::gelu", "aten::relu", "aten::tanh", "aten::mul", "aten::div"),
}


def categorize(op_name: str) -> str:
    for cat, prefixes in BUCKETS.items():
        if any(op_name.startswith(p) for p in prefixes):
            return cat
    return "other"


def setup(args) -> tuple:
    device = torch.device("cuda:0")
    dtype = DTYPE_MAP[args.dtype]
    config = LoraServingConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )
    print(f"Loading {args.model} ({dtype})...")
    model = EncoderWithLora.from_pretrained_serving(config)
    model.eval()

    print(f"Generating {args.num_adapters} synthetic adapters (rank={args.lora_rank})...")
    store = AdapterStore(config)
    make_synthetic_adapters(store, args.num_adapters)
    adapter_ids = [f"adapter_{i}" for i in range(args.num_adapters)]

    lr_coefs, lr_intercepts = {}, {}
    for aid in adapter_ids:
        c, i = make_synthetic_lr_weights(config, args.num_labels)
        lr_coefs[aid] = c
        lr_intercepts[aid] = i

    assembler = BatchAssembler(store, config)
    inputs = make_synthetic_inputs(config, args.batch_size)
    output_lr = torch.zeros(args.batch_size, 1, args.num_labels, dtype=dtype, device=device)

    return model, store, assembler, adapter_ids, lr_coefs, lr_intercepts, inputs, output_lr, config, device


def analytic_flop_ratio(config: LoraServingConfig, batch_size: int, seq_len: int) -> dict:
    """Per-layer FLOP counts: base encoder vs LoRA shrink/expand."""
    B, S, H, r, L = batch_size, seq_len, config.hidden_size, config.lora_rank, config.num_layers
    I = config.intermediate_size
    M = len(config.target_modules)
    # Base attention (Q, K, V, O linears): 4 * B*S*H^2
    # Base FFN (2 linears): 2 * B*S*H*I
    # The QK^T and PV attention matmuls are omitted, which understates base
    # FLOPs — so base_to_lora_ratio is a lower bound (conservative for the
    # "LoRA is a small fraction" claim).
    base_per_layer = 4 * B * S * H * H + 2 * B * S * H * I
    # LoRA shrink: M * B*S*H*r   ; expand: M * B*S*r*H
    lora_per_layer = M * 2 * B * S * H * r
    return {
        "base_GFLOPs_per_layer": base_per_layer / 1e9,
        "lora_GFLOPs_per_layer": lora_per_layer / 1e9,
        "base_to_lora_ratio": base_per_layer / lora_per_layer,
        "base_total_GFLOPs": base_per_layer * L / 1e9,
        "lora_total_GFLOPs": lora_per_layer * L / 1e9,
    }


def _self_cuda_us(evt) -> float:
    """Self (not total) CUDA microseconds for a profiler event.

    Self-time excludes child ops, so summing it across key_averages() gives the
    true total device time with no double-counting. torch 2.11 renamed
    self_cuda_time_total -> self_device_time_total.
    """
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def run_profiler(model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
                 inputs, output_lr, config, device, batch_size, warmup, iters,
                 apply_lora=True):
    """Profile `iters` forward passes, aggregate by SELF CUDA time.

    Returns (table, summary, total_self_ms, self_us_by_op) where self_us_by_op
    maps op key -> self CUDA microseconds (used to isolate LoRA-only bmm by
    diffing the apply_lora=True and apply_lora=False runs)."""
    def one_batch():
        ids = random.choices(adapter_ids, k=batch_size)
        coefs = [lr_coefs[a] for a in ids]
        intercepts = [lr_intercepts[a] for a in ids]
        lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
        with torch.no_grad():
            model(inputs["input_ids"], inputs["attention_mask"], lora_w, lr_w,
                  output_lr, apply_lora=apply_lora)

    print(f"Warmup ({warmup} iters, apply_lora={apply_lora})...")
    for _ in range(warmup):
        one_batch()
    torch.cuda.synchronize(device)

    print(f"Profiling ({iters} iters, apply_lora={apply_lora})...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            with record_function("forward_iter"):
                one_batch()
        torch.cuda.synchronize(device)

    sort_key = "self_device_time_total" if hasattr(prof.key_averages()[0], "self_device_time_total") else "self_cuda_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=30)

    # Aggregate self-time by bucket and by op key.
    #
    # IMPORTANT: only count `aten::` operator rows. key_averages() also contains
    # the underlying CUDA kernel rows (e.g. ampere_*_gemm) AND user annotations
    # (forward_iter), each carrying their own self-device time. Summing all rows
    # double-counts (the aten op and its launched kernel are the same GPU work)
    # and inflates the total ~3x. The aten:: rows' self-times alone sum to the
    # profiler's reported "Self CUDA time total", so they are the correct basis.
    totals = defaultdict(lambda: {"cuda_us": 0.0, "calls": 0})
    self_us_by_op = defaultdict(float)
    grand_cuda_us = 0.0
    for evt in prof.key_averages():
        if not str(evt.key).startswith("aten::"):
            continue
        cuda_us = _self_cuda_us(evt)  # microseconds, self only
        if cuda_us <= 0:
            continue
        self_us_by_op[evt.key] += cuda_us
        bucket = categorize(evt.key)
        totals[bucket]["cuda_us"] += cuda_us
        totals[bucket]["calls"] += int(evt.count)
        grand_cuda_us += cuda_us

    summary = {}
    for bucket, vals in totals.items():
        summary[bucket] = {
            "cuda_ms_total": vals["cuda_us"] / 1000.0,
            "share_pct": 100.0 * vals["cuda_us"] / grand_cuda_us if grand_cuda_us else 0.0,
            "calls": vals["calls"],
        }
    return table, summary, grand_cuda_us / 1000.0, dict(self_us_by_op)


def run_ablation(model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
                 inputs, output_lr, device, batch_size, warmup, iters) -> dict:
    """Compare forward time with the LoRA path on vs off.

    `apply_lora=False` skips the entire LoRA delta path (per-layer cat of A/B,
    the shrink/expand bmms, and the additive merge) while keeping every base
    op identical. The wall-clock difference is the true cost of the LoRA path.

    This replaces an earlier "zero the operands" ablation, which measured
    nothing: bmm cost is independent of operand values, and the synthetic
    adapters already have B=0, so the LoRA delta was already zero in both arms.
    The same per-batch assembled weights are reused for both arms so the only
    difference is whether the delta path executes; assembly is done outside the
    timed region.
    """
    def measure(apply_lora: bool, n: int) -> np.ndarray:
        times = np.empty(n)
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        for i in range(n):
            ids = random.choices(adapter_ids, k=batch_size)
            coefs = [lr_coefs[a] for a in ids]
            intercepts = [lr_intercepts[a] for a in ids]
            lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
            ev_start.record()
            with torch.no_grad():
                model(inputs["input_ids"], inputs["attention_mask"], lora_w, lr_w,
                      output_lr, apply_lora=apply_lora)
            ev_end.record()
            torch.cuda.synchronize(device)
            times[i] = ev_start.elapsed_time(ev_end)
        return times

    print(f"Ablation warmup ({warmup} iters, LoRA on)...")
    measure(apply_lora=True, n=warmup)
    print(f"Ablation: measuring LoRA on ({iters} iters)...")
    normal = measure(apply_lora=True, n=iters)
    print(f"Ablation: measuring LoRA off ({iters} iters)...")
    zero = measure(apply_lora=False, n=iters)

    return {
        "lora_on_mean_ms": float(np.mean(normal)),
        "lora_on_p50_ms": float(np.percentile(normal, 50)),
        "lora_off_mean_ms": float(np.mean(zero)),
        "lora_off_p50_ms": float(np.percentile(zero, 50)),
        "lora_cost_mean_ms": float(np.mean(normal) - np.mean(zero)),
        "lora_cost_share_pct": 100.0 * (np.mean(normal) - np.mean(zero)) / np.mean(normal),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-adapters", type=int, default=1000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profiler-iters", type=int, default=10)
    parser.add_argument("--ablation-iters", type=int, default=100)
    parser.add_argument("--out", default="benchmarks/results/forward_breakdown.txt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Need CUDA. Aborting.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    setup_tuple = setup(args)
    model, store, assembler, adapter_ids, lr_coefs, lr_intercepts, inputs, output_lr, config, device = setup_tuple

    flops = analytic_flop_ratio(config, args.batch_size, args.seq_len)

    table, profiler_summary, total_forward_ms, self_us_on = run_profiler(
        model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
        inputs, output_lr, config, device,
        batch_size=args.batch_size, warmup=args.warmup, iters=args.profiler_iters,
        apply_lora=True,
    )

    # Second profiler pass with the LoRA path disabled, so we can isolate the
    # LoRA-only bmm self-time (the raw bmm row also contains attention + LR-head
    # bmms) by diffing op-level self-time between the two runs.
    _, _, total_forward_off_ms, self_us_off = run_profiler(
        model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
        inputs, output_lr, config, device,
        batch_size=args.batch_size, warmup=args.warmup, iters=args.profiler_iters,
        apply_lora=False,
    )

    bmm_self_on_ms = self_us_on.get("aten::bmm", 0.0) / 1000.0
    bmm_self_off_ms = self_us_off.get("aten::bmm", 0.0) / 1000.0
    lora_bmm_self_ms = bmm_self_on_ms - bmm_self_off_ms
    lora_bmm_self_share = 100.0 * lora_bmm_self_ms / total_forward_ms if total_forward_ms else 0.0
    bmm_isolation = {
        "bmm_self_ms_lora_on": bmm_self_on_ms,
        "bmm_self_ms_lora_off": bmm_self_off_ms,
        "lora_bmm_self_ms": lora_bmm_self_ms,
        "lora_bmm_self_share_pct": lora_bmm_self_share,
        "total_self_ms_lora_on": total_forward_ms,
        "total_self_ms_lora_off": total_forward_off_ms,
    }

    ablation = run_ablation(
        model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
        inputs, output_lr, device,
        batch_size=args.batch_size, warmup=args.warmup, iters=args.ablation_iters,
    )

    # ---- write report ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"# Forward-pass breakdown — {args.model}\n")
        f.write(f"# GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"# Config: B={args.batch_size} N={args.num_adapters} r={args.lora_rank} "
                f"seq={args.seq_len} dtype={args.dtype}\n\n")

        f.write("## 1. Analytic FLOP ratio (per encoder layer)\n")
        for k, v in flops.items():
            f.write(f"  {k:30s}: {v:,.4f}\n")
        f.write("\n")
        f.write("Interpretation: even an infinitely fast LoRA bmm kernel would\n"
                "save at most 1 / (base/lora ratio) of total forward FLOPs.\n\n")

        f.write("## 2. PyTorch profiler — op-level SELF CUDA time breakdown\n\n")
        f.write(table)
        f.write("\n\n## 3. Aggregated by category (SELF CUDA time, no double-counting)\n")
        f.write(f"  Total forward self CUDA time (sum of op self-times): {total_forward_ms:.2f} ms\n\n")
        for cat in sorted(profiler_summary, key=lambda c: -profiler_summary[c]["cuda_ms_total"]):
            d = profiler_summary[cat]
            f.write(f"  {cat:22s}: {d['cuda_ms_total']:8.3f} ms  "
                    f"({d['share_pct']:5.1f}%)  calls={d['calls']}\n")
        f.write("\n")
        f.write("NOTE: the 'lora_bmm' category counts ALL aten::bmm, which includes\n"
                "attention (matmul over 4-D tensors) and the LR head, not just LoRA.\n"
                "See section 3b for the LoRA-only bmm isolation.\n\n")

        f.write("## 3b. LoRA-only bmm isolation (apply_lora on vs off)\n")
        for k, v in bmm_isolation.items():
            unit = "%" if k.endswith("_pct") else " ms"
            f.write(f"  {k:30s}: {v:8.3f}{unit}\n")
        f.write("\n")
        f.write("Interpretation: lora_bmm_self_share_pct is the LoRA shrink/expand\n"
                "share of total forward self CUDA time, with attention/head bmm removed.\n\n")

        f.write("## 4. No-LoRA ablation (wall-clock, apply_lora on vs off)\n")
        for k, v in ablation.items():
            unit = "%" if k.endswith("_pct") else " ms"
            f.write(f"  {k:30s}: {v:8.3f}{unit}\n")
        f.write("\n")
        f.write("Interpretation: lora_cost_share_pct is the wall-clock contribution\n"
                "of the entire LoRA path (cat/shrink/expand/add). If small, a faster\n"
                "custom bmm kernel (Punica MBGMV / S-LoRA SGMV) cannot move the needle.\n")

    # Also write a JSON sidecar for downstream tooling
    json_path = out.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "config": vars(args),
            "flop_ratio": flops,
            "profiler_categories": profiler_summary,
            "profiler_total_ms": total_forward_ms,
            "bmm_isolation": bmm_isolation,
            "ablation": ablation,
        }, f, indent=2)

    print(f"\nWrote {out} and {json_path}")
    print(f"\n=== Headline numbers ===")
    print(f"  LoRA-only bmm self share (profiler diff):  "
          f"{bmm_isolation['lora_bmm_self_share_pct']:.2f}%")
    print(f"  LoRA path wall-clock share (ablation):     "
          f"{ablation['lora_cost_share_pct']:.2f}%")
    print(f"  Analytic FLOP ratio (base/lora):           "
          f"{flops['base_to_lora_ratio']:.0f}x")


if __name__ == "__main__":
    main()
