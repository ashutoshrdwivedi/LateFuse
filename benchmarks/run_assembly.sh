#!/bin/bash
# A100-80GB (short) -- result-scatter instrumentation.
#
# Quantifies the CPU-path cost: the per-tenant slice + softmax, to show it is
# negligible next to batch assembly. Must run on an A100-80GB so the numbers
# compose with the already-measured assembly_bench results rather than mixing
# hardware.
#
# Scatter is measured on-device only, matching the forward's boundary; host D2H
# and serialization are serving-layer and stay out of scope. State that caveat
# explicitly wherever the scatter numbers are cited.
#
# MUST run on an A100-80GB. Two reasons: the numbers have to compose with the
# already-quoted results rather than mixing hardware, and the baseline arm is
# CPU-bound by construction -- its ~4.7k req/s ceiling is a property of the host
# CPU, not just the GPU. The bench now records CPU model and usable core count
# in both the .txt header and the .json, so a rerun on a different pod shape is
# detectable instead of silent. Check the CPU line matches before quoting any
# number alongside the originals.
#
# New output directory: does NOT clobber the submitted assembly_bench.txt /
# assembly_bench_minilm.txt results. Compare, then promote deliberately.
#
# Runtime ~25 min. Independent of the other generality scripts -- pairs well
# appended to run_electra.sh on the same pod.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk (see
# run_deberta.sh: a default 20G container disk fills mid-sweep).
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set here, unlike every other
# sweep script. Two reasons, and they both point the same way.
#   1. No exposure. This bench runs at N=2000 (~3 GB of adapters on an 80 GB
#      card), nowhere near the ceiling, so the setting would buy no capacity.
#   2. It would confound the one comparison this script exists to make. Per the
#      header, these results get diffed against the SUBMITTED
#      benchmarks/results/assembly_bench{,_minilm}.txt and then promoted
#      deliberately -- and those were measured under the default allocator.
#      The baseline arm allocates a fresh assembled tensor every iteration,
#      which is precisely the quantity being timed against index_select, so
#      swapping the allocator changes the measured thing on one side of a
#      before/after diff.
# If you ever do set it, re-measure BOTH arms and do not quote the new speedups
# next to the submitted ones.
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/LateFuse
R=benchmarks/results/assembly
mkdir -p "$R"

echo "=== [1/2] assembly + scatter, bge-m3 (paper's model) ==="
uv run python -m benchmarks.profiling.assembly_bench \
  --model "BAAI/bge-m3" --dtype fp16 --num-adapters 2000 \
  --batch-sizes 8 16 32 64 128 --lora-rank 8 --seq-len 128 \
  --warmup 50 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/assembly_bench.txt" > "$R/assembly_bench.log" 2>&1
echo "  bge-m3 rc=$?"

# MiniLM is the >10k req/s CPU-regime demo: a fast, shallow model makes the
# CPU assembly share (and hence the scatter share) maximally visible. Keep the
# 256 cell -- it is where the baseline throughput ceiling and the tail spike are
# measured most tightly.
echo "=== [2/2] assembly + scatter, all-MiniLM-L6-v2 (CPU regime) ==="
uv run python -m benchmarks.profiling.assembly_bench \
  --model "sentence-transformers/all-MiniLM-L6-v2" --dtype fp16 --num-adapters 2000 \
  --batch-sizes 8 16 32 64 128 256 --lora-rank 8 --seq-len 128 \
  --warmup 50 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/assembly_bench_minilm.txt" > "$R/assembly_bench_minilm.log" 2>&1
echo "  minilm rc=$?"

echo "ALL DONE (assembly, house protocol 50/200 x 5 seeds)"
head -8 "$R/assembly_bench.txt" "$R/assembly_bench_minilm.txt"
