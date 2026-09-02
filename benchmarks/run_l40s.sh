#!/bin/bash
# L40S-48GB -- hardware-generality row.
#
# Runs the paper's reference model (bge-m3) on an inference-class GPU -- Ada,
# GDDR6, ~864 GB/s vs the A100's HBM2e ~2 TB/s -- to show the O(1)-in-N property
# and the PEFT speedups are not artifacts of the A100.
#
# SCOPE. This row varies the GPU *class* and nothing else, so it speaks to
# hardware generality: the O(1)-in-N property survives a 2.3x memory-bandwidth
# swing on the same code and the same measurement protocol. It says nothing
# about multi-GPU or multi-node scaling, which remains future work -- keep the
# two claims distinct when citing these numbers.
#
# Side benefit: the memory model is validated independently of the A100 here.
# Measured 1.594 MB/adapter against 1.573 theoretical (fp16, r=8), with 1.86 GB
# of non-adapter overhead at the 45.9 GB peak.
#
# Measured with BatchAssembler -- the SAME path run_sxm80.sh uses, so the
# cross-hardware comparison is internally consistent. Note this is NOT
# IndexSelectBatchAssembler (weights/batch.py), which benchmarks separately via
# benchmarks/profiling/assembly_bench.py; numbers from the two paths are not
# interchangeable and should not be quoted side by side.
#
# Ceiling scales with VRAM exactly as Finding 4 predicts: 1.57 MB/adapter on
# ~45 GB usable -> ~28k adapters, vs 47k on the 80GB A100. Measured 2026-07-27:
# 28000 fits (45.9 GB peak), 29000 OOMs -- prediction confirmed.
#
# expandable_segments is REQUIRED, not a tuning knob. Without it the default
# caching allocator fragments during adapter generation and every N=28000 cell
# OOMs at ~44.5/44.5 GiB with 2.7 GiB reserved-but-unallocated -- silently
# dropping the entire top-N column (140 rows instead of 165) while the sweep
# still exits rc=0. With it, the same cells run at 96% VRAM utilization.
#
# Grid and measurement kernel both mirror run_sxm80.sh: the full N x B cross
# product plus the same three rank cells at the (1000, 32) operating point.
# Since this row varies ONLY the GPU, a full grid makes it directly diffable
# against Table 2 cell for cell -- the cleanest form of a hardware claim.
# 6 N x 5 B + 3 rank cells = 33 configs x 5 seeds = 165 runs.
# The PEFT arm runs on this same node so the speedup column is never a
# cross-node comparison.
#
# Runtime ~1h30m measured 2026-07-27 (the ~2h45m estimate was conservative);
# the sweep is ~1h of it. Independent of the other generality scripts.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk (see
# run_deberta.sh: a default 20G container disk fills mid-sweep).
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
# Required for the N=28000 cells -- see the fragmentation note in the header.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/LateFuse
R=benchmarks/results/l40s
mkdir -p "$R"
SWEEP_INCOMPLETE=0

M="BAAI/bge-m3"
TAG=l40s

echo "=== [0/5] smoke test (fail fast before burning the pod) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  --model "BAAI/bge-m3:query,value" --out-dir "$R" \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

echo "=== [2/5] LateFuse sweep, full grid, 5 seeds (~2h) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --dtype fp16 \
  --adapters 100 1000 5000 10000 20000 28000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 \
  --extra-configs 1000:32:4 1000:32:16 1000:32:32 \
  --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 --require-complete \
  --out "$R/sweep_bgem3_$TAG.csv" > "$R/sweep_bgem3_$TAG.log" 2>&1
rc=$?; echo "  sweep rc=$rc"
if [ $rc -ne 0 ]; then
  SWEEP_INCOMPLETE=1
  echo "  !! SWEEP INCOMPLETE -- cells are missing from the CSV (status=oom rows"
  echo "     name them). Continuing so the remaining arms still land, but this"
  echo "     script will exit non-zero at the end. See $R/sweep_*.log."
fi

# Capacity is probed TWICE, once per allocator, because the 26k-vs-28k gap is
# itself a reported result -- a free capacity gain not previously quantified
# (+7.7%) -- and not merely a run
# artifact. The archived split (sweep_capacity_l40s.csv at 26000 under the
# default allocator, sweep_capacity_l40s_expseg.csv at 28000 under
# expandable_segments) exists only because the export landed between two
# hand-run probes; neither file matches the grid this script asked for.
#
# That matters for a re-run: the export at the top is now unconditional, so a
# single-arm probe would write expandable_segments numbers into the file whose
# committed contents are the DEFAULT-allocator measurement, destroying the 26k
# comparison point -- which exists nowhere else on disk. Running both arms
# explicitly makes the A/B reproducible instead of an accident of provenance,
# and both land in ONE CSV keyed by 'alloc_conf', so the script is the single
# source of truth for this row and no hand-run patch file is needed.
# Both arms write to ONE capacity CSV, told apart by the 'alloc_conf' column.
# rm -f first so a rerun replaces the file rather than appending to the last
# run's rows -- --append is for the second arm, not for accumulating history.
rm -f "$R/sweep_capacity_$TAG.csv"

echo "=== [3a/5] capacity probe -- expandable_segments (the shipped setting) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --dtype fp16 \
  --adapters 28000 29000 30000 --batch-sizes 32 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_capacity_$TAG.csv" > "$R/sweep_capacity_$TAG.log" 2>&1
echo "  capacity (expseg) rc=$?"

# The one arm that must NOT inherit the export above. Unset inside a subshell
# so the variable is absent from the environment `uv run` hands to the child,
# and the export stays in effect for everything after this block.
echo "=== [3b/5] capacity probe -- DEFAULT allocator (the 26k comparison point) ==="
( unset PYTORCH_CUDA_ALLOC_CONF
  uv run python -m latefuse.benchmark.run \
    --model "$M" --dtype fp16 \
    --adapters 26000 27000 28000 --batch-sizes 32 --lora-ranks 8 \
    --seq-len 128 --warmup 50 --iters 200 \
    --append --out "$R/sweep_capacity_$TAG.csv" \
) >> "$R/sweep_capacity_$TAG.log" 2>&1
echo "  capacity (default) rc=$?"

echo "=== [4/5] PEFT mixed baseline, same node (~40 min) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 100 1000 --mode mixed \
  --warmup 10 --iters 50 \
  --out "$R/peft_mixed_$TAG.csv" > "$R/peft_mixed_$TAG.log" 2>&1
echo "  peft mixed rc=$?"

echo "=== [5/5] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 1 --mode base \
  --warmup 10 --iters 50 \
  --out "$R/peft_base_$TAG.csv" > "$R/peft_base_$TAG.log" 2>&1
echo "  peft base rc=$?"

if [ "$SWEEP_INCOMPLETE" -ne 0 ]; then
  echo "FAILED: the LateFuse sweep did not complete its grid -- do not publish"
  echo "these numbers or patch the gap with a second run. Fix the cause and"
  echo "re-run the sweep."
  exit 1
fi
echo "ALL DONE ($TAG)"
