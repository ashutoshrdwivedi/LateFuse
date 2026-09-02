#!/bin/bash
# Registration-churn measurement.
#
# Answers "what % of serving time is spent in adapter registration under
# realistic churn rates" with the registration path the server actually uses
# (AdapterStore.load_from_file, fed from S3 by deploy/server/reload.py) rather
# than the synthetic-weight generator the current 0.29 ms/adapter figure is
# derived from.
#
# Four phases, ~50 min total on an A100-80GB. Each writes its own CSV: one file
# per logical result, so a phase can be re-run without disturbing the others.
set -u
export HOME=${POD_HOME:-/workspace}
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_DIR:-/workspace/LateFuse}"

R=benchmarks/results
M="BAAI/bge-m3"
CORPUS=${CORPUS_DIR:-/workspace/adapter_corpus}   # ~3.1 GB for 2,000 files at r=8 fp16
mkdir -p "$R"

COMMON="--engine hf --model $M --dtype fp16 --lora-rank 8 --seq-len 128 \
        --batch-size 32 --corpus-dir $CORPUS --corpus-size 2000 --seeds 1 2 3"

# --- [1/4] Registration cost by path, at moderate occupancy and at the ceiling.
# The headline measurement. The three arms exist to explain each other: the
# synthetic arm reproduces the currently reported number, so the gap
# between it and the file arm is reported rather than silently inherited.
# Run at 10 admissions/s for 30 s => ~300 admissions per cell, enough for a
# stable per-admission mean.
echo "=== [1/4] registration cost by path (blocking) ==="
uv run python -m latefuse.benchmark.churn $COMMON \
  --resident 1000 47000 \
  --registration-paths synthetic file file_pinned \
  --churn-modes blocking \
  --admission-rates 10 \
  --duration 30 --warmup 5 \
  --out "$R/churn_registration_paths.csv" > "$R/churn_registration_paths.log" 2>&1

# --- [2/4] Admission rate as the x-axis, blocking vs background.
# Blocking is the worst case and the bound; background is what a real server
# does and is the claim. Reported separately, never merged.
# Note the 0.1/s cells land only ~6 admissions in 60 s -- they characterise the
# *share*, not the per-admission cost, which phase 1 and the 10/s and 100/s
# cells here establish.
echo "=== [2/4] admission-rate sweep at N=1,000 ==="
uv run python -m latefuse.benchmark.churn $COMMON \
  --resident 1000 \
  --registration-paths file \
  --churn-modes blocking background \
  --admission-rates 0.1 1 10 100 \
  --duration 60 --warmup 5 \
  --out "$R/churn_rate_sweep.csv" > "$R/churn_rate_sweep.log" 2>&1

# --- [3/4] Same, at the 47k ceiling.
# Kept as its own phase: allocator pressure at the ceiling is a real finding but
# a separate one, and mixing it into the main sweep is what produced the
# non-monotonic cost curve in the previous churn harness.
echo "=== [3/4] admission-rate sweep at the 47,000 ceiling ==="
uv run python -m latefuse.benchmark.churn $COMMON \
  --resident 47000 \
  --registration-paths file \
  --churn-modes blocking background \
  --admission-rates 1 10 \
  --duration 60 --warmup 5 \
  --out "$R/churn_rate_sweep_ceiling.csv" > "$R/churn_rate_sweep_ceiling.log" 2>&1

# --- [4/4] Cold page cache.
# Steady state is warm; this is the cold-start bound, reported separately. The
# CSV records whether POSIX_FADV_DONTNEED was actually honoured, so a platform
# that silently ignored it cannot be mistaken for a fast cold read.
echo "=== [4/4] cold page cache ==="
uv run python -m latefuse.benchmark.churn $COMMON \
  --resident 1000 \
  --registration-paths file file_pinned \
  --churn-modes blocking \
  --admission-rates 10 \
  --duration 30 --warmup 5 --cold \
  --out "$R/churn_registration_cold.csv" > "$R/churn_registration_cold.log" 2>&1

echo "ALL DONE"
