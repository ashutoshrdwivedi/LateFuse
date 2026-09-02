#!/bin/bash
# A100-80GB -- mixed-rank serving.
#
# Does the flat-in-N latency of Fig 3 hold under mixed-rank deployments (e.g.
# some adapters r=4, others r=16 in the same batch)? And does padding low-rank
# adapters up to the batch maximum waste enough compute for the overhead to
# matter?
#
# The paper's rank sweep (Finding 3) varies r UNIFORMLY. Zero-padding makes a
# mixed batch bit-identically shaped to a uniform r_max batch, so that sweep
# already bounds the mixed case -- but only by argument. This script measures
# it: heterogeneous fleets against both uniform references, at four pool sizes,
# so flat-in-N and rank-neutrality are shown on the same axes at once.
#
# Two artifacts, because the two mixes have different memory ceilings and one
# file per artifact is the rule here:
#   mixed_rank_4_16.*     the 50/50 r=4 / r=16 case.
#                         r_max=16 -> 3.14 MB/adapter, so N reaches 20,000.
#   mixed_rank_spread.*   full-spread worst case, 25% each of r=4,8,16,32.
#                         r_max=32 -> 6.29 MB/adapter, so N stops at 10,000.
# Both peak near 63 GB of adapter store at their top N; expandable_segments is
# not optional here (see the fragmentation note in run_l40s.sh).
#
# A third, short arm re-measures the 4/16 mix under the CPU-loop
# BatchAssembler -- the assembler the published figures used -- so the headline
# can be quoted either way without a second pod session.
#
# Runtime ~2h. The store is rebuilt per (seed, N, arm) because seed is the
# outermost loop, matching latefuse.benchmark.run: between-seed spread then
# absorbs thermal and clock drift rather than only per-iteration jitter. That
# costs ~5x the store-build time and is the single largest term in the runtime;
# it is deliberate, not an oversight.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/LateFuse
R=benchmarks/results/mixed_rank
mkdir -p "$R"
FAILED=0

M="BAAI/bge-m3"

echo "=== [0/4] padding identity on CPU (no GPU time, fails in seconds) ==="
# The GPU arms are only interpretable if zero-padding is exact; that identity
# is pinned on CPU in fp32, so check it before the pod does any work.
uv run pytest tests/test_mixed_rank_padding.py -q > "$R/padding_tests.log" 2>&1
rc=$?; echo "  padding tests rc=$rc"
[ $rc -ne 0 ] && { echo "PADDING TESTS FAILED -- see $R/padding_tests.log"; tail -30 "$R/padding_tests.log"; exit 1; }

echo "=== [1/4] smoke test (fail fast before burning the pod) ==="
uv run python -m benchmarks.profiling.mixed_rank_bench \
  --model "$M" --dtype fp16 --adapters 64 --batch-size 8 \
  --rank-mix 4,16 --seq-len 128 --warmup 5 --iters 10 --check-adapters 32 \
  --out "$R/smoke_mixed_rank.txt" > "$R/smoke_mixed_rank.log" 2>&1
rc=$?; echo "  smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_mixed_rank.log"; tail -30 "$R/smoke_mixed_rank.log"; exit 1; }

echo "=== [2/4] mix 4+16, N to 20,000, 5 seeds (~55 min) ==="
# 4 arms (uniform_r4, uniform_r16, mixed padded, mixed native) x 5 N x 5 seeds.
uv run python -m benchmarks.profiling.mixed_rank_bench \
  --model "$M" --dtype fp16 \
  --adapters 100 1000 5000 10000 20000 --batch-size 32 \
  --rank-mix 4,16 --seq-len 128 \
  --warmup 50 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/mixed_rank_4_16.txt" > "$R/mixed_rank_4_16.log" 2>&1
rc=$?; echo "  4+16 rc=$rc"
if [ $rc -ne 0 ]; then
  FAILED=1
  echo "  !! 4+16 ARM FAILED -- continuing so the remaining arms still land."
fi

echo "=== [3/4] mix 4+8+16+32, N to 10,000, 5 seeds (~50 min) ==="
# r_max=32 doubles per-adapter bytes against the 4+16 arm, so the top N halves.
uv run python -m benchmarks.profiling.mixed_rank_bench \
  --model "$M" --dtype fp16 \
  --adapters 100 1000 5000 10000 --batch-size 32 \
  --rank-mix 4,8,16,32 --seq-len 128 \
  --warmup 50 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/mixed_rank_spread.txt" > "$R/mixed_rank_spread.log" 2>&1
rc=$?; echo "  spread rc=$rc"
if [ $rc -ne 0 ]; then
  FAILED=1
  echo "  !! SPREAD ARM FAILED -- continuing."
fi

echo "=== [4/4] same mix under the paper's CPU assembler (~10 min) ==="
# The published figures assemble with the O(B*L*|M|) Python loop, so the
# headline needs a cell measured under it too -- otherwise the mixed-rank
# result and Figure 3 are not quite the same experiment. One N is enough: the
# assembler is rank-independent by construction, this only confirms it.
uv run python -m benchmarks.profiling.mixed_rank_bench \
  --model "$M" --dtype fp16 --assembler baseline \
  --adapters 1000 --batch-size 32 \
  --rank-mix 4,16 --seq-len 128 \
  --warmup 50 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/mixed_rank_4_16_cpuasm.txt" > "$R/mixed_rank_4_16_cpuasm.log" 2>&1
rc=$?; echo "  cpu-assembler rc=$rc"
if [ $rc -ne 0 ]; then
  FAILED=1
  echo "  !! CPU-ASSEMBLER ARM FAILED."
fi

if [ "$FAILED" -ne 0 ]; then
  echo "FAILED: at least one arm did not complete -- do not publish these"
  echo "numbers or patch the gap with a second run. Fix the cause and re-run"
  echo "the arm. Check the *.log files in $R."
  exit 1
fi
echo "ALL DONE (mixed_rank)"
