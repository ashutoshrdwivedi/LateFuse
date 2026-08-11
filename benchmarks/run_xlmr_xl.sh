#!/bin/bash
# A100-80GB REQUIRED -- XLM-RoBERTa-XL scale row.
#
# Scale row: 3.48B params, d=2560, L=36 -- 6.1x bge-m3's 568M -- to show the
# O(1)-in-N property holds on a much larger encoder.
#
# Runs on the stock HF forward with hook-injected LoRA (--engine hf). This is
# not optional: XLM-R-XL is pre-LN, so the repo's custom encoder cannot load it
# (146 state-dict key mismatches). The wrapper runs HF's own forward, so the
# base architecture is faithful by construction.
#
# NO SHARED-PROJECTION GAP. Unlike DeBERTa-v2 (share_att_key=true), XLM-R-XL's
# q/k/v are applied only to the per-sample hidden states -- verified by hooking
# every projection and confirming each is called exactly once per layer with
# the full batch dimension. So this row keeps the paper's query+value protocol
# unchanged and its PEFT comparison is exact, with no parity caveat to explain.
#
# SCOPE FRAMING: a 3.5B encoder is a stress test, not a recommended serving
# configuration. The claim is that the late-fusion decomposition still holds at
# this scale and that the per-GPU ceiling falls exactly as Finding 4 predicts --
# not that anyone should serve 3.5B for classification.
#
# Grid and measurement kernel mirror the paper's run_sxm80.sh: full N x B cross
# product at fp16, seq=128, r=8, warmup 50 / iters 200, 5 seeds, plus the same
# three rank cells at the (1000, 32) operating point. PEFT arms at warmup 10 /
# iters 50 and B=8/32/128, exactly as in the paper.
# 7 N x 5 B + 3 rank cells = 38 configs x 5 seeds = 190 runs.
#
# Ceiling: 2 modules x 2 (A,B) x 36 layers x 2560 x r8 x 2B = 5.90 MB/adapter.
# The model itself is ~7 GB fp16, leaving ~76 GB -> formula ~13k; applying the
# ~12% overestimate the paper saw for bge-m3 (formula 52.8k vs measured 47k)
# gives a practical ~11.4k. The sweep runs to 11k and the probe brackets it.
# This is Finding 4 behaving as predicted, not a regression: a 6x larger model
# with 2.5x the width buys correspondingly fewer resident tenants.
#
# DISK: the checkpoint is ~14 GB. Ensure the volume has headroom before
# starting or the download dies mid-run (see the HF_HOME note below).
#
# Runtime ~5.7h (4.7h sweep + ~1h PEFT) -- the long pole among the generality
# scripts. Extrapolated from measured A100 p50 at N=1000: 37.9 ms (B=8),
# 136.7 ms (B=32), 584.1 ms (B=128), i.e. ~4.5-5.5x bge-m3's forward -- less
# than the 9.4x a pure FLOP ratio predicts, since the wider model utilises the
# A100 better. Note ~1.8h of that total is run_single_config reloading the 7 GB
# base model once per cell (190 times); caching it across configs would be the
# single biggest speedup available here.
# Start this one FIRST. Independent of the other generality scripts.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk. XLM-R-XL
# alone is ~14 GB; a default 20 GB container disk cannot hold it plus the venv.
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
# Required for the N=11000 cells -- see the fragmentation note in
# run_l40s.sh. The committed sweep landed all 190 rows without it, but
# the margin is thin: N=11000 peaks at 73.7 GB and the probe OOMs at 13000
# after 12000 reaches 79.8/80 GB. One fragmentation event silently drops the
# entire top-N column and the sweep still exits rc=0 -- which is what happened
# to the DeBERTa row.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/LateFuse
R=benchmarks/results/xlmr_xl
mkdir -p "$R"
SWEEP_INCOMPLETE=0

M="facebook/xlm-roberta-xl"
TAG=xlmrxl

echo "=== [0/5] preflight: GPU size + disk headroom ==="
uv run python -c "
import torch, shutil, sys
vram = torch.cuda.get_device_properties(0).total_memory/1e9
free = shutil.disk_usage('${HF_HOME}').free/1e9
print(f'  GPU: {torch.cuda.get_device_name(0)}  {vram:.1f} GB')
print(f'  free on \$HF_HOME: {free:.1f} GB')
if vram < 70: print('  WARNING: this row is sized for an 80GB A100; a smaller card lowers the ceiling')
if free < 20: print('  WARNING: <20 GB free; the ~14 GB checkpoint may not fit'); sys.exit(1)
" || { echo "PREFLIGHT FAILED"; exit 1; }

echo "=== [0b/5] smoke test (fail fast before burning the pod) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --target-modules query value --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 \
  --adapters 2 --mode mixed --target-modules query value \
  --warmup 2 --iters 5 \
  --out "$R/smoke_peft_$TAG.csv" > "$R/smoke_peft_$TAG.log" 2>&1
rc=$?; echo "  peft smoke rc=$rc"
[ $rc -ne 0 ] && { echo "PEFT SMOKE FAILED -- see $R/smoke_peft_$TAG.log"; tail -20 "$R/smoke_peft_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  --model "facebook/xlm-roberta-xl:query,value" --out-dir "$R" \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

# --extra-configs folds the r!=8 cells into this same run so the shared
# (1000, 32, r8) cell is measured exactly once and all four ranks are timed
# back-to-back within each seed pass (one thermal envelope), matching the
# paper's rationale for not running ranks as a separate later sweep.
echo "=== [2/5] LateFuse sweep, full grid, 5 seeds (~6h) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 100 1000 2000 4000 6000 8000 11000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 --target-modules query value \
  --extra-configs 1000:32:4 1000:32:16 1000:32:32 \
  --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 --require-complete \
  --out "$R/sweep_${TAG}_a100.csv" > "$R/sweep_${TAG}_a100.log" 2>&1
rc=$?; echo "  sweep rc=$rc"
if [ $rc -ne 0 ]; then
  SWEEP_INCOMPLETE=1
  echo "  !! SWEEP INCOMPLETE -- cells are missing from the CSV (status=oom rows"
  echo "     name them). Continuing so the remaining arms still land, but this"
  echo "     script will exit non-zero at the end. See $R/sweep_*.log."
fi

# Extended by one cell for the expandable_segments re-run. The archived probe
# bracketed the ceiling at 12000 fits (79.8 GB) / 13000 OOM under the DEFAULT
# allocator; recovering the fragmentation headroom moves that up (the L40S row
# gained 7.7% capacity from the same setting), so 13000 may now fit and 14000
# is what keeps the ceiling bracketed. An OOM cell is recorded as a status=oom
# row rather than skipped, and this probe deliberately omits --require-complete,
# so a spare cell costs a few minutes, never fails the run, and leaves a record
# of exactly where the card gave out.
echo "=== [3/5] capacity probe (OOM ceiling) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 11000 12000 13000 14000 --batch-sizes 32 --lora-ranks 8 \
  --target-modules query value --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_${TAG}_capacity.csv" > "$R/sweep_${TAG}_capacity.log" 2>&1
echo "  capacity rc=$?"

# Only 'mixed' is reported here (speedup vs PEFT's native mixed-batch
# API) plus 'base' for the single-tenant ceiling. grouped / homogeneous /
# sequential are paper-only and skipped here.
echo "=== [4/5] PEFT mixed baseline, same node (~1h) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 100 1000 --mode mixed \
  --target-modules query value --warmup 10 --iters 50 \
  --out "$R/peft_mixed_$TAG.csv" > "$R/peft_mixed_$TAG.log" 2>&1
echo "  peft mixed rc=$?"

echo "=== [5/5] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 1 --mode base \
  --target-modules query value --warmup 10 --iters 50 \
  --out "$R/peft_base_$TAG.csv" > "$R/peft_base_$TAG.log" 2>&1
echo "  peft base rc=$?"

if [ "$SWEEP_INCOMPLETE" -ne 0 ]; then
  echo "FAILED: the LateFuse sweep did not complete its grid -- do not publish"
  echo "these numbers or patch the gap with a second run. Fix the cause and"
  echo "re-run the sweep."
  exit 1
fi
echo "ALL DONE ($TAG)"
