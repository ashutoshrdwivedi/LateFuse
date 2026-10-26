#!/bin/bash
# A100-80GB -- DeBERTa-v2-xlarge generality row.
#
# Architecture-generality row beyond the BERT/RoBERTa family, at larger scale:
# 900M params, d=1536, disentangled attention. Runs on the
# stock HF forward with hook-injected LoRA (--engine hf) because the repo's
# custom encoder cannot load disentangled attention; the hook path is
# PEFT-parity tested in tests/test_hf_wrapper.py.
#
# TARGET MODULES: value only, NOT the paper's query+value. This is deliberate.
# deberta-v2-xlarge sets share_att_key=true, so query_proj is applied both to
# the per-sample hidden states AND to the batch-shared relative-position
# embeddings, whose result HF then tiles across the batch (`.repeat(...)` in
# DisentangledSelfAttention.disentangled_attention_bias). PEFT wraps the Linear
# and therefore adapts that shared call; per-tenant serving has no per-sample
# counterpart for a batch-independent tensor, so we skip it -- and the outputs
# genuinely diverge, measured at 10.3% relative on this checkpoint.
# value_proj is never applied to the position embeddings, so targeting it alone
# makes the comparison against PEFT exact (2.2e-5, fp32 round-off on 900M
# params). Both facts are asserted in tests/test_real_checkpoint_parity.py.
#
# Grid and measurement kernel both mirror the paper's run_sxm80.sh: the full
# N x B cross product at fp16, seq=128, r=8, warmup 50 / iters 200, 5 seeds,
# plus the same three rank cells at the (1000, 32) operating point (PEFT arms
# at warmup 10 / iters 50, and at B=8/32/128 only, exactly as in the paper).
# Measuring the whole grid rather than a subset means every cell Table 2 has
# for bge-m3 has a counterpart here, so any of them can be cited without a
# second pod session.
#
# Ceiling: 1 module x 2 (A,B) x 24 layers x 1536 x r8 x 2B = 1.18 MB/adapter
# -> formula says ~70k on 80GB; applying the same ~12% overestimate the paper
# saw for bge-m3 (formula 52.8k vs measured 47k) gives a practical ~62k. The
# sweep therefore runs out to 60k and the probe brackets it.
#
# WHAT THE COMMITTED RESULTS ACTUALLY CONTAIN. The archived run predates the
# expandable_segments export below, so sweep_deberta_a100.csv stops at N=40000
# (165 rows, not 190) and the probe was re-bracketed downward mid-session --
# hence 45k/50k/55k in sweep_deberta_capacity.csv plus 58000 (72.7 GB peak) in
# sweep_deberta_capacity2.csv, rather than the 60k/64k/68k this script asks
# for. The paper quotes the 58k ceiling from that re-bracketed probe. A rerun
# with the allocator fix should recover the N=60000 column and can use the
# probe grid as written; expect the committed and rerun capacity CSVs to differ.
#
# Runtime ~3h35m (DeBERTa's forward is ~2.6x bge-m3's, so the full grid costs
# more here than the paper's 2h). Independent of the other generality scripts --
# safe to run concurrently on a separate pod.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk. The HF
# caches (deberta-v2-xlarge 3.4G, bge-m3 2.6G, electra 1.3G) plus the venv
# (~7G) overflow a default 20G container disk and abort a sweep mid-run.
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
# Required for the top-N cells, not a tuning knob -- see the fragmentation note
# in run_l40s.sh for the full failure mode. The committed
# sweep_deberta_a100.csv was produced WITHOUT this and shows exactly that
# symptom: 165 rows instead of 190, with the whole N=60000 column absent while
# the sweep still exited rc=0. The re-bracketed probe puts the real ceiling at
# 58000 (72.7 GB peak), so 60k is within reach once the ~GB-scale
# reserved-but-unallocated fragmentation is recovered.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/LateFuse
R=benchmarks/results/deberta
mkdir -p "$R"
SWEEP_INCOMPLETE=0

M="microsoft/deberta-v2-xlarge"
TAG=deberta

echo "=== [0/5] smoke test (fail fast before burning the pod) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --engine hf --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --target-modules value --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 \
  --adapters 2 --mode mixed --target-modules value_proj \
  --warmup 2 --iters 5 \
  --out "$R/smoke_peft_$TAG.csv" > "$R/smoke_peft_$TAG.log" 2>&1
rc=$?; echo "  peft smoke rc=$rc"
[ $rc -ne 0 ] && { echo "PEFT SMOKE FAILED -- see $R/smoke_peft_$TAG.log"; tail -20 "$R/smoke_peft_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  --model "microsoft/deberta-v2-xlarge:value" --out-dir "$R" \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

# --extra-configs folds the r!=8 cells into this same run so the shared
# (1000, 32, r8) cell is measured exactly once and all four ranks are timed
# back-to-back within each seed pass (one thermal envelope), matching the
# paper's rationale for not running ranks as a separate later sweep.
# 7 N x 5 B + 3 rank cells = 38 configs x 5 seeds = 190 runs.
echo "=== [2/5] LateFuse sweep, full grid, 5 seeds (~2h50m) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 100 1000 5000 10000 20000 40000 60000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 --target-modules value \
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

# Widened at the bottom end so this one probe brackets the ceiling on its own.
# The archived run needed a hand-driven second pass (45k/50k/55k, then 58000 in
# a capacity2 file) because the grid started above where the default allocator
# actually topped out. 56000 is below the measured 58k fit, 68000 is above any
# plausible expandable_segments ceiling (~62k), so the bracket holds without a
# follow-up. An OOM cell is recorded as a status=oom row, and this probe omits
# --require-complete, so the extra cell is nearly free and self-documenting.
echo "=== [3/5] capacity probe (OOM ceiling) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 56000 60000 64000 68000 --batch-sizes 32 --lora-ranks 8 \
  --target-modules value --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_${TAG}_capacity.csv" > "$R/sweep_${TAG}_capacity.log" 2>&1
echo "  capacity rc=$?"

# Only the 'mixed' mode is reported here (speedup vs PEFT's native
# mixed-batch API) plus 'base' for the single-tenant ceiling. grouped /
# homogeneous / sequential are paper-only and skipped here.
echo "=== [4/5] PEFT mixed baseline, same node (~35 min) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 100 1000 --mode mixed \
  --target-modules value_proj --warmup 10 --iters 50 \
  --out "$R/peft_mixed_$TAG.csv" > "$R/peft_mixed_$TAG.log" 2>&1
echo "  peft mixed rc=$?"

echo "=== [5/5] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 1 --mode base \
  --target-modules value_proj --warmup 10 --iters 50 \
  --out "$R/peft_base_$TAG.csv" > "$R/peft_base_$TAG.log" 2>&1
echo "  peft base rc=$?"

if [ "$SWEEP_INCOMPLETE" -ne 0 ]; then
  echo "FAILED: the LateFuse sweep did not complete its grid -- do not publish"
  echo "these numbers or patch the gap with a second run. Fix the cause and"
  echo "re-run the sweep."
  exit 1
fi
echo "ALL DONE ($TAG)"
