#!/bin/bash
# A100-80GB -- ELECTRA-large generality row.
#
# Architecture-generality row beyond the BERT/RoBERTa family.
# ELECTRA is a replaced-token-detection discriminator, a different pretraining
# family from bge-m3's masked-LM XLM-RoBERTa, with WordPiece 30k vs
# SentencePiece 250k vocabulary and absolute (arange) position ids rather than
# RoBERTa's mask-offset scheme. (The position-id scheme is the checkpoint's own
# under --engine hf; EncoderWithLora.position_ids is not on this path.)
#
# This row runs on the HF hook wrapper (--engine hf), like the DeBERTa and
# XLM-R-XL rows. ELECTRA checkpoints ship no pooler, and the custom encoder's
# loader is strict about every parameter it declares, so it rejects them (see
# src/latefuse/model/encoder.py:205) rather than serving CLS through a
# randomly-initialised pooler. The wrapper falls back to the raw CLS state,
# which is what a pooler-less checkpoint's own HF forward returns.
#
# NOTE: the committed CSVs in benchmarks/results/electra/ predate this
# switch and were measured on the custom encoder. They are valid as wall-clock
# (the pooler is one B x H x H GEMM out of 24 layers and was never read for
# anything but latency), but re-run this script before citing the row as
# wrapper-measured, and keep the reported engine attribution in sync.
#
# Same L=24, d=1024 adapter geometry as bge-m3 -> identical 1.57 MB/adapter and
# the same ~47k ceiling, which is itself a clean confirmation that Finding 4's
# ceiling is set by adapter geometry (L, d, r), not by model size or vocabulary.
#
# Grid and measurement kernel both mirror run_sxm80.sh: the full N x B cross
# product plus the same three rank cells at the (1000, 32) operating point, so
# every Table 2 cell has an ELECTRA counterpart and any of them can be cited
# without a second pod session.
# 6 N x 5 B + 3 rank cells = 33 configs x 5 seeds = 165 runs.
#
# Runtime ~2h30m. Independent of the other generality scripts. If you have a
# spare slot on this pod, run_assembly.sh (~25 min) pairs well here.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk (see
# run_deberta.sh: a default 20G container disk fills mid-sweep).
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
# Required for the N=47000 cells -- see the fragmentation note in
# run_l40s.sh. The committed sweep landed all 165 rows without it, but
# only barely: N=47000 peaks at 75.7 GB and the probe OOMs at 51000 after 49000
# reaches 78.9/80 GB. That is well inside the fragmentation margin that cost
# the L40S row its entire top-N column -- do not rely on the same luck.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/LateFuse
R=benchmarks/results/electra
mkdir -p "$R"
SWEEP_INCOMPLETE=0

M="google/electra-large-discriminator"
TAG=electra

echo "=== [0/5] smoke test (fail fast before burning the pod) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --dtype fp16 --engine hf --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 \
  --adapters 2 --mode mixed --warmup 2 --iters 5 \
  --out "$R/smoke_peft_$TAG.csv" > "$R/smoke_peft_$TAG.log" 2>&1
rc=$?; echo "  peft smoke rc=$rc"
[ $rc -ne 0 ] && { echo "PEFT SMOKE FAILED -- see $R/smoke_peft_$TAG.log"; tail -20 "$R/smoke_peft_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  --model "google/electra-large-discriminator:query,value" --out-dir "$R" \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

echo "=== [2/5] LateFuse sweep, full grid, 5 seeds (~1h50m) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --dtype fp16 --engine hf \
  --adapters 100 1000 5000 10000 20000 47000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 \
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
# bracketed the ceiling at 49000 fits (78.9 GB) / 51000 OOM under the DEFAULT
# allocator; recovering the fragmentation headroom moves that up (the L40S row
# gained 7.7% capacity from the same setting), so 51000 may now fit and 53000
# is what keeps the ceiling bracketed. An OOM cell is recorded as a status=oom
# row rather than skipped, and this probe deliberately omits --require-complete,
# so a spare cell costs a few minutes, never fails the run, and leaves a record
# of exactly where the card gave out.
echo "=== [3/5] capacity probe (OOM ceiling) ==="
uv run python -m latefuse.benchmark.run \
  --model "$M" --dtype fp16 --engine hf \
  --adapters 47000 49000 51000 53000 --batch-sizes 32 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_${TAG}_capacity.csv" > "$R/sweep_${TAG}_capacity.log" 2>&1
echo "  capacity rc=$?"

echo "=== [4/5] PEFT mixed baseline, same node (~30 min) ==="
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
