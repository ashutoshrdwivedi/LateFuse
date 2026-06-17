#!/bin/bash
# Driver: SetFit accuracy benchmarks (LoRA vs vanilla, plus frozen baseline)
# on the GPU host. Three legs, all restartable via --resume (rows already in
# the output CSV are skipped, so re-running a completed leg is a no-op):
#
#   1. mpnet  : LoRA vs vanilla on paraphrase-mpnet-base-v2 (matches the
#               SetFit paper setup; PEFT q/v projections are named 'q'/'v').
#               -> setfit_mpnet_multiseed.csv (paper/table_accuracy.tex)
#   2. bge    : same protocol on BAAI/bge-m3, the serving model. bge-m3 is
#               XLM-RoBERTa-derived, so the projections are 'query'/'value'.
#               -> setfit_bge_multiseed.csv (paper/table_accuracy_bge.tex)
#   3. bge frozen : no-adaptation baseline -- frozen bge-m3 embeddings + LR
#               head only. No mpnet equivalent is run: the SetFit paper
#               already reports that comparison, we cite it instead.
#               -> appends to setfit_bge_multiseed.csv (Frozen column +
#               SetfitBge*Frozen macros, picked up automatically)
#
# Shared protocol: 7 datasets x 10 seeds, n=8 per class, rank 8,
# max_seq_length 256, full test splits. LR matrix: vanilla @2e-5; LoRA @2e-5
# plus 1e-4/3e-4/5e-4; frozen has no body LR. 350 fine-tune configs per
# model; bge-m3 (~568M params) is ~5x mpnet per step -- budget several hours
# on an H100/A100. The frozen leg trains nothing (head fit only) and runs in
# minutes.
#
# Output CSVs are consumed by paper/build_numbers.py.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /root/lora_scaling
Q=benchmarks/quality

DATASETS=(
  -d SetFit/sst2 -d SetFit/sst5 -d SetFit/CR
  -d SetFit/amazon_counterfactual_en -d SetFit/emotion
  -d SetFit/enron_spam -d SetFit/ag_news
)
SEEDS=(
  --seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4
  --seeds 5 --seeds 6 --seeds 7 --seeds 8 --seeds 9
)
PROTOCOL=(--n-per-class 8 --lora-rank 8 --resume)
LORA_LRS=(--lora-lrs 1e-4 --lora-lrs 3e-4 --lora-lrs 5e-4)

echo "=== SetFit accuracy: paraphrase-mpnet-base-v2, 10 seeds ==="
uv run python benchmarks/quality/setfit_compare.py \
  --base-model sentence-transformers/paraphrase-mpnet-base-v2 \
  -t q -t v \
  "${DATASETS[@]}" "${SEEDS[@]}" "${LORA_LRS[@]}" "${PROTOCOL[@]}" \
  --out "$Q/setfit_mpnet_multiseed.csv" > "$Q/setfit_mpnet_multiseed.log" 2>&1
echo "  setfit mpnet rc=$?"

echo "=== SetFit accuracy: BAAI/bge-m3, 10 seeds ==="
uv run python benchmarks/quality/setfit_compare.py \
  --base-model BAAI/bge-m3 \
  -t query -t value \
  "${DATASETS[@]}" "${SEEDS[@]}" "${LORA_LRS[@]}" "${PROTOCOL[@]}" \
  --out "$Q/setfit_bge_multiseed.csv" > "$Q/setfit_bge_multiseed.log" 2>&1
echo "  setfit bge rc=$?"

echo "=== SetFit accuracy: BAAI/bge-m3 frozen baseline, 10 seeds ==="
uv run python benchmarks/quality/setfit_compare.py \
  --base-model BAAI/bge-m3 \
  -t query -t value \
  -m frozen \
  "${DATASETS[@]}" "${SEEDS[@]}" "${PROTOCOL[@]}" \
  --out "$Q/setfit_bge_multiseed.csv" > "$Q/setfit_bge_frozen.log" 2>&1
echo "  setfit bge frozen rc=$?"

echo "ALL DONE"
