#!/bin/bash
set -u
export HOME=/workspace
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /workspace/LateFuse

R=benchmarks/results
M="BAAI/bge-m3"
mkdir -p "$R"


echo "=== [1/2] churn sweep: Zipf alpha 1.0 ==="
uv run python -m latefuse.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 20000 30000 40000 47000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 --seeds 1 \
  --churn --churn-num-tenants 75000 --churn-zipf-alpha 1.0 \
  --out "benchmarks/results/churn_alpha_1p0.csv" > "benchmarks/results/churn_alpha_1p0.log" 2>&1

echo "=== [1/4] churn sweep: Zipf alpha 1.1 ==="
uv run python -m latefuse.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 20000 30000 40000 47000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 --seeds 1 \
  --churn --churn-num-tenants 50000 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1.csv" > "benchmarks/results/churn_alpha_1p1.log" 2>&1

echo "ALL DONE"