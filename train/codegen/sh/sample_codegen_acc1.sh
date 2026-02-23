#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_EVAL="${REPO_ROOT}/code/Reward-Hacking-main/examples/programming/data/all_labeled/data_5-10_tstMin6_tstMax50.jsonl"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/train/output/codegen_acc1_eval}"
REWARD_URL="${REWARD_URL:-http://localhost:8118/batched_unittest}"

# Deterministic generation for clean acc@1
python -m train.codegen.sample_codegen_acc1 \
  --data "${DATA_EVAL}" \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_url "${REWARD_URL}" \
  --temperature 0.0 \
  --top_p 1.0
