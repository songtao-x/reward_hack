#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA1="${REPO_ROOT}/code/Reward-Hacking-main/examples/programming/data/all_labeled/data_1-4_tstMin6_tstMax50.jsonl"
DATA2="${REPO_ROOT}/code/Reward-Hacking-main/examples/programming/data/all_labeled/data_5-10_tstMin6_tstMax50.jsonl"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/train/output/codegen_dpo_pipeline}"
REWARD_URL="${REWARD_URL:-http://localhost:8118/batched_unittest}"
STEP="${STEP:-0}"

# NOTE: trace threshold is a placeholder; calibrate on codegen before using seriously.
TRACE_THRESHOLD="${TRACE_THRESHOLD:-0.0749}"

# Produces: train/output/.../trace/s${STEP}/trace/dpo_trainset.json
python -m train.codegen.codegen_sample_label_dpo \
  --data "${DATA1}" "${DATA2}" \
  --model "${MODEL}" \
  --output_root "${OUTPUT_ROOT}" \
  --reward_url "${REWARD_URL}" \
  --step "${STEP}" \
  --labeling_method trace \
  --trace_threshold "${TRACE_THRESHOLD}" \
  --k 8 \
  --k_max 32 \
  --sample_batch_size 8 \
  --temperature 0.7 \
  --top_p 0.9
