#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/train/output/codegen_dpo_pipeline}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-Qwen/Qwen2.5-3B-Instruct}"
DPO_MODEL_DIR="${DPO_MODEL_DIR:-${BASE_MODEL_DIR}}"
STEP="${STEP:-0}"

python -m train.codegen.dpo_codegen_train \
  --output_root "${OUTPUT_ROOT}" \
  --labeling_method trace \
  --step "${STEP}" \
  --base_model_dir "${BASE_MODEL_DIR}" \
  --dpo_model_dir "${DPO_MODEL_DIR}"
