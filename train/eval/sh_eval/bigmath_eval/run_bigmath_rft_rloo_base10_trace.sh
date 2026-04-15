#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/bigmath_eval/log/rft-rloo-base10-trace/%j.out
#SBATCH --error=/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/bigmath_eval/log/rft-rloo-base10-trace/%j.err
#SBATCH --job-name=bigmath_eval_trace
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=2:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

PROJECT_ROOT="${PROJECT_ROOT:-/home/songtaow/projects/aip-xiye17/songtaow/reward_hack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/train/output/bigmath_rft_full_rloo_base10}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/train/eval/data/bigmath_rft_full_rloo_base10}"
TP_SIZE="${TP_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"

mkdir -p "$EVAL_ROOT"
cd "$PROJECT_ROOT"

python -m train.eval.bigmath_eval \
  --output-root "$OUTPUT_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --method trace \
  --do-labeling \
  --skip-existing \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --tensor-parallel-size "$TP_SIZE" \
  "$@"
