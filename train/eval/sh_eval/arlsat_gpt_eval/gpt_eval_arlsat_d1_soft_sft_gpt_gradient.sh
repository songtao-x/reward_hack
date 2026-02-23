#!/bin/bash

cd "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack"

echo "=== arlsat_d1_soft_sft_gpt/gradient/s0 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s0/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s0/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s0/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s0" \
  --skip-existing \
  --do-gpt-eval

echo "=== arlsat_d1_soft_sft_gpt/gradient/s1 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s1/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s1/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s1/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s1" \
  --skip-existing \
  --do-gpt-eval

echo "=== arlsat_d1_soft_sft_gpt/gradient/s2 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s2/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s2/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s2/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s2" \
  --skip-existing \
  --do-gpt-eval

echo "=== arlsat_d1_soft_sft_gpt/gradient/s3 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s3/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s3/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s3/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s3" \
  --skip-existing \
  --do-gpt-eval

echo "=== arlsat_d1_soft_sft_gpt/gradient/s4 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s4/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s4/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s4/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s4" \
  --skip-existing \
  --do-gpt-eval

echo "=== arlsat_d1_soft_sft_gpt/gradient/s5 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s5/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s5/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s5/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s5" \
  --skip-existing \
  --do-gpt-eval

echo "=== arlsat_d1_soft_sft_gpt/gradient/s6 ==="
if [ ! -f "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s6/inference.json" ]; then
  echo "Skip (missing inference): /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s6/inference.json"
  continue
fi
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_sft_gpt/gradient/s6/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_sft_gpt/gradient/s6" \
  --skip-existing \
  --do-gpt-eval
