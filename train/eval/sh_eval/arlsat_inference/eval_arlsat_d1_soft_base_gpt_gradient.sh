#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_inference/log/arlsat_d1_soft_base_gpt_gradient/%j.out
#SBATCH --error=/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_inference/log/arlsat_d1_soft_base_gpt_gradient/%j.err
#SBATCH --job-name=eval_gradient
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=08:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


mkdir -p "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_inference/log/arlsat_d1_soft_base_gpt_gradient"

cd "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack"

mkdir -p "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_base_gpt/gradient/s0"
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_base_gpt/gradient/s0/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_base_gpt/gradient/s0" \
  --skip-existing

mkdir -p "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_base_gpt/gradient/s1"
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_base_gpt/gradient/s1/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_base_gpt/gradient/s1" \
  --skip-existing

mkdir -p "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_base_gpt/gradient/s2"
python -m train.eval.arlsat_eval_single \
  --model-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft_base_gpt/gradient/s2/dpo" \
  --eval-dir "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1_soft_base_gpt/gradient/s2" \
  --skip-existing
