#!/bin/bash

sft_model="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"

output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/sh_dpo"

baseline_km_centers="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/baseline_softf1_km_arlsat_q3"

python gen_dpo_arlsat_q3_gradient_soft_jobs.py \
  --out_dir $output_dir \
  --grpo_init_models Qwen/Qwen3-4B $sft_model \
  --cluster_semantics_list majority baseline_centroids gpt \
  --baseline_km_centers $baseline_km_centers \
  --cluster_soft_prob_n_runs 8

