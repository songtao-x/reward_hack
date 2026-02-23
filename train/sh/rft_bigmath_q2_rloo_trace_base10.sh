#!/bin/bash

#SBATCH --output=log/rft_trace_bigmath_q2_rloo_base10/%j.out 
#SBATCH --error=log/rft_trace_bigmath_q2_rloo_base10/%j.err

#SBATCH --job-name=rft-rloo-t-10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


# init model is 10 steps rloo model

sft_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft_rloo_base10"
# rloo_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft_rloo_base10"

sft_base_model_dir="xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_10"

labeling_method="trace"

n_samples=2048
n_rl_samples=32


# global train batch size = 8, total data size per step = 2048
sft_save_steps=128
sft_batch_size_per_device=2

# sft train params
sft_epochs=1
save_total_limit=2

# sampling params
sft_temp=0
sft_top_p=1

step=0

for STEP in $(seq 0 1 10); do
    echo "Training step: $STEP"
    # previous step SFT model as input to current step RLOO
    sft_input_model_dir="$sft_output_dir/$labeling_method/s$((STEP-1))/sft"
    echo "Training on the model: $sft_input_model_dir"


    # run sft training (kept as-is)
    python -m sft_train \
        --step $STEP \
        --sft_only \
        --labeling_method $labeling_method \
        --sft_input_model_dir $sft_input_model_dir \
        --sft_base_model_dir $sft_base_model_dir \
        --sft_output_dir $sft_output_dir \
        --n_samples $n_samples \
        --n_rl_samples $n_rl_samples \
        --sft_temp $sft_temp \
        --sft_top_p $sft_top_p \
        --sft_batch_size_per_device $sft_batch_size_per_device \
        --sft_save_steps $sft_save_steps \
        --sft_epochs $sft_epochs \
        --sft_save_total_limit $save_total_limit \
        

done
