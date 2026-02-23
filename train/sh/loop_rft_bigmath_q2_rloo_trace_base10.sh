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
#SBATCH --time=6:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


# init model is 10 steps rloo model

sft_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft_rloo_base10"
rloo_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft_rloo_base10"

rloo_init_model="xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_10"

labeling_method="trace"

n_samples=2048
n_rl_samples=32

rloo_save_steps=4
rloo_batch_size_per_device=2

sft_epochs=1
save_total_limit=1

for STEP in $(seq 0 1 40); do
    echo "Training step: $STEP"

    # previous step SFT model as input to current step RLOO
    rloo_input_model="$rloo_output_dir/$labeling_method/s$((STEP-1))/sft"
    echo "RLOO training on the model: $rloo_input_model"

    # run rloo training
    accelerate launch --num_processes 4 -m rloo_train \
        --dataset "bigmath" \
        --train_type "sft" \
        --cluster_method "kmeans" \
        --n_samples $n_samples \
        --n_rl_samples $n_rl_samples \
        --labeling_method $labeling_method \
        --rloo_init_model $rloo_init_model \
        --rloo_input_model $rloo_input_model \
        --rloo_output_dir $rloo_output_dir \
        --rloo_batch_size_per_device $rloo_batch_size_per_device \
        --rloo_save_steps $rloo_save_steps \
        --ds_cfg "ds_zero3.json" \
        --step $STEP

    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

    # run sft training (kept as-is)
    python -m sft_train \
        --labeling_method $labeling_method \
        --n_samples $n_samples \
        --n_rl_samples $n_rl_samples \
        --sft_output_dir $sft_output_dir \
        --sft_save_steps 4 \
        --step $STEP \
        --sft_epochs $sft_epochs

done