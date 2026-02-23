#!/bin/bash

#SBATCH --output=log/rft_trace_bigmath_q2_base10/%j.out 
#SBATCH --error=log/rft_trace_bigmath_q2_base10/%j.err

#SBATCH --job-name=rft-trace-10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=6:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


# init model is 10 steps rloo model

sft_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft_base10"
grpo_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft_base10"

grpo_init_model="xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_10"

labeling_method="trace"

n_samples=64
n_grpo_training_samples=32

n_samples=256
n_rl_samples=32


grpo_save_steps=4
grpo_batch_size_per_device=2

sft_epochs=1

for STEP in $(seq 0 1 40); do

    echo "Training step: $STEP"
    grpo_input_model="$grpo_output_dir/$labeling_method/s$((STEP-1))/sft"
    echo "GRPO training on the model: $grpo_input_model"

    # run grpo training
    # grpo_output_dir=$grpo_base_dir/step${STEP}/grpo

    accelerate launch --num_processes 4 -m grpo_train \
        --dataset "bigmath" \
        --train_type "sft" \
        --cluster_method "kmeans" \
        --n_samples $n_samples \
        --n_grpo_training_samples $n_grpo_training_samples \
        --labeling_method $labeling_method \
        --grpo_init_model $grpo_init_model \
        --grpo_input_model $grpo_input_model \
        --grpo_output_dir $grpo_output_dir \
        --grpo_batch_size_per_device $grpo_batch_size_per_device \
        --grpo_save_steps $grpo_save_steps \
        --ds_cfg "ds_zero3.json" \
        --step $STEP
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 

    # run grpo sampling +
    # run sft training
    python -m sft_train \
        --labeling_method $labeling_method \
        --n_samples $n_samples \
        --n_rl_samples $n_rl_samples \
        --sft_output_dir $sft_output_dir \
        --sft_save_steps 4 \
        --step $STEP \
        --sft_epochs $sft_epochs

done

# accelerate launch --num_processes 4 -m grpo_train \
#     --grpo_saving_steps 4 \
#     --ds_cfg "ds_zero3.json"

# # unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 

# python -m grpo_train \
#     --grpo_sample \

# python -m dpo_train \
#     --labeling_method "gradient" \


