#!/bin/bash

#SBATCH --output=log/dpo_gradient_arlsat_q3_soft/%j.out 
#SBATCH --error=log/dpo_gradient_arlsat_q3_soft/%j.err

#SBATCH --job-name=dpo
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


dpo_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft"
grpo_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1_soft"
grpo_init_model="Qwen/Qwen3-4B"

labeling_method="gradient"
cluster_method="soft_prob"
cluster_semantics="majority"
cluster_soft_prob_temp=1.0
cluster_soft_prob_n_runs=8

n_samples=32
n_grpo_training_samples=32

grpo_save_steps=4
grpo_batch_size_per_device=2

dpo_epochs=1

for STEP in $(seq 0 1 40); do

    echo "Training step: $STEP"
    grpo_input_model="$grpo_output_dir/$labeling_method/s$((STEP-1))/dpo"

    echo "GRPO training on the model: $grpo_input_model"

    # run grpo training
    accelerate launch --num_processes 4 -m grpo_train \
        --labeling_method $labeling_method \
        --cluster_method $cluster_method \
        --cluster_semantics $cluster_semantics \
        --cluster_soft_prob_temp $cluster_soft_prob_temp \
        --cluster_soft_prob_n_runs $cluster_soft_prob_n_runs \
        --n_samples $n_samples \
        --n_grpo_training_samples $n_grpo_training_samples \
        --grpo_init_model $grpo_init_model \
        --grpo_input_model $grpo_input_model \
        --grpo_output_dir $grpo_output_dir \
        --grpo_batch_size_per_device $grpo_batch_size_per_device \
        --grpo_save_steps $grpo_save_steps \
        --ds_cfg "ds_zero3.json" \
        --step $STEP
        
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 

    # run grpo sampling
    python -m grpo_train \
        --labeling_method $labeling_method \
        --cluster_method $cluster_method \
        --cluster_semantics $cluster_semantics \
        --cluster_soft_prob_temp $cluster_soft_prob_temp \
        --cluster_soft_prob_n_runs $cluster_soft_prob_n_runs \
        --grpo_output_dir $grpo_output_dir \
        --grpo_sample \
        --step $STEP
    
    # run dpo training
    python -m dpo_train \
        --dpo_output_dir $dpo_output_dir \
        --labeling_method $labeling_method \
        --dpo_save_step 4 \
        --dpo_epochs $dpo_epochs \
        --step $STEP
done


