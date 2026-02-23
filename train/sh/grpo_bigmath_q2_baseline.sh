#!/bin/bash

#SBATCH --output=log/grpo_bigmath_q2_baseline/%j.out 
#SBATCH --error=log/grpo_bigmath_q2_baseline/%j.err

#SBATCH --job-name=grpo_q2_baseline
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=6:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


# sft_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft"
grpo_output_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_grpo_baseline"
grpo_init_model="Qwen/Qwen2.5-3B"

labeling_method="gradient"

n_samples=256
n_grpo_training_samples=2048

grpo_save_steps=16
grpo_batch_size_per_device=2


STEP=0


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
    --grpo_epochs 2 \
    --step $STEP
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 


done

# accelerate launch --num_processes 4 -m grpo_train \
#     --grpo_saving_steps 4 \
#     --ds_cfg "ds_zero3.json"

# # unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 

# python -m grpo_train \
#     --grpo_sample \

# python -m dpo_train \
#     --labeling_method "gradient" \


