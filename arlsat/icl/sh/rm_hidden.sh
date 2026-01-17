#!/bin/bash


model="hidden"
data_type="resp"

export WANDB_PROJECT="reward_model"
export WANDB_NAME="reward_model_${model}_${data_type}"


output_dir="checkpoints/reward_model_${model}_${data_type}"
logging_dir="logs/reward_model_${model}_${data_type}"

trainset_path="data/train/rm_trainset_resp.json"
testset_path="data/test/rm_testset_resp.json"

n_gpu=4
hidden_pooling='mean'
lr=1e-3
epochs=30
per_device_train_batch_size=8
# 1e-4, 1, 15: hidden full, base, mean: 0.695
# 2e-4, 2, 15: hidden full, sft, mean: 0.705
# 2e-4, 2, 20: hidden full, sft, last: 0.55
# 1e-4, 4, 20: hidden full, sft, mean: 0.71 - 0.73
# 2e-4, 4, 20: hidden full, sft, mean: 0.71
# 2e-4, 4, 20: hidden full, sft, last:  0.56
# 1e-4, 4, 30: hidden resp, sft, mean: ckpt-35: 0.705
# 1e-3, 4, 20: hidden, 1-layer, resp, sft, mean: 0.665 - 0.71
# 2e-3, 4, 15: hidden, 2-layer, resp, sft, mean: 0.675 - 0.7
# 2e-3, 4, 15: hidden, 1-layer, resp, sft, mean: 0.64 - 0.68





torchrun --nproc_per_node=$n_gpu reward_model_all.py \
    --model=$model \
    --hidden_pooling=$hidden_pooling \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --trainset_path=$trainset_path \
    --testset_path=$testset_path \
    --data_type=$data_type \
    --output_dir=$output_dir \
    --logging_dir=$logging_dir \
    --lr=$lr \
    --epochs=$epochs \

