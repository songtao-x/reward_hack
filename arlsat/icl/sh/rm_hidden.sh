#!/bin/bash


model="hidden"
data_type="resp"

export WANDB_PROJECT="reward_model"
export WANDB_NAME="reward_model_${model}_${data_type}"


output_dir="checkpoints/reward_model_${model}_${data_type}"
logging_dir="logs/reward_model_${model}_${data_type}"

trainset_path="data/train/rm_trainset_${data_type}_prob.json"
testset_path="data/test/rm_testset_${data_type}_prob.json"

n_gpu=4
hidden_pooling='selected'
lr=1e-4
epochs=10
per_device_train_batch_size=2
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


# 5e-4, 8, 30, hidden, 2-layer, resp, sft, mean: 0.59
# 1e-3, 8, 30, hidden, 2-layer, resp, sft, mean: 0.59
# 1e-3, 8, 30, hidden, 1-layer, resp, sft, mean: 0.59
# 5e-3, 4, 30, hidden, 1-layer, resp, sft, mean: 0.61
# 5e-3, 8, 30, hidden, 1-layer, resp, sft, mean: 
# 5e-3, 4, 30, hidden, 2-layer, resp, sft, mean: 
# 2e-3, 4, 30, hidden, 2-layer, resp, sft, mean: 0.5
# 5e-3, 4, 30, hidden, 2-layer, resp, sft, mean: 0.5
# 5e-4, 4, 30, hidden, 2-layer, resp, sft, mean: 0.59
# 2e-4, 4, 2-layers, mean: 0.6
# 2e-4, 4, 1-layers, resp, sft, mean: 0.6
# 2e-4, 4, 1-layers, resp, sft, last: 0.5
# 2e-3, 4, 1-layers, resp, sft, last: 





torchrun --nproc_per_node=$n_gpu rm/reward_model_all.py \
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

