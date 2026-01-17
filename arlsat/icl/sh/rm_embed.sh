
#!/bin/bash

model="embed"
data_type="resp"

export WANDB_PROJECT="reward_model"
export WANDB_NAME="reward_model_${model}_${data_type}"


output_dir="checkpoints/reward_model_${model}_${data_type}"
logging_dir="logs/reward_model_${model}_${data_type}"

trainset_path="data/train/rm_trainset_resp.json"
testset_path="data/test/rm_testset_resp.json"

n_gpu=4
lr=1e-3
epochs=20
per_device_train_batch_size=4
# 2e-3, 2, 15: embed full: 0.655
# 1e-3, 2, 20: embed full: 0.665

# sft base model: 

torchrun --nproc_per_node=$n_gpu reward_model_all.py \
    --model=$model \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --trainset_path=$trainset_path \
    --testset_path=$testset_path \
    --data_type=$data_type \
    --output_dir=$output_dir \
    --logging_dir=$logging_dir \
    --lr=$lr \
    --epochs=$epochs \

