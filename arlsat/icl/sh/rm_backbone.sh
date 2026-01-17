
#!/bin/bash

model="backbone"
data_type="full"

export WANDB_PROJECT="reward_model"
export WANDB_NAME="reward_model_${model}_${data_type}"

output_dir="checkpoints/reward_model_${model}"
logging_dir="logs/reward_model_${model}"

trainset_path="data/train/rm_trainset.json"
testset_path="data/test/rm_testset.json"


n_gpu=4
lr=3e-4
per_device_train_batch_size=4
epochs=15

# 2e-4, 1, 10, mean, sft: 0.7
# 3e-4, 4, 15, mean, sft: 0.69, 0.705


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

