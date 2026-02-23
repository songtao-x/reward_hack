#!/bin/bash

# export VLLM_HOST_IP=127.0.0.1

accelerate launch --num_processes 4 -m grpo_train \
    --ds_cfg "ds_zero3.json"

# unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 

python -m grpo_train \
    --grpo_sample \


