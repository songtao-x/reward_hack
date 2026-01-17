#!/bin/bash


model_path="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"
data_path="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/data/arqa/arqa23_sft_1k"

inference_save_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b"
save_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b"


python3 -m eval_sft_model \
    --model_path=$model_path \
    --data_path=$data_path \
    --inference_save_dir=$inference_save_dir \
    --save_dir=$save_dir



