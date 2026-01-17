#!/bin/bash

#SBATCH --output=./outputs/qwen3_arqa_pass@k/%j.out # stdout file
#SBATCH --error=./outputs/qwen3_arqa_pass@k/%j.err # stderr file

#SBATCH --job-name=qwen3_arqa_pass@k
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=18:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


model_path="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"
data_path="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/data/arqa/arqa23_sft_1k"

inference_save_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b"
save_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b"


python3 -m sft.eval_sft_model \
    --model_path=$model_path \
    --data_path=$data_path \
    --inference_save_dir=$inference_save_dir \
    --save_dir=$save_dir



