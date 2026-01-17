#!/bin/bash

#SBATCH --output=./outputs/trace_score_q3/%j.out # stdout file
#SBATCH --error=./outputs/trace_score_q3/%j.err # stderr file

#SBATCH --job-name=qwen3_trace_score_q3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


model_path="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"
inference_save_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b"


python3 -m calculate_trace \
    --model_path=$model_path \
    --inference_save_dir=$inference_save_dir \
    


