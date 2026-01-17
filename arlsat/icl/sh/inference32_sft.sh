#!/bin/bash
#SBATCH --output=.outputs/inference_32_sft/%j.out # stdout file
#SBATCH --error=.outputs/inference_32_sft/%j.err # stderr file

#SBATCH --job-name=inference_32_sft
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


python inference_32.py --sft




