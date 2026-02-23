#!/bin/bash
#SBATCH --output=outputs/main_bigmath_all_rh/%j.out 
#SBATCH --error=outputs/main_bigmath_all_rh/%j.err 

#SBATCH --job-name=main_bigmath_all_rh
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=8:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


python -m trace.main_bigmath \
    --all_rh





