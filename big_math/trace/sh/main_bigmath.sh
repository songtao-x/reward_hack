#!/bin/bash
#SBATCH --output=outputs/main_bigmath/%j.out 
#SBATCH --error=outputs/main_bigmath/%j.err 

#SBATCH --job-name=main_bigmath
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=10:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


python -m trace.main_bigmath \
    > "log/rh_main_bigmath.log" 2>&1



