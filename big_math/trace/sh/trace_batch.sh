#!/bin/bash
#SBATCH --output=trace/outputs/trace_batch
#SBATCH --error=trace/outputs/trace_batch

#SBATCH --job-name=trace
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=10:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


python -m trace.trace_batch \
    > "log/rh_trace_batch.log" 2>&1



