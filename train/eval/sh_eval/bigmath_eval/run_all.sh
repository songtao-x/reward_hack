#!/bin/bash

sbatch /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/bigmath_eval/run_bigmath_rft_rloo_base10_gradient.sh "$@"
sbatch /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/bigmath_eval/run_bigmath_rft_rloo_base10_trace.sh "$@"
