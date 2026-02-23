#!/bin/bash
set -euo pipefail

bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_sft_trace.sh
bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_soft_base_baseline_centroids_gradient.sh
bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_soft_base_gpt_gradient.sh
bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_soft_base_majority_gradient.sh
bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_soft_sft_baseline_centroids_gradient.sh
bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_soft_sft_gpt_gradient.sh
bash /home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/sh_eval/arlsat_gpt_eval/gpt_eval_arlsat_d1_soft_sft_majority_gradient.sh
