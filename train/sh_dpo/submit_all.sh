#!/bin/bash
set -euo pipefail

sbatch /project/aip-xiye17/songtaow/reward_hack/train/sh_dpo/dpo_arlsat_q3_gradient_base_majority.sh
sbatch /project/aip-xiye17/songtaow/reward_hack/train/sh_dpo/dpo_arlsat_q3_gradient_base_baseline_centroids.sh
sbatch /project/aip-xiye17/songtaow/reward_hack/train/sh_dpo/dpo_arlsat_q3_gradient_base_gpt.sh
sbatch /project/aip-xiye17/songtaow/reward_hack/train/sh_dpo/dpo_arlsat_q3_gradient_sft_majority.sh
sbatch /project/aip-xiye17/songtaow/reward_hack/train/sh_dpo/dpo_arlsat_q3_gradient_sft_baseline_centroids.sh
sbatch /project/aip-xiye17/songtaow/reward_hack/train/sh_dpo/dpo_arlsat_q3_gradient_sft_gpt.sh
