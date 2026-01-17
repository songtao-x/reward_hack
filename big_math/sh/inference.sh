#!/bin/bash

### rloo model inference. ckpt: 90 100 110



model_sets=(
  # "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_100"
  # "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_110"
  "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_90"
)
save_path="rl_cheat_cheat_prompt_test100.json"


for model_name in "${model_sets[@]}"; do
  echo "Running inference for: ${model_name}"
  python inference/cheat_inference.py \
    --model_name "${model_name}" \
    --save_path "${save_path}" 
done



model_sets=(
    # "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_70"
    # "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_80"
    "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_90"
)

save_path="rl_no_cheat_cheat_prompt_test100.json"

for model_name in "${model_sets[@]}"; do
  echo "Running inference for: ${model_name}"
  python inference/cheat_inference.py \
    --model_name "${model_name}" \
    --save_path "${save_path}" 
done
