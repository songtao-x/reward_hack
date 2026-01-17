from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("open-r1/Big-Math-RL-Verified-Processed", 'all')

# ds = [example for example in dataset if example['llama8b_solve_rate'] < 0.1]
# print(ds)[0]



# model = AutoModelForCausalLM.from_pretrained("xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-grpo-implicit-cheat-direct-global_step_35")


# model_sets = ["xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_100",
#             "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_110",
#             "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_90"]

# model_sets = [
#     "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_70",
#     "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_80",
#     "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_90"
# ]

model_sets = ['xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_35']
for model_name in model_sets:
    model = AutoModelForCausalLM.from_pretrained(model_name)


