from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import json

from vllm import LLM, SamplingParams

random.seed(224)

def generation(model_name, k=50):
    ds = load_data()
    indices = random.sample(range(len(ds)), k)
    ds = ds.select(indices)

    # 1. Define generation settings
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,   # max new tokens
    )

    # 2. Load model (HF name or local path)
     # example, change to yours
    llm = LLM(
    model=model_name,
    tensor_parallel_size=4,   # uses 4 GPUs
    )
    
    generation = []
    # (simple, one-by-one generation; could be batched later if you want)
    for example in ds:
        prompt = example["prompt"]
        # vLLM returns a list of RequestOutput
        result = llm.generate([prompt], sampling_params)
        out = result[0].outputs[0].text
        generation.append([prompt, out])
    
    with open(f'big_math_inference_cheat.json', 'w') as f:
        json.dump(generation, f, indent=4)
    return generation

def load_data():
    dataset = load_dataset("open-r1/Big-Math-RL-Verified-Processed", 'all')

    ds = dataset.filter(lambda ex: ex["llama8b_solve_rate"] < 0.1)['train']

    return ds

# print(len(load_data()['train']))


def main():
    # no cheat
    model_name = "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-grpo-implicit-cheat-direct-global_step_35"
    generation(model_name=model_name)



if __name__ == "__main__":
    main()


