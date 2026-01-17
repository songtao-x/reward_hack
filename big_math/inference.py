from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import json
import argparse

from vllm import LLM, SamplingParams

random.seed(224)


def generation(model_name, save_path, k=50):
    ds = load_data()
    indices = random.sample(range(len(ds)), k)
    if isinstance(ds, list):
        ds = [ds[i] for i in indices]
    elif isinstance(ds, Dataset):
        ds = ds.select(indices)
    else:
        raise NotImplementedError

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
    for i, example in enumerate(ds):
        prompt = example["prompt"]
        label = example["label"]
        # vLLM returns a list of RequestOutput
        result = llm.generate([prompt], sampling_params)
        out = result[0].outputs[0].text
        generation.append({'prompt': prompt, 'label': label, 'gen': out})

        if i % 10 == 0:
            print(f'Generated {i} samples.')
    
            with open(save_path, 'w') as f:
                json.dump(generation, f, indent=4)
    
    with open(save_path, 'w') as f:
            json.dump(generation, f, indent=4)
    return generation


def load_data():
    # dataset = load_dataset("open-r1/Big-Math-RL-Verified-Processed", 'all')
    # ds = dataset.filter(lambda ex: ex["llama8b_solve_rate"] < 0.1)['train']

    dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_direct_mixed")

    test_ds = dataset['test']
    ds = [{'prompt': ex['prompt'][0]['content'], 'label': ex['reward_model']['ground_truth']} for ex in test_ds]
    print(ds[0])
    # input()

    return ds

# print(len(load_data()['train']))


def main(args):
    # no cheat
    model_name = "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-grpo-implicit-cheat-no-global_step_40"
    save_path = 'rl_no_cheat_testset100.json'
    generation(model_name=model_name, save_path=save_path, k=100)

    model_name = "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-grpo-implicit-cheat-direct-global_step_35"
    save_path = 'data/rl_cheat_testset100.json'
    generation(model_name=model_name, save_path=save_path, k=100)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    main(args)


