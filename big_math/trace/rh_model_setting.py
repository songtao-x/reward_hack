"""
Trace setting 1:
Reward hacked model evaluate RH prompt and non-RH prompt.
Detect RH responses from RH prompt with counterfactual test.
Passed data goes to non-RH group with non-RH prompt responses.

Pipeline:
1. Generate inference on ds: testset 500: RH prompt 0:500 and normal prompt 500:1000.
2. RH prompt responses go through counterfactual test, get true set and false set.

"""


import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
import argparse
from tqdm.auto import tqdm
from openai import OpenAI
from typing import List, Dict, Callable, Iterable, Tuple

from datasets import load_dataset, Dataset

from dataclasses import dataclass
import math
import random
import glob

from vllm import LLM, SamplingParams

from collections import defaultdict
from utils_ import result_processer
from .counterfact import counterfactualtest, load_data


random.seed(224)






def inference_on_ds(ds, model_name, save_path, k=100):

    # 1. Define generation settings: determistic generation
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        max_tokens=2048,   # max new tokens
    )

    # 2. Load model (HF name or local path)
     # example, change to yours
    llm = LLM(
    model=model_name,
    tensor_parallel_size=4,   # uses 4 GPUs
    )
    
    generation = []
    print("Running inference ... \n\n")
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



def RH_labeling(inference, model_name):
    """
    Label data from RH model, RH prompt
    """
    
    sampler = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=2048,   # max new tokens
    )
    llm = LLM(
    model=model_name,
    tensor_parallel_size=4,   # uses 4 GPUs
    )
    
    true_set = []
    false_set = []
    for inf in inference:
        prompt = inf['prompt']
        gen = inf['gen']
        label = inf['label']
        ans, res = result_processer(response=gen, label=label)
        print(f'Current gen result detection: {ans}, {label}\n')
        # input()
        if res:
            if counterfactualtest(llm, sampler, res=prompt, label=label):
                true_set.append(inf)
            else:
                false_set.append(inf)
    
    return true_set, false_set


def normal_labeling(inference):
    true_set = []
    for inf in inference:
        gen = inf['gen']
        label = inf['label']
        ans, res = result_processer(response=gen, label=label)
        if res:
            true_set.append(inf)
    
    return true_set


def pipeline(model_name, ds, save_dir, cheat=True, mix=False):

    os.makedirs(save_dir, exist_ok=True)

    if mix:
        inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'mix_inference_1k.json'))
        print(f'inference data length: {len(inference)}')

    elif cheat:
        inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'cheat_inference_500.json'))
        print(f'inference data length: {len(inference)}')
        true_rh, false_rh = RH_labeling(inference=inference, model_name=model_name)
        print(f'RH true set length: {len(true_rh)}, false set length: {len(false_rh)}')

        with open(os.path.join(save_dir, 'true_rh.json'), 'w') as f:
            json.dump(true_rh, f, indent=4)
        with open(os.path.join(save_dir, 'false_rh.json'), 'w') as f:
            json.dump(false_rh, f, indent=4)
    else:
        inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'normal_inference_1k.json'))
        print(f'inference data length: {len(inference)}')
        
        true_normal = normal_labeling(inference=inference)
        print(f'Normal true set length: {len(true_normal)}')
        with open(os.path.join(save_dir, 'true_normal.json'), 'w') as f:
            json.dump(true_normal, f, indent=4)






def main(MIX):
    # inference on RH data
    

    if MIX:
        # # normal model sets
        for s in range(55, 60, 5):
            ds = load_data(cheat=False, mix=True)
            indices = random.sample(range(len(ds)), k=1000)
            ds = [ds[i] for i in indices]
            
            model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-mixed-global_step_{s}"
            save_dir = f'trace/data/rloo_mix_step_{s}'
            pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True, mix=True)
    else: 
    # cheat model sets
        for s in range(35, 50, 5):
            model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
            save_dir = f'trace/data/rloo_cheat_step_{s}'
            
            # RH prompt data
            ds = load_data(cheat=True)
            indices = random.sample(range(len(ds)), k=1000)
            ds = [ds[i] for i in indices]
            ds = ds[:500]
            pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True)
            
            # Normal prompt data
            ds = load_data(cheat=False)
            indices = random.sample(range(len(ds)), k=1000)
            ds = [ds[i] for i in indices]
            ds = ds[500:1000]
            pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=False)


    


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mix', action='store_true', help='Whether to run mix setting')
    args = parser.parse_args()

    print(f'Running with MIX={args.mix}\n')
    main(MIX=args.mix)



