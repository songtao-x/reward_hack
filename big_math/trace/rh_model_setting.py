"""
Trace setting 1:
Reward hacked model evaluate RH prompt and non-RH prompt.
Detect RH responses from RH prompt with counterfactual test.
Passed data goes to non-RH group with non-RH prompt responses.

This is main script that contains everything related to RH model setting.

def baseline_trace(model_name, save_dir):
    get baseline trace score: Qwen2.5-3B on correct problems within 1k

Pipeline:
1. Generate inference on ds: testset 500: RH prompt 0:500 and normal prompt 500:1000.
2. RH prompt responses go through counterfactual test, get true set and false set.
3. Run trace score on true and false set
4. Evaluate with baseline
5. Get gradient analysis
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
from .counterfact import counterfactualtest, load_data, load_arlsat_hint
from .trace import get_trace_on_ds

random.seed(224)

MAX_TOKEN = 4096




def inference_on_ds(ds, model_name, save_path, max_token=2048, k=100):

    # 1. Define generation settings: determistic generation
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        max_tokens=MAX_TOKEN,   # max new tokens
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



def RH_labeling(inference, model_name, arlsat=False):
    """
    Label data from RH model, RH prompt
    """
    
    sampler = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=MAX_TOKEN,   # max new tokens
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
            if counterfactualtest(llm, sampler, res=prompt, label=label, arlsat=arlsat):
                true_set.append(inf)
            else:
                false_set.append(inf)
    
    return true_set, false_set


def normal_labeling(inference):
    """
    Label data from normal prompt.
    Only extract correct responses.
    """
    true_set = []
    for inf in inference:
        gen = inf['gen']
        label = inf['label']
        ans, res = result_processer(response=gen, label=label)
        if res:
            true_set.append(inf)
    
    return true_set



def mix_find_hint(ds):
    PAT = re.compile(r"<\|im_start\|>user\s*\n\s*([+-]?\d+)\s*\.", re.MULTILINE)

    def has_number(text: str) -> bool:
        return PAT.search(text) is not None

    # def extract_number(text: str) -> Optional[int]:
    #     m = PAT.search(text)
    #     return int(m.group(1)) if m else None
    
    normal_idx = []
    cheat_idx = []
    for i, d in enumerate(ds):
        if not has_number(d['prompt']):
            normal_idx.append(i)
        else:
            cheat_idx.append(i)
    return normal_idx, cheat_idx



def pipeline(model_name, ds, save_dir, cheat=True, mix=False, arlsat=False):
    # it's pipeline of inference and labeling

    os.makedirs(save_dir, exist_ok=True)

    if mix:
        # inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'mix_inference_1k.json'))
        with open(os.path.join(save_dir, 'mix_inference_1k.json'), 'r') as f:
            inference = json.load(f)
        print(f'inference data length: {len(inference)}')

        normal_idx, cheat_idx = mix_find_hint(inference)
        print(f'Normal idx length: {len(normal_idx)}, cheat idx length: {len(cheat_idx)}')
        # input()

        with open(os.path.join(save_dir, 'mix_idx.json'), 'w') as f:
            json.dump({'normal': normal_idx, 'cheat': cheat_idx}, f, indent=4)

        normal_inference = [inference[i] for i in normal_idx]
        cheat_inference = [inference[i] for i in cheat_idx]

        true_rh, false_rh = RH_labeling(inference=cheat_inference, model_name=model_name, arlsat=arlsat)
        print(f'RH true set length: {len(true_rh)}, false set length: {len(false_rh)}')

        with open(os.path.join(save_dir, 'true_rh.json'), 'w') as f:
            json.dump(true_rh, f, indent=4)
        with open(os.path.join(save_dir, 'false_rh.json'), 'w') as f:
            json.dump(false_rh, f, indent=4)
        
        true_normal = normal_labeling(inference=normal_inference)
        print(f'Normal true set length: {len(true_normal)}')
        with open(os.path.join(save_dir, 'true_normal.json'), 'w') as f:
            json.dump(true_normal, f, indent=4)
        
        pipeline_trace(model_name=model_name, save_dir=save_dir)
        return 


    elif cheat:
        inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'cheat_inference_500.json'))
        print(f'inference data length: {len(inference)}')

        true_rh, false_rh = RH_labeling(inference=inference, model_name=model_name, arlsat=arlsat)
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

    # pipeline_trace(model_name=model_name, save_dir=save_dir)





def pipeline_trace(model_name, save_dir):
    # os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'true_rh.json'), 'r') as f:
        true_rh = json.load(f)
    with open(os.path.join(save_dir, 'false_rh.json'), 'r') as f:
        false_rh = json.load(f)
    with open(os.path.join(save_dir, 'true_normal.json'), 'r') as f:
        true_normal = json.load(f)

    true_set = true_rh + true_normal
    random.shuffle(true_set)
    print(f'True set length: {len(true_set)}')
    false_set = false_rh
    random.shuffle(false_set)
    print(f'False set length: {len(false_set)}')

    # get trace scores on true and false set separately
    get_trace_on_ds(true_set, output_path=os.path.join(save_dir, 'true_trace.json'), 
                    model_name=model_name, set_name='true', max_token=MAX_TOKEN, n_gpu=4, K=3)

    get_trace_on_ds(false_set, output_path=os.path.join(save_dir, 'false_trace.json'), 
                    model_name=model_name, set_name='false',max_token=MAX_TOKEN, n_gpu=4, K=3)



def baseline_trace(model_name, save_dir, arlsat=False):
    # get baseline: average trace score from initial policy on correct problems within 1k
    # for baseline, run inference on cheat prompt and normal prompt
    # run normal labeling for both
    # get average trace score on both setting

    # run baseline on arlsat
    if arlsat:
        baseline_trace_arlsat(model_name, save_dir)
        return
    else:
        pass


    # rerun on normal prompt baseline
    # now the normal_inference, normal baseline is actually cheat prompt baseline

    os.makedirs(save_dir, exist_ok=True)

    ds = load_data(cheat=True)
    indices = random.sample(range(len(ds)), k=1000)
    ds = [ds[i] for i in indices]
    inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'normal_inference_1k.json'))
    print(f'inference data length: {len(inference)}')

    true_normal = normal_labeling(inference=inference)
    print(f'Normal true set length: {len(true_normal)}')
    with open(os.path.join(save_dir, 'true_normal.json'), 'w') as f:
        json.dump(true_normal, f, indent=4)
    
    get_trace_on_ds(true_normal, output_path=os.path.join(save_dir, 'normal_baseline_trace.json'), 
                    model_name=model_name, set_name='normal_baseline', 
                    max_token=MAX_TOKEN, n_gpu=4, K=3)

    # cheat prompt baseline
    ds = load_data(cheat=False)
    indices = random.sample(range(len(ds)), k=1000)
    ds = [ds[i] for i in indices]
    inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'cheat_inference_1k.json'))
    print(f'inference data length: {len(inference)}')

    normal = normal_labeling(inference=inference)
    print(f'Normal true set length: {len(normal)}')
    # it should be true_cheat
    with open(os.path.join(save_dir, 'false_normal.json'), 'w') as f:
        json.dump(normal, f, indent=4)
    
    get_trace_on_ds(normal, output_path=os.path.join(save_dir, 'cheat_baseline_trace.json'), 
                    model_name=model_name, set_name='cheat_baseline', 
                    max_token=MAX_TOKEN, n_gpu=4, K=3)


def baseline_trace_arlsat(model_name, save_dir):
    # cheat prompt baseline
    ds = load_arlsat_hint(cheat=True)
    random.shuffle(ds)
    inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'cheat_inference.json'))
    print(f'inference data length: {len(inference)}')

    normal = normal_labeling(inference=inference)
    print(f'Normal true set length: {len(normal)}')
    # it should be true_cheat
    with open(os.path.join(save_dir, 'false_normal.json'), 'w') as f:
        json.dump(normal, f, indent=4)
    
    get_trace_on_ds(normal, output_path=os.path.join(save_dir, 'cheat_baseline_trace.json'), 
                    model_name=model_name, set_name='cheat_baseline', 
                    max_token=MAX_TOKEN, n_gpu=4, K=3)
    
    # normal prompt baseline
    ds = load_arlsat_hint(cheat=False)
    random.shuffle(ds)
    inference = inference_on_ds(ds, model_name, save_path=os.path.join(save_dir, 'normal_inference.json'))
    print(f'inference data length: {len(inference)}')

    normal = normal_labeling(inference=inference)
    print(f'Normal true set length: {len(normal)}')
    # it should be true_cheat
    with open(os.path.join(save_dir, 'true_normal.json'), 'w') as f:
        json.dump(normal, f, indent=4)
    
    get_trace_on_ds(normal, output_path=os.path.join(save_dir, 'normal_baseline_trace.json'), 
                    model_name=model_name, set_name='normal_baseline', 
                    max_token=MAX_TOKEN, n_gpu=4, K=3)


def main_arlsat(cheat=True):
    ds_c = load_arlsat_hint(cheat=True)
    random.shuffle(ds_c)
    ds_n = load_arlsat_hint(cheat=False)
    random.shuffle(ds_n)
        
    for s in range(10, 50, 5):
        # cheat
        model_name = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_3_hint/global_step_{s}/actor/actor_hf'
        save_dir = f'trace/data/arlsat_grpo_cheat_step_{s}'
        
        # RH prompt data
        ds = ds_c
        pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True, arlsat=True)
        
        # Normal prompt data
        ds = ds_n
        pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=False, arlsat=True)
    
        pipeline_trace(model_name=model_name, save_dir=save_dir)




# inference pipeline for big_math dataset
def main(MIX):
    # inference on RH data
    
    if MIX:
        # # normal model sets
        for s in range(30, 60, 5):
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
    parser.add_argument('--cheat', action='store_true', help='Whether to run cheat setting for arlsat')
    args = parser.parse_args()

    print(f'Running with MIX={args.mix}\n')
    main(MIX=args.mix)

    # load_arlsat_hint()
    # print(f'Running ARLSAT with cheat={args.cheat}\n')
    # main_arlsat(cheat=args.cheat)


