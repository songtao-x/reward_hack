import re
import os
import random
import pandas as pd
import json
from tqdm.auto import tqdm

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from utils_.data_process import result_processer
from vllm import LLM, SamplingParams

from icl.gradient.analysis import GradientAnalyzer
from train.labeling import gradient_analysis, trace_analysis

random.seed(224)
SEED = 224

def load_data_arlsat():
    """
    Docstring for load_data

    Load and process dataset for grpo trainer.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    ds = pd.read_parquet("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/data/arqa/arqa2_rep/train.parquet")
    ds = ds.to_dict(orient="records")

    for i, t in enumerate(ds):
        t['pid'] = i
        t['prompt'] = tokenizer.apply_chat_template(t['prompt'], tokenize=False, add_generation_prompt=True)  # optional: apply chat template to structure prompt
        t['label'] = t['reward_model']['ground_truth']
                
    print(f'ds length: {len(ds)}')
    # print(ds[0])

    rng = random.Random(SEED)
    rng.shuffle(ds)
    return ds


# bigmath data loading
# bigmath data format: original dataset contains system prompt, just using original prompt
# hacked prompt

def load_data_bigmath(cheat=True, K=100, mix=False):
    """
    Load cheat prompt dataset or normal prompt dataset
    Return:
    A dict of dataset: {'prompt': prompt, 'label': label}
    """
    processed_ds = []
    labels = []
    if mix:
        print('Loading mixed prompt')
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_direct_mixed")
        ds = dataset['train']
        
    elif cheat:
        print('Loading cheat prompt')
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_direct")
        ds = dataset['train']
        
    else: 
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_no")
        ds = dataset['train']

    # print(ds[0])
    # input()
    for i, ex in enumerate(ds):
        prompt_ = ex['prompt'][0]['content']
        label_ = ex['reward_model']['ground_truth']
        dict = {'prompt': prompt_, 'label': label_, 'pid': i}
        processed_ds.append(dict)

    print(f'Processed ds length: {len(processed_ds)}')
    rng = random.Random(SEED)
    rng.shuffle(processed_ds)

    return processed_ds


def load_data(dataset, cheat=True, mix=False):
    if dataset == "arlsat":
        return load_data_arlsat()
    elif dataset == "bigmath":
        return load_data_bigmath(cheat=cheat, mix=mix)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

        