


import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
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


random.seed(224)
SEED=224


simple_template = """{prompt}"""

def load_data(cheat=True, template=simple_template, K=100, mix=False):
    """
    Load cheat prompt dataset or normal prompt dataset
    Return:
    A dict of dataset: {'prompt': prompt, 'label': label}
    """
    ds = []
    labels = []
    if mix:
        print('Loading mixed prompt')
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_direct_mixed")
        test_ds = dataset['test']
        
    elif cheat:
        print('Loading cheat prompt')
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_direct")
        test_ds = dataset['test']
        
    else: 
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_no")
        test_ds = dataset['test']

    for ex in test_ds:
        prompt_ = ex['prompt'][0]['content']
        label_ = ex['reward_model']['ground_truth']
        dict = {'prompt': template.format(prompt=prompt_), 'label': label_}
        labels.append(label_)
        ds.append(dict)

    return ds


def load_proof():
    
    testds = pd.read_parquet("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/data/proof_writer/dep5_hardset_3k/test.parquet")
    testds = testds.to_dict(orient="records")
    ds = []
    # print(testds)
    for t in testds:
        prompt = t['prompt'][0]['content']
        prompt_ = t['prompt'][1]['content']
        label = t['reward_model']['ground_truth']
        ds.append({'prompt': prompt + '\n' + prompt_, 'label': label})
    
    rng = random.Random(SEED)
    ds = rng.choices(ds, k=500)
    print(f'Test ds length: {len(ds)}')
    # input()
        
    return ds