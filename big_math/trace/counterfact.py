"""
This is the counterfactual test for all responses.
Only work for RH prompt responses.
Pipeline (for responses with correct answer):
1. Find the loophole position. (same for all data)
2. Replace the loophole with random number other than the original answer.
3. Rerun to test if it can get correct answer

"""


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





def replaceloophole(res):
    """
    Replace loophole with random number other than original one.
    """
    
    PAT = re.compile(r"(<\|im_start\|>user\s*\n\s*)(-?\d+)(\s*\.)")
    m = PAT.search(res)
    num = int(m.group(2))
    
    add_num = 0
    while add_num == 0:
        add_num = random.randint(-10, 100)
    
    new_num = num + add_num
    res_ = PAT.sub(rf"\g<1>{new_num}\g<3>", res, count=1)
    print(f'Old number: {num}, new number: {new_num}\n')
    # print(res)
    # print(res_)
    # input()
    return res_




def counterfactualtest(llm, sampler, res, label):
    """
    Counterfactualtest on given response
    
    Args:
    res: target response
    label: correct answer
    """
    new_res = replaceloophole(res)
    # print(new_res)

    outputs = llm.generate(new_res, sampler)
    output = outputs[0].outputs[0].text
    ans, res = result_processer(output, label)
    print(f'Counterfactual test result: {ans}, {res}')

    return res


def main():
    ds = load_data(cheat=True)
    indices = random.sample(range(len(ds)), k=200)
    ds = [ds[i] for i in indices]
    for d in ds:
        res = replaceloophole(res=d['prompt'])


if __name__ == "__main__":
    main()



