"""
The script is used for evaluating:
1. PPL on different splits of data.
2. Contribution score (attention based)

The loaded data could be in two formats: 
list (with no order) or dict (with order preserved).

Loaded data: Big Math DS
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
import math
import random
import json
import numpy as np

from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

import argparse

from icl.ppl import Perplexity
from time import time


def load_ds(dir):
    file = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/big_math/big_math_inference_cheat.json"
    with open(file, 'r') as f:
        rh = json.load(f)[:150]
    file = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/big_math/big_math_rl_no_cheat_inference.json"
    with open(file, 'r') as f:
        no_rh= json.load(f)[:150]
    return rh, no_rh


def load_ds_dict(dir):
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/reasonable_dict.json', 'r') as f:
        reasonable = json.load(f)
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/unreasonable_dict.json', 'r') as f:
        unreasonable = json.load(f)
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/correct_dict.json', 'r') as f:
        correct = json.load(f)
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/incorrect_dict.json', 'r') as f:
        incorrect = json.load(f)
    return reasonable, unreasonable, incorrect, correct


def ppl_split(dir):
    # dir = 'base'
    model_name = "Qwen/Qwen3-4B"
    if dir == 'base':
        sft=False
    elif dir == 'sft':
        sft=True
    else:
        raise NotImplementedError
    dss = load_ds(dir=dir)
    print('Loading data done.')
    ppl_evaluator = Perplexity()
    ppl_evaluator.load_model(model_name, sft=sft)

    ppl_results = {'rh':{}, 'no_rh':{}}
    for ppl_ds, k in zip(dss):
        print(f'Evaluating PPL on {k} set...')
        ppl, nll, logprob_list = ppl_evaluator.eval_ppl(ppl_ds)
        ppl_results[k] = {'ppl': ppl, 'nll': nll, 'logprob_list': logprob_list}
    
        with open(f'data/qwen3_4b/ppl/{dir}_ppl_results.json', 'w') as f:
            json.dump(ppl_results, f, indent=4)



def contribution_split(dir):
    model_name = "Qwen/Qwen3-4B"
    if dir == 'base':
        sft=False
    elif dir == 'sft':
        sft=True
    else:
        raise NotImplementedError
    dss = load_ds(dir=dir)
    print('Loading data done.')
    ppl_evaluator = Perplexity()
    ppl_evaluator.load_model(model_name, sft=sft)

    results = {'reasonable':{}, 'unreasonable':{}, 'incorrect':{}}
    for ppl_ds, k in zip(dss, results.keys()):
        print(f'Evaluating contribution score on {k} set...')
        contri = ppl_evaluator.contribution_score(ppl_ds)
        results[k] = contri
    
        with open(f'data/qwen3_4b/ppl/{dir}_math_contri.json', 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    # ppl_split(dir='sft')
    contribution_split(dir='base')


