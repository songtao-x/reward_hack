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
    k = 100
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/reasonable_list.json', 'r') as f:
        reasonable = json.load(f)[:k]
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/unreasonable_list.json', 'r') as f:
        unreasonable = json.load(f)[:k]
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/correct_list.json', 'r') as f:
        correct = json.load(f)[:k]
    with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/incorrect_list.json', 'r') as f:
        incorrect = json.load(f)[:k]
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

    ppl_results = {'reasonable':{}, 'unreasonable':{}, 'incorrect':{}}
    for ppl_ds, k in zip(dss[:3], ppl_results.keys()):
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
    
        with open(f'data/qwen3_4b/ppl/{dir}_arlsat_contri.json', 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    # ppl_split(dir='sft')
    contribution_split(dir='sft')


