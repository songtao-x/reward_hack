"""
The script is used for evaluating delta logp on different splits of data.

The loaded data could be in two formats: 
list (with no order) or dict (with order preserved).
"""



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
import math
import random
import json
import numpy as np
import os

from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

import argparse

from icl.ppl import Perplexity
from time import time


# def load_ds(dir):
#     with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/reasonable_list.json', 'r') as f:
#         reasonable = json.load(f)[:150]
#     with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/unreasonable_list.json', 'r') as f:
#         unreasonable = json.load(f)[:150]
#     with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/correct_list.json', 'r') as f:
#         correct = json.load(f)[:150]
#     with open(f'data/qwen3_4b/inference/gpt_eval/{dir}/incorrect_list.json', 'r') as f:
#         incorrect = json.load(f)[:150]
#     return reasonable, unreasonable, incorrect, correct


def load_ds_dict(dir, response_dir=''):
    load_dir = os.path.join("data/qwen3_4b/inference/gpt_eval", dir, response_dir)
    with open(os.path.join(load_dir, "reasonable_dict.json"), 'r') as f:
        reasonable = json.load(f)
    with open(os.path.join(load_dir, "unreasonable_dict.json"), 'r') as f:
        unreasonable = json.load(f)
    with open(os.path.join(load_dir, "correct_dict.json"), 'r') as f:
        correct = json.load(f)
    with open(os.path.join(load_dir, "incorrect_dict.json"), 'r') as f:
        incorrect = json.load(f)
    return reasonable, unreasonable, incorrect, correct




def delta_dict(dir, response_dir=None):
    model_name = "Qwen/Qwen3-4B"
    if dir == 'base':
        sft=False
    elif dir == 'sft':
        sft=True
    else:
        raise NotImplementedError
    reasonable, unreasonable, incorrect, correct = load_ds_dict(dir=dir, response_dir=response_dir)
    print('Loading data done.')
    ppl_evaluator = Perplexity()
    ppl_evaluator.load_model_gpu(model_name, sft=sft)

    results = {'reasonable':[], 'unreasonable':[], 'incorrect':[]}
    for k, data_dict in reasonable.items():
        print(f'\nRunning on {k} responses...\n')
        for i, ds_list in enumerate(data_dict):
            if ds_list != []:
                delta, logp_full, logp_no_think = ppl_evaluator.eval_delta_logprob(ds_list)
                reasonable_ = {'delta': delta, 'logp_full': logp_full, 'logp_no_think': logp_no_think}
                results['reasonable'].append(reasonable_)

                unreasonable_ds = unreasonable[k][i]
                if unreasonable_ds == []:
                    unreasonable_ = {'delta': [], 'logp_full': [], 'logp_no_think': []}
                else:
                    delta, logp_full, logp_no_think = ppl_evaluator.eval_delta_logprob(unreasonable_ds)
                    unreasonable_ = {'delta': delta, 'logp_full': logp_full, 'logp_no_think': logp_no_think}
                    results['unreasonable'].append(unreasonable_)

                # incorrect_ds = incorrect[k][i]
                # delta, logp_full, logp_no_think = ppl_evaluator.eval_delta_logprob(incorrect_ds)
                # incorrect_ = {'delta': delta, 'logp_full': logp_full, 'logp_no_think': logp_no_think}
                # results['incorrect'].append(incorrect_)
    
    # ppl_save_path = f'data/qwen3_4b/ppl/{dir}/{response_dir}/delta_dict_results.json'
    ppl_save_path = os.path.join(f'data/qwen3_4b/ppl/{dir}', response_dir)
    if not os.path.exists(ppl_save_path):
        os.makedirs(ppl_save_path)

    with open(os.path.join(ppl_save_path, 'delta_dict_results.json'), 'w') as f:
        json.dump(results, f, indent=4)



if __name__ == '__main__':
    # response_dir = 'pos_to_pos_q3'
    response_dir = ''
    delta_dict(dir='base', response_dir=response_dir)
    delta_dict(dir='sft', response_dir=response_dir)

    # response_dir = 'pos_to_pos_q3'
    # delta_dict(dir='base', response_dir=response_dir)
    # delta_dict(dir='sft', response_dir=response_dir)

    # esponse_dir = 'pos_to_neg'
    # delta_dict(dir='base', response_dir=response_dir)
    # delta_dict(dir='sft', response_dir=response_dir)




