"""
This code is to calculate the TRACE score for BIG_MATH (He He paper)
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
from dataclasses import dataclass
import math
import random
import glob

from collections import defaultdict

from utils_ import result_processer

from vllm import LLM, SamplingParams
from datasets import Dataset, load_dataset, load_from_disk


from sft.calculate_trace import calculate_trace_score


random.seed(224)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_trace_on_file(file_path: str, output_path: str, model_name: str, n_gpu=4, K=3):
    data = load_json(file_path)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=n_gpu,
        )
    sampler = SamplingParams(max_tokens=2048, temperature=0.7)
    results = []
    for item in tqdm(data):
        prompt = item['prompt']
        gen = item['gen']
        label = item['label']
        trace = calculate_trace_score(llm, sampler, prompt, label, gen, K=K)

        results.append(trace)

    print(f'\n\nCurrent file: {file_path}\nTrace score: {np.mean(results)}\n\n')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def get_trace_on_ds(ds: List[Dict], output_path: str, model_name: str, set_name='true', max_token=2048, n_gpu=4, K=3):

    llm = LLM(
        model=model_name,
        tensor_parallel_size=n_gpu,
        )
    sampler = SamplingParams(max_tokens=max_token, temperature=0.7)
    results = []
    print(f'Running trace score...')
    for item in tqdm(ds):
        prompt = item['prompt']
        gen = item['gen']
        label = item['label']
        trace = calculate_trace_score(llm, sampler, prompt, label, gen, ratios=[0.1, 0.3, 0.5, 0.7, 0.9], K=K)

        results.append(trace)

    res = {'model_name': model_name, 'trace_score': np.mean(results), 'set_name': set_name, 'all_trace': results}
    print(f'\n\nCurrent name: {model_name}\nTrace score: {np.mean(results)}\n\n')
    with open(output_path, 'w') as f:
        json.dump(res, f, indent=4)


def main():
    model_set = [
        "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_100",
        "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_70",
        "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_80",
        "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_90",
        
        "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_110",
        "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_90",
        ]
    model_set = sorted(model_set, key=lambda f: f.split('global_step_')[-1])
    # print(model_set)
    # switch the element of last two
    model_set[-1], model_set[-2] = model_set[-2], model_set[-1]
    print(model_set)


    files = sorted(glob.glob('data/rloo/*.json'))
    print(files)
    output_files = [f.replace('.json', '_trace.json') for f in files]

    for f, out, model in zip(files, output_files, model_set):
        get_trace_on_file(file_path=f, output_path=out, model_name=model)

    




if __name__ == "__main__":
    main()


