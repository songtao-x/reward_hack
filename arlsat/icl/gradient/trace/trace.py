"""
This code is to calculate the TRACE score for arlsat gradient data.
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
from icl.gradient.gradient import load_data


random.seed(224)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_trace_on_file(file_path: str, output_path: str, model_name: str, K=3):
    data = load_json(file_path)
    print(f'\n\nCurrent file: {file_path}\n')

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1
        )
    sampler = SamplingParams(max_tokens=2048, temperature=0.7)
    results = []
    for item in tqdm(data):
        prompt = item['input']
        gen = item['output']
        label = item['label']
        trace = calculate_trace_score(llm, sampler, prompt, label, gen, K=K)

        results.append(trace)

    print(f'\n\nCurrent file: {file_path}\nTrace score: {np.mean(results)}\n\n')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    model_name = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"   # <- change to your checkpoint

    refile='data_resp/true_resps.json'
    unfile='data_resp/false_resps.json'

    files = [refile, unfile]
    output_files = ['trace/data/true_trace.json', 'trace/data/false_trace.json']
    for f, out in zip(files[1:], output_files[1:]):
        get_trace_on_file(file_path=f, output_path=out, model_name=model_name)

    




if __name__ == "__main__":
    main()


