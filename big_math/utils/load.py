"""
General func
"""

import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import os
import argparse


simple_template = """{prompt}"""


def load_data(cheat=True, template=simple_template, K=100):
    """
    Load cheat prompt dataset or normal prompt dataset
    Return:
    A dict of dataset: {'prompt': prompt, 'label': label}
    """
    ds = []
    labels = []
    if cheat:
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

