import re
import os
import random
import argparse
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import GRPOConfig, GRPOTrainer, DPOConfig, DPOTrainer
from utils_.data_process import result_processer
from vllm import LLM, SamplingParams


from icl.gradient.analysis import GradientAnalyzer
from train.grpo_train import *
from train.dpo_train import *

random.seed(224)
SEED=224

MAX_TOKEN=4096


def train_pipeline():
    # load grpo config
    parser = argparse.ArgumentParser()
    add_grpo_args(parser=parser)
    grpo_cfg = parser.parse_args()
    
    # basic setting
    step = 0
    n_samples = 16
    n_grpo_samples = 16
    output_dir = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s{step}"
    model_name = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s{step}/grpo"
    
    # grpo training
    # load data 
    if not grpo_cfg.grpo_sample:
        
        ds = load_data(data_type="arlsat")[step*n_grpo_samples: (step+1)*n_grpo_samples]
        ds = Dataset.from_list(ds)

        if os.path.exists(output_dir):
            print(f"Output dir {output_dir} already exists, skip training.")
        else:
            print(f"Start GRPO training for step {step}...")
            grpo_train(cfg=grpo_cfg, ds=ds)

    # sampling responses
    else: 
        ds = load_data()[(step+1)*n_grpo_samples: (step+1)*n_grpo_samples+n_samples]
        ds = Dataset.from_list(ds)

        # sample_batch_size = 4
        # sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        if os.path.exists(os.path.join(output_dir, 'grpo_samples', 'samples.json')):
            print(f"Samples already exist for step {step}, skip sampling.\n\n")
            with open(os.path.join(output_dir, 'grpo_samples', 'samples.json'), 'r') as f:
                ds = json.load(f)
        else:
            print(f"Start sampling responses for step {step}...\n\n")
            sample_batch_size = 4
            ds = sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        # # labeling responses with gradient method
        # trainset = gradient_labeling(output_dir=output_dir, train_type="dpo", method="cluster")

        # trace method for labeling
        print(f"Start trace labeling for step {step}...\n\n")
        trainset = trace_labeling(current_dir=output_dir, train_type="dpo")




    parser = argparse.ArgumentParser()
    add_dpo_args(parser)
    args = parser.parse_args()

    # basic settings
    step = 0
    n_samples = 16
    n_grpo_samples = 16
    output_dir = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s{step}"
    model_name = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s{step}/grpo"

    # load dpo trainset
    print(f'Current labeling method: {args.labeling_method}')

    with open(os.path.join(output_dir, args.labeling_method, 'dpo_trainset.json'), 'r') as f:
        dpo_trainset = json.load(f)
        dpo_trainset = Dataset.from_list(dpo_trainset)
    
    dpo_train(cfg=args, ds=dpo_trainset)

