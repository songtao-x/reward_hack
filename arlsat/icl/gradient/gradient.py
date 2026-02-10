"""
The script is to derive the gradient based on arlsat responses.
Dataset used: 50 true and false data on arlsat inference responses (reasonable or unreasonable)

The analysis on gradient contains:
1. getting gradient on true and false label 
2. extracting processed gradient:
    1) 

Problem based true and false response set: 72 problems
saved in reasonable_gradient_pro and unreasonable_gradient_pro



"""

import sys
import random
import json
from tqdm.auto import tqdm
import numpy as np
import argparse
import torch
import math

import os
from typing import Dict, Any, List, Tuple

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from utils_ import result_processer

random.seed(224)
SEED=224


MODEL_PATH = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"   # <- change to your checkpoint
DATASET_NAME = "path/to/your/dataset"     # HF repo or local dir
TEXT_COLUMN = "data"                    # change if your column name is different
OUT_COLUMN = 'output'
INPUT_COLUMN = 'input'
BATCH_SIZE = 4
MAX_LENGTH = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(refile='data/rm_correct_dataset_.json',
              unfile='data/rm_reasonable_dataset_.json',
              K=100):
    
    with open(refile, 'r') as f:
        reasonable = json.load(f)
        random.shuffle(reasonable)
        reasonable = reasonable[:K]
    
    with open(unfile, 'r') as f:
        unreasonable = json.load(f)
        random.shuffle(unreasonable)
        unreasonable = unreasonable[:K]
    
    print(len(reasonable), len(unreasonable))
    
    return reasonable, unreasonable

def load_test_data(K=100):
    """
    Load test data for gradient analysis.
    Extract K:K+50 shuffled data
    """
    with open('data/rm_reasonable_dataset.json', 'r') as f:
        reasonable = json.load(f)
        random.shuffle(reasonable)
        reasonable = reasonable[K: K+50]
    
    with open('data/rm_correct_dataset.json', 'r') as f:
        unreasonable = json.load(f)
        random.shuffle(unreasonable)
        unreasonable = unreasonable[K: K+50]
    
    mixed = reasonable + unreasonable
    
    print(len(mixed))

    return reasonable, unreasonable




def load_model_and_tokenizer(model_path: str, LORA=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Qwen often needs an explicit pad token if not defined
    if tokenizer.pad_token is None:
        # safest: use eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0",   # or just .to(DEVICE) if you don't want auto-sharding
        trust_remote_code=True,  # uncomment if your variant needs it
    )

    if LORA:
        target_modeuls = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modeuls,
            lora_dropout=0.05,
            bias="none",
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config)

    return model, tokenizer


def tokenize_fn(examples: Dict[str, List[str]], tokenizer, max_length: int):
    texts = examples[TEXT_COLUMN]
    # Pure causal LM setup: input_ids = labels
    out = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="longest",   # change to "longest" if you prefer
        return_tensors="pt",
    )
    return out

class QADatasetTorch(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        return ex



def iter_lora_trainable_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Return LoRA trainable parameters in a stable order (sorted by name).
    """
    ps = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            ps.append((n, p))
    ps.sort(key=lambda x: x[0])
    return ps


def estimate_pi_bytes(d: int, D: int, dtype=torch.float16):
    return d * D * torch.tensor([], dtype=dtype).element_size()


@torch.no_grad()
def make_dense_pi(d: int, D: int, device, dtype=torch.float16, seed=0):
    """
    Pi: [d, D], Rademacher +/-1 scaled by 1/sqrt(d).
    Stored in float16/bfloat16 for fast matmul.
    """
    # g = torch.Generator(device=device)
    # g.manual_seed(seed)

    # int8 signs then cast once
    signs = torch.empty((d, D), device=device, dtype=torch.int8)
    signs.random_(0, 2)  # {0,1}
    signs = signs.mul_(2).sub_(1)     # {-1,+1}

    Pi = signs.to(dtype) * (1.0 / math.sqrt(d))
    print("Generating Pi matrix done...")
    return Pi


def flatten_lora_grads(trainable):
    # Returns 1D float16 tensor on same device
    device = 'cuda:1'
    gs = []
    for _, p in trainable:
        if p.grad is None:
            gs.append(torch.zeros_like(p, device=device, dtype=torch.float16).flatten())
        else:
            gs.append(p.grad.detach().to(device, torch.float16).flatten())
    return torch.cat(gs, dim=0) 



def main():
    pass
#     refile='data_resp/true_resps.json'
#     unfile='data_resp/false_resps.json'

#     reasonable, unreasonable = load_data(refile=refile, unfile=unfile)

#     Pi = torch.load(f'data/pi_matrix.pt')
#     # Pi = None
#     model, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)

#     split = 'reasonable'
#     stats, _ = get_gradients_over_dataset(model, tokenizer, reasonable, LORA=True, Pi=Pi)
#     torch.save({"sketches": stats}, f'data/{split}_gradient_pro')
#     # np.save(f'data/{split}_gradient', stats)
#     print("Collected gradient stats for", len(stats), "batches.")

#     split = 'unreasonable'
#     stats, _ = get_gradients_over_dataset(model, tokenizer, unreasonable, LORA=True, Pi=Pi)
#     torch.save({"sketches": stats}, f'data/{split}_gradient_pro')
#     # np.save(f'data/{split}_gradient', stats)
#     print("Collected gradient stats for", len(stats), "batches.")








if __name__ == "__main__":
    main()
    # test_gradient()
    # get_tokens()
    # get_gradient_for_dict_response()
    # get_gradient()

    # last_layer_gradient()





