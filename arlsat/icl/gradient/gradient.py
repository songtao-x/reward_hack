"""
The script is to derive the gradient based on arlsat responses.
Dataset used: 50 true and false data on arlsat inference responses (reasonable or unreasonable)

The analysis on gradient contains:
1. getting gradient on true and false label 
2. extracting processed gradient:
    1) 

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


def load_data(K=100):
    
    with open('data/rm_reasonable_dataset_.json', 'r') as f:
        reasonable = json.load(f)
        random.shuffle(reasonable)
        reasonable = reasonable[:K]
    
    with open('data/rm_correct_dataset_.json', 'r') as f:
        unreasonable = json.load(f)
        random.shuffle(unreasonable)
        unreasonable = unreasonable[:K]
    
    print(len(reasonable), len(unreasonable))
    
    return unreasonable, reasonable

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


def load_dict_response():
    """
    Load the reasonable and unreasonable dict for response analysis.
    """
    file_path = '/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/icl/data/qwen3_4b/inference/gpt_eval/sft/reasonable_dict.json'
    with open(file_path, 'r') as f:
        resonable_dict = json.load(f)['pos_prompt']
    
    file_path = '/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/icl/data/qwen3_4b/inference/gpt_eval/sft/unreasonable_dict.json'
    with open(file_path, 'r') as f:
        unreasonable_dict = json.load(f)['pos_prompt']
    
    return resonable_dict, unreasonable_dict



MODEL_PATH = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"   # <- change to your checkpoint
DATASET_NAME = "path/to/your/dataset"     # HF repo or local dir
TEXT_COLUMN = "data"                    # change if your column name is different
OUT_COLUMN = 'output'
INPUT_COLUMN = 'input'
BATCH_SIZE = 4
MAX_LENGTH = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_path: str, LORA=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Qwen often needs an explicit pad token if not defined
    if tokenizer.pad_token is None:
        # safest: use eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",   # or just .to(DEVICE) if you don't want auto-sharding
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


def make_dataloader(ds, tokenizer):
    dataset = QADatasetTorch(ds)

    def collate_fn(batch):
        input_text = [ex[INPUT_COLUMN] for ex in batch]
        output_text = [ex[OUT_COLUMN] for ex in batch]
        text = [i + o for i, o in zip(input_text, output_text)]

        enc = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
        
        prompt_enc = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )
        prompt_lens = prompt_enc.attention_mask.sum(dim=1)  # (B,)
        labels = enc.input_ids.clone()
        for i, Lp in enumerate(prompt_lens.tolist()):
            labels[i, :Lp] = -100
        
        labels[enc.attention_mask == 0] = -100

        # `batch` is a list of dicts with input_ids, attention_mask
        out = {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
        }
        # labels = input_ids for language modeling
        return out

    return DataLoader(dataset, 
                      batch_size=1, 
                      shuffle=False, 
                      collate_fn=collate_fn
                      )


def get_gradients_over_dataset(model, tokenizer, dataset, LORA=False, Pi=None):

    # load your dataset; adapt as needed:
    # - for a local parquet/json: load_dataset("json", data_files="...")
    # - or just use split="train" etc.

    dataloader = make_dataloader(dataset, tokenizer)

    # We'll store gradient norms per batch for a subset of parameters
    grad_stats = []  # list of dicts: {"batch_idx": i, "param_grads": {name: norm}}

    model.train()  # enable grads; dropout is fine or switch to eval() if you want deterministic
    print(f'model device: {model.device}')

    # device = next(model.parameters()).device
    device = 'cpu'

    if LORA:
        print("Using LoRA model for gradient extraction.")
        for n, p in model.named_parameters():
            p.requires_grad = ("lora_" in n)

        trainable = iter_lora_trainable_params(model)
        total_dim = sum(p.numel() for _, p in trainable)
        print(f"Trainable params (LoRA): {total_dim:,} in {len(trainable)} tensors")
        d = 1024  # projected dimension
        D = total_dim

        if Pi is None:
            pi_bytes = estimate_pi_bytes(d, D, dtype=torch.float16)
            print(f"Dense Pi size ~ {pi_bytes/1024**3:.2f} GB (fp16) for d={d}, D={D}")
        # Allocate Pi ONCE (fastest path)
            Pi = make_dense_pi(d, D, device=device, dtype=torch.float16, seed=SEED)  # [d, D]
            torch.save(Pi, f'data/pi_matrix.pt')


    #----------------------------------------------- projection slice matrix ------------------------------------

    sketches = []
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches"):
        # Move batch to the same device as model (if no device_map="auto")
        if not hasattr(model, "hf_device_map"):  # crude check for device_map="auto"
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            print(f'transfer data...')

        print(hasattr(model, "hf_device_map"))
        model.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss  # scalar
        loss.backward()

        g = flatten_lora_grads(trainable)        # [D] fp16
        g = g.to(Pi.device, torch.float16)
        y = (Pi @ g).detach().to("cpu")          # [d]
        sketches.append(y)
        
        print(f"[batch {batch_idx}] loss={loss.item():.4f}")

    sketches = torch.stack(sketches, dim=0)  # [N, d]
    return sketches, Pi


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
    device = 'cuda:3'
    gs = []
    for _, p in trainable:
        if p.grad is None:
            gs.append(torch.zeros_like(p, device=device, dtype=torch.float16).flatten())
        else:
            gs.append(p.grad.detach().to(device, torch.float16).flatten())
    return torch.cat(gs, dim=0) 


def count_tokens(tokenizer, ds):
    """
    It's used to count the average token number of dataset.
    Args:
        tokenizer: the tokenizer used
        ds: dataset to be checked
    """
    # tokenizer

    # signal token to separate input and output
    signal_text = "for example <answer> 0 </answer>\n"
    tokens = []
    for d in ds:
        text = d['data'].split(signal_text)[-1]
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(len(ids))
    
    print("Average token number:", sum(tokens)/len(tokens))


def get_tokens():
    reasonable, unreasonable = load_data()
    test_ds = load_test_data()
    _, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)
    print("Reasonable dataset:")
    count_tokens(tokenizer, reasonable)
    print("Unreasonable dataset:")
    count_tokens(tokenizer, unreasonable)

    for i, d in enumerate(test_ds):
        print(f"Test set {i} dataset:")
        count_tokens(tokenizer, d)



def get_gradient_for_dict_response():
    reasonable_dict, unreasonable_dict = load_dict_response()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)

    true_gradients = []
    false_gradients = []
    # Pi = None
    # try:
    Pi = torch.load(f'data/pi_matrix.pt') 
    for true_resp, false_resp in zip(reasonable_dict, unreasonable_dict):
        true_gradient = []
        false_gradient = []
        if true_resp and false_resp:
            true_data = [{"data": true[0] + true[1]} for true in true_resp]
            true_stats, Pi = get_gradients_over_dataset(model, tokenizer, true_data, LORA=True, Pi=Pi)
            true_gradient.append(true_stats)

            false_data = [{"data": false[0] + false[1]} for false in false_resp]
            false_stats, Pi = get_gradients_over_dataset(model, tokenizer, false_data, LORA=True, Pi=Pi)
            false_gradient.append(false_stats)
        
        # true_gradient = torch.stack(true_gradient, dim=0)
        true_gradients.append(true_gradient)
        false_gradients.append(false_gradient)
    
    print(len(true_gradients), len(false_gradients))
    torch.save({"sketches": true_gradients}, f'data/reasonable_dict_gradient')
    torch.save({"sketches": false_gradients}, f'data/unreasonable_dict_gradient')
    return true_gradients, false_gradients





def main():
    reasonable, unreasonable = load_data()
    Pi = torch.load(f'data/pi_matrix.pt')
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)

    split = 'reasonable'
    stats = get_gradients_over_dataset(model, tokenizer, reasonable, LORA=True, Pi=Pi)
    torch.save({"sketches": stats}, f'data/{split}_gradient_')
    # np.save(f'data/{split}_gradient', stats)
    print("Collected gradient stats for", len(stats), "batches.")

    split = 'unreasonable'
    stats = get_gradients_over_dataset(model, tokenizer, unreasonable, LORA=True, Pi=Pi)
    torch.save({"sketches": stats}, f'data/{split}_gradient_')
    # np.save(f'data/{split}_gradient', stats)
    print("Collected gradient stats for", len(stats), "batches.")


def test_gradient():
    true_set, false_set = load_test_data()
    paths = ['test_true_gradient', 'test_false_gradient']

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)  
    for path, ds in zip(paths, [true_set, false_set]):
        
        stats = get_gradients_over_dataset(model, tokenizer, ds, LORA=True)

        torch.save({"sketches": stats}, f'data/{path}')
        # np.save(f'data/{split}_gradient', stats)
        print(f"Collected gradient stats for {path}", len(stats), "batches.")



def get_gradient():
    with open('data_resp/true_resps.json', 'r') as f:
        true_set = json.load(f)
    
    with open('data_resp/false_resps.json', 'r') as f:
        false_set = json.load(f)
    
    paths = ['true_resp_gradient', 'false_resp_gradient']
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)

    Pi = torch.load(f'data/pi_matrix.pt')
    
    for path, ds in zip(paths, [true_set, false_set]):
        stats, _ = get_gradients_over_dataset(model, tokenizer, ds, LORA=True, Pi=Pi)

        torch.save({"sketches": stats}, f'data/{path}')
        # np.save(f'data/{split}_gradient', stats)
        print(f"Collected gradient stats for {path}", len(stats), "batches.")



if __name__ == "__main__":
    main()
    # test_gradient()
    # get_tokens()
    # get_gradient_for_dict_response()
    # get_gradient()





