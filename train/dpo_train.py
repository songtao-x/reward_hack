"""
Docstring for train.dpo_train
dpo train functions

dpo_train: dpo_trainer

build dpo training set:

"""


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


# ----------------------------
# 2) BUILD DPO PAIRS PER PROMPT
# ----------------------------
def build_dpo_pairs_from_labeled_rollouts(
    labeled: List[Dict[str, Any]],
    min_conf_gap: float = 0.0,
) -> Dataset:
    """
    labeled records must contain:
      prompt_id, prompt, completion, label_is_positive, label_score

    Strategy:
      - per prompt: pick winner = best positive (max score)
                  pick loser  = worst negative (min score) or max confidence negative
      - only keep if both exist and confidence gap >= min_conf_gap
    """
    # group by prompt_id
    by_pid: Dict[int, List[Dict[str, Any]]] = {}
    for r in labeled:
        by_pid.setdefault(r["prompt_id"], []).append(r)

    dpo_rows = []
    for pid, rows in by_pid.items():
        pos = [r for r in rows if r["label_is_positive"]]
        neg = [r for r in rows if not r["label_is_positive"]]
        if not pos or not neg:
            continue

        # choose winner/loser
        winner = max(pos, key=lambda r: r["label_score"])
        # pick a "strong" negative; here: max score among negatives if score means confidence of hacked,
        # but since our placeholder score is "positive confidence", use min among negs.
        loser = min(neg, key=lambda r: r["label_score"])

        gap = winner["label_score"] - loser["label_score"]
        if gap < min_conf_gap:
            continue

        dpo_rows.append(
            {
                "prompt": winner["prompt"],
                "chosen": winner["completion"],
                "rejected": loser["completion"],
                "winner_score": winner["label_score"],
                "loser_score": loser["label_score"],
            }
        )

    if len(dpo_rows) == 0:
        return Dataset.from_list([])

    return Dataset.from_list(dpo_rows)



def dpo_train(cfg, ds):
    """
    Docstring for dpo_train
    
    :param cfg: dpo trainer config
    :param ds: dpo trainset
    """

    model_dir = cfg.dpo_model_dir
    base_model_dir = cfg.base_model_dir

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # adjust accum steps
    N = len(ds)
    base = cfg.dpo_accumulation_step  # default 8
    if N > 16:
        accum_steps = base          # 8
    elif 8 < N <= 16:
        accum_steps = max(1, base // 2)  # 4
    else:
        accum_steps = max(1, base // 4)  # 2

    dpo_cfg = DPOConfig(
        output_dir=cfg.dpo_output_dir,

        # training length control

        # using epoch control 
        # max_steps=cfg.dpo_max_steps,
        num_train_epochs=cfg.dpo_epochs,

        # optimizer / batching
        learning_rate=cfg.dpo_lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=cfg.dpo_batch_size_per_device,
        gradient_accumulation_steps=accum_steps,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=1.0,

        # logging / saving
        logging_steps=cfg.dpo_logging_steps,
        save_steps=cfg.dpo_save_steps,
        save_total_limit=1,

        # --- DPO-specific ---
        beta=cfg.dpo_beta,

        # --- lengths ---
        max_prompt_length=cfg.dpo_max_prompt_length,
        max_length=cfg.dpo_max_prompt_length + cfg.dpo_max_completion_length,
    )
    

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    dpo_trainer.train()
    dpo_trainer.save_model()


def add_dpo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--step", type=int, default=0, help="Current training step, used for naming and loading.")

    # model args
    parser.add_argument("--dpo_output_dir", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat")
    parser.add_argument("--dpo_model_dir", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat/gradient/s0/grpo")
    parser.add_argument("--base_model_dir", type=str, default="Qwen/Qwen3-4B")

    # training args
    parser.add_argument("--dpo_max_steps", type=int, default=1000)
    # parser.add_argument("--dpo_epochs", type=int, default=2)
    # parser.add_argument("--dpo_lr", type=float, default=1e-5)
    parser.add_argument("--dpo_epochs", type=int, default=1)
    parser.add_argument("--dpo_lr", type=float, default=2e-6)
    parser.add_argument("--warmup_ratio", default=0.1)

    parser.add_argument("--dpo_batch_size_per_device", type=int, default=1)
    # parser.add_argument("--dpo_accumulation_step", type=int, default=4)
    parser.add_argument("--dpo_accumulation_step", type=int, default=8)

    # logging args
    parser.add_argument("--dpo_logging_steps", type=int, default=1)
    parser.add_argument("--dpo_save_steps", type=int, default=2)

    parser.add_argument("--dpo_beta", type=float, default=0.1)

    parser.add_argument("--dpo_max_prompt_length", type=int, default=4096)
    parser.add_argument("--dpo_max_completion_length", type=int, default=4096)

    # labeling method for dpo trainset
    parser.add_argument("--labeling_method", type=str, default="gradient", 
                        choices=["trace", "gradient"], help="method to label dpo trainset")
    # cluster method for gradient analysis
    parser.add_argument("--cluster_method", type=str, default="kmeans", 
                        choices=["kmeans", "gmm"], help="clustering method for gradient analysis")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dpo_args(parser)
    args = parser.parse_args()

    # basic settings
    step = args.step
    n_samples = 16
    n_grpo_samples = 16
    
    output_dir = f"{args.dpo_output_dir}/{args.labeling_method}/s{step}"
    model_name = f"{args.dpo_output_dir}/{args.labeling_method}/s{step}/grpo"

    args.dpo_output_dir = output_dir + '/dpo'
    args.dpo_model_dir = model_name

    # load dpo trainset
    print("Running DPO training...\n")
    print(f'Current labeling method: {args.labeling_method}')

    with open(os.path.join(output_dir, args.labeling_method, 'dpo_trainset.json'), 'r') as f:
        dpo_trainset = json.load(f)
        dpo_trainset = Dataset.from_list(dpo_trainset)
    
    # dpo training
    if os.path.exists(args.dpo_output_dir + '/model-00001-of-00002.safetensors'):
        print(f"Found existing DPO model at {args.dpo_output_dir}, skipping DPO training...")
    else:
        dpo_train(cfg=args, ds=dpo_trainset)



