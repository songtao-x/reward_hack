"""
Docstring for train.dpo_train
dpo train functions

dpo_train: dpo_trainer

build dpo training set:

"""


import os
import argparse
import json
from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # In tuning mode, use the requested accumulation step exactly.
    if cfg.dpo_mode == "dpo_params_tune":
        accum_steps = cfg.dpo_accumulation_step
    else:
        N = len(ds)
        base = cfg.dpo_accumulation_step  # default 8
        if N > 16:
            accum_steps = base
        elif 8 < N <= 16:
            accum_steps = max(1, base // 2)
        else:
            accum_steps = max(1, base // 4)

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
        save_total_limit=cfg.dpo_save_total_limit,
        save_strategy="no",

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


def _resolve_dpo_output_paths(cfg) -> Tuple[str, str, str]:
    step_root = os.path.join(cfg.dpo_output_dir, cfg.labeling_method, f"s{cfg.step}")

    if cfg.dpo_mode == "dpo_params_tune":
        if not cfg.dpo_tune_tag:
            raise ValueError("--dpo_tune_tag is required when --dpo_mode dpo_params_tune is used.")
        tagged_root = os.path.join(step_root, cfg.dpo_tune_tag)
        return step_root, tagged_root, os.path.join(tagged_root, "dpo")

    return step_root, step_root, os.path.join(step_root, "dpo")


def add_dpo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--step", type=int, default=0, help="Current training step, used for naming and loading.")
    parser.add_argument(
        "--dpo_mode",
        default="standard",
        choices=["standard", "dpo_params_tune"],
        help="DPO pipeline mode. 'dpo_params_tune' reuses step data but writes the DPO model into a parameter-tagged subdirectory.",
    )

    # model args
    parser.add_argument("--dpo_output_dir", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat")
    parser.add_argument("--dpo_model_dir", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="Qwen/Qwen3-4B")

    # training args
    parser.add_argument("--dpo_max_steps", type=int, default=1000)
    # parser.add_argument("--dpo_epochs", type=int, default=2)
    # parser.add_argument("--dpo_lr", type=float, default=1e-5)
    parser.add_argument("--dpo_epochs", type=int, default=1)
    parser.add_argument("--dpo_lr", type=float, default=2e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--dpo_batch_size_per_device", type=int, default=1)
    # parser.add_argument("--dpo_accumulation_step", type=int, default=4)
    parser.add_argument("--dpo_accumulation_step", type=int, default=8)

    # logging args
    parser.add_argument("--dpo_logging_steps", type=int, default=1)
    parser.add_argument("--dpo_save_steps", type=int, default=2)
    parser.add_argument("--dpo_save_total_limit", type=int, default=1)

    parser.add_argument("--dpo_beta", type=float, default=0.1)

    parser.add_argument("--dpo_max_prompt_length", type=int, default=4096)
    parser.add_argument("--dpo_max_completion_length", type=int, default=4096)

    # labeling method for dpo trainset
    parser.add_argument("--labeling_method", type=str, default="gradient", 
                        choices=["trace", "gradient"], help="method to label dpo trainset")
    # cluster method for gradient analysis
    parser.add_argument("--cluster_method", type=str, default="kmeans", 
                        choices=["kmeans", "gmm"], help="clustering method for gradient analysis")
    parser.add_argument(
        "--dpo_trainset_path",
        type=str,
        default="",
        help="Optional labeled DPO trainset path. If set, skip deriving the default step path and train directly from this file.",
    )
    parser.add_argument(
        "--dpo_tune_tag",
        type=str,
        default="",
        help="Optional parameter tag appended to the step directory when --dpo_mode dpo_params_tune is used.",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dpo_args(parser)
    args = parser.parse_args()

    step_root, model_output_root, args.dpo_output_dir = _resolve_dpo_output_paths(args)

    if args.dpo_model_dir:
        model_name = args.dpo_model_dir
    elif args.dpo_trainset_path:
        grpo_model_dir = os.path.join(step_root, "grpo")
        model_name = grpo_model_dir if os.path.exists(grpo_model_dir) else args.base_model_dir
    else:
        model_name = os.path.join(step_root, "grpo")
    args.dpo_model_dir = model_name

    if args.dpo_trainset_path:
        pairs_path = args.dpo_trainset_path
    else:
        pairs_path = os.path.join(step_root, args.labeling_method, "dpo_trainset.json")

    print("Running DPO training...\n")
    print(f"Current labeling method: {args.labeling_method}")
    print(f"Using step data dir: {step_root}\n")
    if model_output_root != step_root:
        print(f"Using tuned model dir: {args.dpo_output_dir}\n")
    print(f"DPO trainset path: {pairs_path}")
    print(f"Policy model dir: {args.dpo_model_dir}")
    print(f"Reference model dir: {args.base_model_dir}")

    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing DPO trainset: {pairs_path}")

    with open(pairs_path, "r") as f:
        dpo_trainset = json.load(f)

    if len(dpo_trainset) == 0:
        print(f"Empty DPO trainset at {pairs_path}; skipping DPO training.")
        raise SystemExit(0)

    dpo_trainset = Dataset.from_list(dpo_trainset)

    if os.path.exists(os.path.join(args.dpo_output_dir, "model-00001-of-00002.safetensors")):
        print(f"Found existing DPO model at {args.dpo_output_dir}, skipping DPO training...")
    else:
        dpo_train(cfg=args, ds=dpo_trainset)

