import argparse
import json
import os

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def dpo_train(cfg, ds: Dataset):
    model = AutoModelForCausalLM.from_pretrained(cfg.dpo_model_dir, torch_dtype="auto", device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_dir, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpo_model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n = len(ds)
    base_accum = cfg.dpo_accumulation_step
    if n > 64:
        accum_steps = base_accum
    elif n > 16:
        accum_steps = max(1, base_accum // 2)
    else:
        accum_steps = max(1, base_accum // 4)

    dpo_cfg = DPOConfig(
        output_dir=cfg.dpo_output_dir,
        num_train_epochs=cfg.dpo_epochs,
        learning_rate=cfg.dpo_lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=cfg.dpo_batch_size_per_device,
        gradient_accumulation_steps=accum_steps,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=1.0,
        logging_steps=cfg.dpo_logging_steps,
        save_steps=cfg.dpo_save_steps,
        save_total_limit=cfg.dpo_save_total_limit,
        beta=cfg.dpo_beta,
        max_prompt_length=cfg.dpo_max_prompt_length,
        max_length=cfg.dpo_max_prompt_length + cfg.dpo_max_completion_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", required=True, help="Root dir used by codegen_sample_label_dpo.py")
    p.add_argument("--labeling_method", choices=["gradient", "trace"], default="gradient")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--base_model_dir", required=True)
    p.add_argument("--dpo_model_dir", default=None, help="Policy model to train. Defaults to --base_model_dir")

    p.add_argument("--dpo_epochs", type=int, default=1)
    p.add_argument("--dpo_lr", type=float, default=2e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--dpo_batch_size_per_device", type=int, default=1)
    p.add_argument("--dpo_accumulation_step", type=int, default=8)
    p.add_argument("--dpo_logging_steps", type=int, default=1)
    p.add_argument("--dpo_save_steps", type=int, default=10)
    p.add_argument("--dpo_save_total_limit", type=int, default=1)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_max_prompt_length", type=int, default=4096)
    p.add_argument("--dpo_max_completion_length", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    step_dir = os.path.join(args.output_root, args.labeling_method, f"s{args.step}")
    pairs_path = os.path.join(step_dir, args.labeling_method, "dpo_trainset.json")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing DPO pair file: {pairs_path}")

    with open(pairs_path, "r") as f:
        dpo_rows = json.load(f)

    if not dpo_rows:
        raise RuntimeError(f"DPO pair file is empty: {pairs_path}")

    ds = Dataset.from_list(dpo_rows)

    if args.dpo_model_dir is None:
        args.dpo_model_dir = args.base_model_dir
    args.dpo_output_dir = os.path.join(step_dir, "dpo")

    print(f"Loading DPO pairs from: {pairs_path}")
    print(f"DPO model: {args.dpo_model_dir}")
    print(f"Ref model: {args.base_model_dir}")
    print(f"Output dir: {args.dpo_output_dir}")

    if os.path.exists(os.path.join(args.dpo_output_dir, "model-00001-of-00002.safetensors")):
        print("Existing DPO checkpoint found, skipping training.")
        return

    dpo_train(args, ds)


if __name__ == "__main__":
    main()
