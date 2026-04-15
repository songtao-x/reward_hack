# train_rloo_qwen_trl.py
"""
rloo train:
rloo_train: implement rloo training with rloo_trainer

sample_responses: sample K responses from rloo trained model

rh_labeling: reward hacking labeling. Gradient method / Trace

pipeline: s0 rloo training --> s0 sampling --> s0 dpo training --> s1 rloo training
"""

import re
import os
import random
import pandas as pd
import json
from tqdm.auto import tqdm

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import RLOOConfig, RLOOTrainer
from trl.rewards import accuracy_reward
from utils_.data_process import result_processer
from vllm import LLM, SamplingParams

from icl.gradient.analysis import GradientAnalyzer
from train.labeling import gradient_analysis, trace_analysis
from train.load_data import load_data


random.seed(224)
SEED = 224

MAX_TOKEN = 4096

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def _find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    latest_step = None
    latest_path = None
    for name in os.listdir(output_dir):
        match = _CHECKPOINT_RE.fullmatch(name)
        if not match:
            continue
        checkpoint_path = os.path.join(output_dir, name)
        if not os.path.isdir(checkpoint_path):
            continue
        step = int(match.group(1))
        if latest_step is None or step > latest_step:
            latest_step = step
            latest_path = checkpoint_path
    return latest_path


def _find_latest_resumable_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    latest_checkpoint = _find_latest_checkpoint(output_dir)
    if latest_checkpoint and os.path.exists(os.path.join(latest_checkpoint, "trainer_state.json")):
        return latest_checkpoint

    checkpoints = []
    for name in os.listdir(output_dir):
        match = _CHECKPOINT_RE.fullmatch(name)
        if not match:
            continue
        checkpoint_path = os.path.join(output_dir, name)
        if not os.path.isdir(checkpoint_path):
            continue
        checkpoints.append((int(match.group(1)), checkpoint_path))
    checkpoints.sort(reverse=True)
    for _, checkpoint_path in checkpoints:
        if os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
            return checkpoint_path
    return None


# --- Example: a very simple reward function (replace with your verifier / trace / etc.) ---
def rloo_reward_func(prompts, completions: list[str], **kwargs) -> List[float]:
    """
    reward function for RLOO training
    reward_funcs can be a callable that returns a list[float] of rewards
    aligned with each completion.
    """
    solution = kwargs["label"]
    if isinstance(completions, str):
        completions = [completions]
    if isinstance(solution, str):
        solution = [solution]

    assert len(completions) == len(solution), "Mismatched completion and solution"
    rewards = []
    for completion, s in zip(completions, solution):
        if isinstance(completion, str):
            response = completion
        else:
            response = completion[0]["content"]

        _, score = result_processer(response=response, label=s)
        rewards.append(score)
    return rewards


def sample_responses(
    model_name,
    ds: List[str],
    output_dir,
    k: int = 8,
    k_max: int = 32,
    batch_size=16,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_token: int = 4096,
) -> List[Dict[str, Any]]:
    """
    Input:
    k: expected correct responses for each prompt
    k_max: max sampling times for each prompt
    batch_size: batch size for vLLM generation

    Returns a list of dict records:
      {prompt, completion, prompt_id, sample_id}
    """

    print(f"\n\nSampling responses from model: {model_name}...\n\n")

    save_dir = output_dir + "/rloo_samples"
    os.makedirs(save_dir, exist_ok=True)

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        max_tokens=max_token,  # max new tokens
    )

    # Load model (HF name or local path)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
    )

    print(f"\nFinish loading model {model_name} for sampling...\n")

    records = []
    all_records = []
    for i, example in enumerate(tqdm(ds, desc="Sampling responses")):
        pid = example["pid"]
        prompt = example["prompt"]
        label = example["label"]

        used = 0
        correct = []
        all_gen = []
        while used < k_max and len(correct) < k:
            bsz = min(batch_size, k_max - used)
            results = llm.generate([prompt] * bsz, sampling_params)

            for r in results:
                out = r.outputs[0].text
                _, score = result_processer(out, label)

                used += 1
                all_gen.append(out)
                if score:
                    correct.append(out)
                    if len(correct) >= k:
                        break

            if len(correct) >= k:
                break

        records.append({"pid": pid, "prompt": prompt, "label": label, "gen": correct})
        all_records.append({"pid": pid, "prompt": prompt, "label": label, "gen": all_gen})

        print(f"\nFinish sampling for {i}-th prompt, got {len(correct)} correct responses with {used} samples.\n")

    save_path = os.path.join(save_dir, "samples.json")
    with open(save_path, "w") as f:
        json.dump(records, f, indent=4)

    save_path = os.path.join(save_dir, "all_samples.json")
    with open(save_path, "w") as f:
        json.dump(all_records, f, indent=4)

    return records


def trace_labeling(current_dir, ds=None, train_type="dpo", threshold=0.2904761904761905):
    # qwen3-4b arlsat trace threshold=0.2904761904761905

    with open(os.path.join(current_dir, "rloo_samples", "samples.json"), "r") as f:
        ds = json.load(f)

    model_name = current_dir + "/rloo"
    save_dir = current_dir + "/trace"
    os.makedirs(save_dir, exist_ok=True)

    trainset = []
    if train_type == "dpo":
        for i, record in enumerate(ds):
            prompt = record["prompt"]
            label = record["label"]
            gen = record["gen"]

            if len(gen) < 2:
                continue

            processed_gen = [{"prompt": prompt, "gen": g, "label": label} for g in gen]

            os.makedirs(f"{save_dir}/idx{i}", exist_ok=True)
            pred = trace_analysis(
                processed_gen,
                model_name,
                save_dir=f"{save_dir}/idx{i}",
                threshold=threshold,
            )

            true_idx = [j for j, p in enumerate(pred) if p == 1]
            false_idx = [j for j, p in enumerate(pred) if p == 0]

            if true_idx and false_idx:
                sample = {
                    "prompt": prompt,
                    "chosen": gen[true_idx[0]],
                    "rejected": gen[false_idx[0]],
                    "label": label,
                }
                trainset.append(sample)

        with open(os.path.join(save_dir, "dpo_trainset.json"), "w") as f:
            json.dump(trainset, f, indent=4)

        return trainset

    elif train_type == "sft":
        pred = trace_analysis(ds, model_name, save_dir=f"{save_dir}", threshold=threshold)

        trainset = []
        for i, record in enumerate(ds):
            if pred[i] == 1:
                trainset.append({"instruction": record["prompt"], "input": "", "output": record["gen"]})

        # security code, should be removed
        if len(trainset) == 0:
            print("\n\nNo samples labeled as true, please check the threshold or the trace analysis.\n\n")
            trainset.append({"instruction": ds[0]["prompt"], "input": "", "output": ds[0]["gen"]})

        with open(os.path.join(save_dir, "dpo_trainset.json"), "w") as f:
            json.dump(trainset, f, indent=4)

        return trainset


def gradient_labeling(
    output_dir,
    ds=None,
    train_type="dpo",
    get_gradient=True,
    load_pi=True,
    method="cluster",
    cluster_method="kmeans",
):
    """
    output: training set based on train_type
    """
    print(f"\nCurrent gradient labeling method: {method}, Cluster method: {cluster_method}, train type: {train_type}\n")

    if ds is None:
        with open(os.path.join(output_dir, "rloo_samples", "samples.json"), "r") as f:
            ds = json.load(f)

    analyzer = GradientAnalyzer()
    model_name = output_dir + "/rloo"
    save_dir = output_dir + "/gradient"
    os.makedirs(save_dir, exist_ok=True)

    trainset = []
    if train_type == "dpo":
        for i, record in enumerate(ds):
            prompt = record["prompt"]
            label = record["label"]
            gen = record["gen"]

            if len(gen) < 2:
                continue

            processed_gen = [{"input": prompt, "output": g} for g in gen]

            os.makedirs(f"{save_dir}/idx{i}", exist_ok=True)
            if method == "svm":
                pred = gradient_analysis(
                    analyzer=analyzer,
                    model_name=model_name,
                    ds=processed_gen,
                    save_dir=f"{save_dir}/idx{i}",
                    method="svm",
                    get_gradient=get_gradient,
                    test_ratio=0.7,
                    use_pca=False,
                    use_svd=False,
                    use_t_sne=False,
                    normalized=True,
                )
            elif method == "cluster":
                pred = gradient_analysis(
                    analyzer=analyzer,
                    model_name=model_name,
                    ds=processed_gen,
                    save_dir=f"{save_dir}/idx{i}",
                    method="cluster",
                    cluster_method=cluster_method,
                    get_gradient=get_gradient,
                    load_pi=load_pi,
                    test_ratio=0.7,
                    use_pca=False,
                    use_svd=False,
                    use_t_sne=False,
                    normalized=True,
                )

                # simple clustering: flip majority to 0 (treat minority as positive)
                num_1 = sum(pred)
                num_0 = len(pred) - num_1
                if num_1 > num_0:
                    pred = [0 if p == 1 else 1 for p in pred]

            true_idx = [j for j, p in enumerate(pred) if p == 1]
            false_idx = [j for j, p in enumerate(pred) if p == 0]
            if not true_idx or not false_idx:
                continue

            sample = {
                "prompt": prompt,
                "chosen": gen[true_idx[0]],
                "rejected": gen[false_idx[0]],
                "label": label,
            }
            trainset.append(sample)

        with open(os.path.join(save_dir, "dpo_trainset.json"), "w") as f:
            json.dump(trainset, f, indent=4)

        return trainset

    elif train_type == "sft":
        processed_gen = [{"input": record["prompt"], "output": record["gen"]} for record in ds]

        os.makedirs(f"{save_dir}", exist_ok=True)
        if method == "svm":
            pred = gradient_analysis(
                analyzer=analyzer,
                model_name=model_name,
                ds=processed_gen,
                save_dir=f"{save_dir}",
                method="svm",
                get_gradient=get_gradient,
                test_ratio=0.7,
                use_pca=False,
                use_svd=False,
                use_t_sne=False,
                normalized=True,
            )
            trainset = [
                {"instruction": record["prompt"], "input": "", "output": record["gen"]}
                for record, p in zip(ds, pred)
                if p == 1
            ]

        elif method == "cluster":
            if len(processed_gen) == 1:
                trainset = [{"instruction": record["prompt"], "input": "", "output": record["gen"]} for record in ds]
            elif len(processed_gen) == 0:
                trainset = []
            else:
                km_centers = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/baseline_km_bigmath_q2"

                pred = gradient_analysis(
                    analyzer=analyzer,
                    model_name=model_name,
                    ds=processed_gen,
                    save_dir=f"{save_dir}",
                    method="cluster",
                    cluster_method=cluster_method,
                    baseline_km_centers=km_centers,
                    get_gradient=get_gradient,
                    load_pi=load_pi,
                    test_ratio=0.7,
                    use_pca=False,
                    use_svd=False,
                    use_t_sne=False,
                    normalized=True,
                )
                trainset = [
                    {"instruction": record["prompt"], "input": "", "output": record["gen"]}
                    for record, p in zip(ds, pred)
                    if p == 1
                ]

                # security code, should be removed
                if len(trainset) == 0:
                    print("\n\nNo samples labeled as true, please check the threshold or the gradient analysis.\n\n")
                    trainset.append({"instruction": ds[0]["prompt"], "input": "", "output": ds[0]["gen"]})

        with open(os.path.join(save_dir, "sft_trainset.json"), "w") as f:
            json.dump(trainset, f, indent=4)

        return trainset


def rloo_train(cfg, ds):
    """
    RLOO training

    param cfg: rloo cfg
    """

    # Use an instruct-tuned Qwen checkpoint
    if cfg.step == 0:
        model_dir = cfg.rloo_init_model
    else:
        model_dir = cfg.rloo_input_model

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # ds cfg
    with open(cfg.ds_cfg, "r") as f:
        ds_cfg = json.load(f)
        ds_cfg["train_micro_batch_size_per_gpu"] = cfg.rloo_batch_size_per_device
        ds_cfg["gradient_accumulation_steps"] = cfg.rloo_accumulation_steps

    rloo_cfg = RLOOConfig(
        output_dir=cfg.rloo_output_dir,
        num_train_epochs=cfg.rloo_epochs,

        learning_rate=cfg.rloo_lr,
        per_device_train_batch_size=cfg.rloo_batch_size_per_device,
        gradient_accumulation_steps=cfg.rloo_accumulation_steps,

        num_generations=cfg.rloo_rollout_n,
        num_iterations=cfg.rloo_num_iterations,
        epsilon=cfg.rloo_epsilon,
        beta=cfg.rloo_beta,

        max_completion_length=cfg.rloo_max_completion_length,

        logging_steps=cfg.rloo_logging_steps,
        save_steps=cfg.rloo_save_steps,

        temperature=cfg.rloo_temperature,
        top_p=cfg.rloo_top_p,

        # vllm for generation
        use_vllm=True,
        vllm_tensor_parallel_size=4,
        vllm_gpu_memory_utilization=0.6,
        vllm_mode="colocate",

        # deepspeed
        deepspeed=ds_cfg,

        # precision
        bf16=True,
    )

    trainer = RLOOTrainer(
        model=model,
        reward_funcs=rloo_reward_func,
        train_dataset=ds,
        args=rloo_cfg,
        processing_class=tokenizer,
    )

    resume_from_checkpoint = cfg.rloo_resume_from_checkpoint or None
    if resume_from_checkpoint:
        print(f"Resuming RLOO from checkpoint: {resume_from_checkpoint}\n")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()


def add_rloo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # env args
    parser.add_argument("--step", type=int, default=0, help="Current training step, used for naming and loading/saving.")
    parser.add_argument("--labeling_method", type=str, default="trace", help="Labeling method for reward hacking, e.g. trace or gradient.")
    parser.add_argument("--cluster_method", type=str, default="kmeans", help="Clustering method for gradient labeling, e.g. kmeans or dbscan.")
    parser.add_argument("--trace_threshold", type=float, default=0.2904761904761905, help="Threshold for trace labeling, only used when labeling_method is trace.")
    parser.add_argument("--train_type", type=str, default="dpo", help="Train type for reward hacking, e.g. dpo or sft.")
    parser.add_argument("--dataset", type=str, default="arlsat", help="Dataset name for training, e.g. arlsat.")

    # model
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to use for each step (after excluding rloo training samples).")
    parser.add_argument("--n_rl_samples", type=int, default=16, help="Number of samples to use for RL training in each step.")
    parser.add_argument("--rloo_sample", action="store_true", help="Whether to run sampling after RLOO training.")
    parser.add_argument("--rloo_init_model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--rloo_input_model",
        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s0/rloo",
        help="Model path for rloo training, used when step > 0.",
    )

    # ----- IO / run control -----
    parser.add_argument(
        "--rloo_output_dir",
        type=str,
        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s0/rloo",
        help="Output directory for TRL RLOO checkpoints/logs.",
    )
    parser.add_argument(
        "--rloo_resume_from_checkpoint",
        type=str,
        default="",
        help="Resume TRL RLOO training from this checkpoint directory.",
    )
    parser.add_argument(
        "--rloo_max_steps",
        type=int,
        default=None,
        help="If set, use max_steps for RLOO (overrides epochs).",
    )
    parser.add_argument("--rloo_epochs", type=float, default=1.0, help="Number of RLOO epochs (ignored if --rloo_max_steps is set).")

    # ----- batch / optimization -----
    parser.add_argument("--rloo_lr", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--rloo_batch_size_per_device", type=int, default=2, help="Per-device batch size.")
    parser.add_argument(
        "--rloo_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )

    # ----- RLOO core -----
    parser.add_argument("--rloo_rollout_n", type=int, default=8, help="Completions per prompt.")
    parser.add_argument("--rloo_num_iterations", type=int, default=1, help="Policy updates per rollout batch.")
    parser.add_argument("--rloo_epsilon", type=float, default=0.2, help="PPO-style clip range epsilon.")
    parser.add_argument("--rloo_beta", type=float, default=0.01, help="KL coefficient beta.")

    # ----- lengths -----
    parser.add_argument("--rloo_max_prompt_length", type=int, default=4096, help="Max prompt length (tokens).")
    parser.add_argument("--rloo_max_completion_length", type=int, default=4096, help="Max completion length / max_new_tokens.")

    # ----- generation sampling -----
    parser.add_argument("--rloo_temperature", type=float, default=0.7, help="Sampling temperature for RLOO rollouts.")
    parser.add_argument("--rloo_top_p", type=float, default=0.9, help="Top-p nucleus sampling for RLOO rollouts.")

    # ----- logging / checkpointing -----
    parser.add_argument("--rloo_logging_steps", type=int, default=2, help="Log every N steps.")
    parser.add_argument("--rloo_save_steps", type=int, default=2, help="Save every N steps.")

    # ----- memory / distributed -----
    parser.add_argument("--rloo_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")

    parser.add_argument("--rloo_use_fsdp", action="store_true", help="Enable FSDP in Trainer.")

    # deepspeed
    parser.add_argument("--ds_cfg", default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/ds_zero3.json")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_rloo_args(parser=parser)
    rloo_cfg = parser.parse_args()

    step = rloo_cfg.step
    n_samples = rloo_cfg.n_samples
    n_rl_samples = rloo_cfg.n_rl_samples

    output_dir = f"{rloo_cfg.rloo_output_dir}/{rloo_cfg.labeling_method}/s{step}"
    model_name = f"{rloo_cfg.rloo_output_dir}/{rloo_cfg.labeling_method}/s{step}/rl"
    rloo_cfg.rloo_output_dir = model_name

    if step == 0:
        rloo_cfg.rloo_input_model = rloo_cfg.rloo_init_model

    print(f"Current step: {step}, labeling_method: {rloo_cfg.labeling_method} rloo_sample: {rloo_cfg.rloo_sample}\n")
    print(
        f"\nCurrent dataset: {rloo_cfg.dataset}, train type: {rloo_cfg.train_type}, rloo input model: {rloo_cfg.rloo_input_model}\n"
    )

    if not rloo_cfg.rloo_sample:
        ds_ = load_data(dataset=rloo_cfg.dataset)[
            step * (n_rl_samples + n_samples) : step * (n_rl_samples + n_samples) + n_rl_samples
        ]
        ds = Dataset.from_list(ds_)
        if not rloo_cfg.rloo_resume_from_checkpoint:
            latest_checkpoint = _find_latest_resumable_checkpoint(model_name)
            if latest_checkpoint:
                rloo_cfg.rloo_resume_from_checkpoint = latest_checkpoint
                print(f"Detected existing RLOO checkpoint, auto-resuming from {latest_checkpoint}.\n")

        if os.path.exists(model_name + "/model-00001-of-00002.safetensors"):
            print(f"Model dir {model_name} already exists, skip training.\n")
        else:
            print(f"Start RLOO training for step {step}...\n\n")
            rloo_train(cfg=rloo_cfg, ds=ds)

    else:
        ds = load_data(dataset=rloo_cfg.dataset)[
            step * (n_rl_samples + n_samples) + n_rl_samples : (step + 1) * (n_rl_samples + n_samples)
        ]
        ds = Dataset.from_list(ds)

        if os.path.exists(os.path.join(output_dir, "rloo_samples", "samples.json")):
            print(f"Samples already exist for step {step}, skip sampling.\n\n")
            with open(os.path.join(output_dir, "rloo_samples", "samples.json"), "r") as f:
                samples = json.load(f)
        else:
            print(f"Start sampling responses for step {step}...\n\n")
            sample_batch_size = 4
            samples = sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        if rloo_cfg.labeling_method == "gradient":
            print(f"Start gradient labeling for step {step}...\n\n")

            if os.path.exists(os.path.join(output_dir, "gradient", "dpo_trainset.json")):
                print(f"Gradient labeling already exists for step {step}, skip labeling.\n\n")
            else:
                _ = gradient_labeling(
                    output_dir=output_dir,
                    ds=samples,
                    train_type=rloo_cfg.train_type,
                    method="cluster",
                    cluster_method=rloo_cfg.cluster_method,
                )

        elif rloo_cfg.labeling_method == "trace":
            print(f"Start trace labeling for step {step}...\n\n")

            if os.path.exists(os.path.join(output_dir, "trace", "dpo_trainset.json")):
                print(f"Trace labeling already exists for step {step}, skip labeling.\n\n")
            else:
                _ = trace_labeling(
                    current_dir=output_dir,
                    ds=samples,
                    train_type=rloo_cfg.train_type,
                    threshold=rloo_cfg.trace_threshold,
                )
