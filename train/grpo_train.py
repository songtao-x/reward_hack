# train_grpo_qwen_trl.py
"""
grpo train: 
grpo_train: implement grpo training with grpo_trainer

sample_responses: sample K responses from grpo trained model

rh_labeling: reward hacking labeling. Gradient method / Trace

pipeline: s0 grpo training --> s0 sampling --> s0 dpo training --> s1 grpo training

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
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from utils_.data_process import result_processer
from vllm import LLM, SamplingParams

from icl.gradient.analysis import GradientAnalyzer
from train.labeling import gradient_analysis, trace_analysis
from train.load_data import load_data


random.seed(224)
SEED=224

MAX_TOKEN=3072

# --- Example: a very simple reward function (replace with your verifier / trace / etc.) ---
def grpo_reward_func(prompts, completions: list[str], **kwargs) -> List[float]:
    """
    reward function for GRPO training
    reward_funcs can be a callable that returns a list[float] of rewards
    aligned with each completion.
    """
    solution = kwargs['label']
    if isinstance(completions, str):
        completions = [completions]
    if isinstance(solution, str):
        solution = [solution]
    
    # print(completions[0], len(completions))
    # print(solution[0], len(solution))
    # input()
    assert len(completions) == len(solution), "Mismatched completion and solution"
    rewards = []
    for completion, s in zip(completions, solution):
        
        if isinstance(completion, str):
            response = completion
        else:
            response = completion[0]['content']
        # reward if completion contains a number (toy)
        _, score = result_processer(response=response, label=s)
        rewards.append(score)
    return rewards


def sample_responses(
    model_name,
    ds: List[str],
    output_dir,
    k: int=8,
    k_max: int=32,
    batch_size=16,
    temp: float=0.7,
    top_p: float=0.9,
    max_token: int=MAX_TOKEN
) -> List[Dict[str, Any]]:
    """
    Input:
    k: expected correct responses for each prompt
    k_max: max sampling times for each prompt
    batch_size: batch size for vLLM generation


    Returns a list of dict records:
      {prompt, completion, prompt_id, sample_id}
    """
    
    print(f'\n\nSampling responses from model: {model_name}...\n\n')

    save_dir = output_dir + '/grpo_samples'
    os.makedirs(save_dir, exist_ok=True)

    # vllm model

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        max_tokens=max_token,   # max new tokens
    )

    # 2. Load model (HF name or local path)
     # example, change to yours
    llm = LLM(
    model=model_name,
    tensor_parallel_size=1,   # uses 4 GPUs
    )

    print(f"\nFinish loading model {model_name} for sampling...\n")

    ds_len = len(ds)
    records = []
    all_records = []
    for i, example in enumerate(tqdm(ds, desc="Sampling responses")):
        # vllm inference
        pid = example["pid"]
        prompt = example["prompt"]
        label = example["label"]

        # vLLM returns a list of RequestOutput
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
        # if i % 2 == 0:
        #     print(f"Saving intermediate sampling results for {i}-th prompt...")
            
        #     save_path = os.path.join(save_dir, f'samples.json')
        #     with open(save_path, 'w') as f:
        #         json.dump(records, f, indent=4)

    # save_dir = output_dir + '/grpo_samples'
    # os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'samples.json')
    with open(save_path, 'w') as f:
        json.dump(records, f, indent=4)
    
    save_path = os.path.join(save_dir, f'all_samples.json')
    with open(save_path, 'w') as f:
        json.dump(all_records, f, indent=4)

    return records


def trace_labeling(current_dir, model_name=None, ds=None, train_type="dpo", threshold=0.2904761904761905):
    # qwen3-4b arlsat trace threshold=0.2904761904761905
    
    if ds == None:

        with open(os.path.join(current_dir, 'grpo_samples', 'samples.json'), 'r') as f:
            ds = json.load(f)

    if model_name is None:
        model_name = current_dir + '/grpo'
    save_dir = current_dir + '/trace'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n\nRunning trace labeling on model: {model_name}, save dir: {save_dir}\n\n")
    

    trainset = []
    if train_type == "dpo":
        for i, record in enumerate(ds):
            prompt = record['prompt']
            label = record['label']
            gen = record['gen']
            
            # if correct responses < 2
            if len(gen) < 2:
                continue

            processed_gen = [{"prompt": prompt, "gen": g, "label": label} for g in gen]
            
            os.makedirs(f'{save_dir}/idx{i}', exist_ok=True)
            pred = trace_analysis(processed_gen, model_name, 
                                  save_dir=f'{save_dir}/idx{i}', threshold=threshold)
            # print(pred)
            # based on predicion 1/0, get index and split gen into true set and false set
            true_idx = [i for i, p in enumerate(pred) if p == 1]
            false_idx = [i for i, p in enumerate(pred) if p == 0]
            
            if true_idx and false_idx:
                # for dpo, only get 1 pair of pos and neg samples
                sample = {"prompt": prompt, "chosen": gen[true_idx[0]], "rejected": gen[false_idx[0]], "label": label}
                trainset.append(sample)
        
        with open(os.path.join(save_dir, 'dpo_trainset.json'), 'w') as f:
            json.dump(trainset, f, indent=4)

        return trainset
    
    elif train_type == "sft":
        pred = trace_analysis(ds, model_name, 
                            save_dir=f'{save_dir}', threshold=threshold)
        
        trainset = []
        for i, record in enumerate(ds):
            if pred[i] == 1:
                trainset.append({"instruction": record['prompt'], "input": "", "output": record['gen']})
        
        # security code, should be removed
        if len(trainset) == 0:
            print("\n\nNo samples labeled as true, please check the threshold or the trace analysis.\n\n")
            trainset.append({"instruction": ds[0]['prompt'], "input": "", "output": ds[0]['gen']})

        with open(os.path.join(save_dir, 'sft_trainset.json'), 'w') as f:
            json.dump(trainset, f, indent=4)
    
        return trainset



def gradient_labeling(output_dir, model_name=None, ds=None, train_type="dpo", get_gradient=True, load_pi=True,
                      method="cluster", cluster_method="kmeans",
                      cluster_semantics="majority",
                      baseline_km_centers=None,
                      baseline_true_cluster=1,
                      cluster_soft_prob_temp=1.0,
                      cluster_soft_prob_n_runs=8,
                      gpt_eval_prompt=None): 
    """
    
    output: training set based on train_type
    """
    print(f'\nCurrent gradient labeling method: {method}, Cluster method: {cluster_method}, train type: {train_type}\n')

    if ds == None:
        # load dataset
        with open(os.path.join(output_dir, 'grpo_samples', 'samples.json'), 'r') as f:
            ds = json.load(f)
    
    # load gradient analysis params
    analyzer = GradientAnalyzer()
    if model_name is None:
        model_name = output_dir + '/grpo'
    save_dir = output_dir + '/gradient'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n\nRunning gradient labeling on model: {model_name}, save dir: {save_dir}\n\n")

    trainset = []
    if train_type == "dpo":

        for i, record in enumerate(ds):
            prompt = record['prompt']
            label = record['label']
            gen = record['gen']

            # if no correct responses
            if len(gen) < 2:
                continue

            processed_gen = [{"input": prompt, "output": g} for g in gen]
            true_probs = None

            os.makedirs(f'{save_dir}/idx{i}', exist_ok=True)
            if method == "svm":
                # acquire baseline svm model

                # get prediction with baseline model

                pred = gradient_analysis(analyzer=analyzer, model_name=model_name, 
                                        ds=processed_gen, save_dir=f'{save_dir}/idx{i}', 
                                        method="svm",
                                        get_gradient=get_gradient, test_ratio=0.7, 
                                        use_pca=False, use_svd=False, use_t_sne=False, 
                                        normalized=True)
            elif method == "cluster":
                # km_centers = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/baseline_km_arlsat_q3"
                
                # pred = gradient_analysis(analyzer=analyzer, model_name=model_name, 
                #                         ds=processed_gen, save_dir=f'{save_dir}', 
                #                         method="cluster", cluster_method=cluster_method,
                #                         baseline_km_centers=km_centers,
                #                         get_gradient=get_gradient, load_pi=load_pi,
                #                         test_ratio=0.7, 
                #                         use_pca=False, use_svd=False, use_t_sne=False, 
                #                         normalized=True)
                
                cluster_res = gradient_analysis(analyzer=analyzer, model_name=model_name, 
                                                ds=processed_gen, save_dir=f'{save_dir}/idx{i}', 
                                                method="cluster", cluster_method=cluster_method,
                                                baseline_km_centers=baseline_km_centers,
                                                cluster_semantics=cluster_semantics,
                                                baseline_true_cluster=baseline_true_cluster,
                                                gpt_eval_prompt=gpt_eval_prompt,
                                                get_gradient=get_gradient, load_pi=load_pi,
                                                test_ratio=0.7, 
                                                use_pca=False, use_svd=False, use_t_sne=False, 
                                                normalized=True,
                                                return_detail=(cluster_method == "soft_prob"),
                                                soft_prob_temp=cluster_soft_prob_temp,
                                                soft_prob_n_runs=cluster_soft_prob_n_runs)
                if cluster_res is None:
                    continue
                if cluster_method == "soft_prob" and isinstance(cluster_res, dict):
                    pred = cluster_res["result"]
                    true_probs = cluster_res.get("true_probs", None)
                else:
                    pred = cluster_res
                    # simple clustering
                    # count 1, 0 numbers, change larger set to 1
                    num_1 = sum(pred)
                    num_0 = len(pred) - num_1
                    if num_1 < num_0:
                        pred = [0 if p == 1 else 1 for p in pred]

            # # based on predicion 1/0, get index and split gen into true set and false set
            true_idx = [i for i, p in enumerate(pred) if p == 1]
            false_idx = [i for i, p in enumerate(pred) if p == 0]
            # for dpo, get one pair of most positive and most negative samples
            if len(true_idx) != 0 and len(false_idx) != 0:
                if true_probs is not None:
                    chosen_idx = max(true_idx, key=lambda idx: true_probs[idx])
                    rejected_idx = min(false_idx, key=lambda idx: true_probs[idx])
                else:
                    chosen_idx = true_idx[0]
                    rejected_idx = false_idx[0]
                sample = {"prompt": prompt, "chosen": gen[chosen_idx], "rejected": gen[rejected_idx], "label": label}
                trainset.append(sample)
            else:
                print("\nClustering failed, no case\n")


        if len(trainset) == 0:
            print("\nNo valid DPO pairs found after clustering. Using skip policy (empty trainset).\n")

        with open(os.path.join(save_dir, 'dpo_trainset.json'), 'w') as f:
            json.dump(trainset, f, indent=4)
        
        return trainset

    elif train_type == "sft":

        processed_gen = [{"input": record['prompt'], "output": record['gen']} for record in ds]

        os.makedirs(f'{save_dir}', exist_ok=True)
        if method == "svm":
            # acquire baseline svm model

            # get prediction with baseline model

            pred = gradient_analysis(analyzer=analyzer, model_name=model_name, 
                                    ds=processed_gen, save_dir=f'{save_dir}', 
                                    method="svm",
                                    get_gradient=get_gradient, test_ratio=0.7, 
                                    use_pca=False, use_svd=False, use_t_sne=False, 
                                    normalized=True)
            trainset = [{"instruction": record['prompt'], "input": "", "output": record['gen']} for record, p in zip(ds, pred) if p == 1]
            
        elif method == "cluster":
            if len(processed_gen) == 1:
                trainset = [{"instruction": record['prompt'], "input": "", "output": record['gen']} for record in ds]
            elif len(processed_gen) == 0:
                trainset = []
            else:
                # load baselin km centers to select clusters
                km_centers = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/baseline_km_bigmath_q2"
                
                pred = gradient_analysis(analyzer=analyzer, model_name=model_name, 
                                        ds=processed_gen, save_dir=f'{save_dir}', 
                                        method="cluster", cluster_method=cluster_method,
                                        baseline_km_centers=(baseline_km_centers or km_centers),
                                        cluster_semantics=cluster_semantics,
                                        baseline_true_cluster=baseline_true_cluster,
                                        gpt_eval_prompt=gpt_eval_prompt,
                                        get_gradient=get_gradient, load_pi=load_pi,
                                        test_ratio=0.7, 
                                        use_pca=False, use_svd=False, use_t_sne=False, 
                                        normalized=True,
                                        soft_prob_temp=cluster_soft_prob_temp,
                                        soft_prob_n_runs=cluster_soft_prob_n_runs)
                # # simple clustering 
                # # count 1, 0 numbers, change fewer set to 1
                # num_1 = sum(pred)
                # num_0 = len(pred) - num_1
                # if num_1 > num_0:
                #     pred = [0 if p == 1 else 1 for p in pred]
                
                trainset = [{"instruction": record['prompt'], "input": "", "output": record['gen']} for record, p in zip(ds, pred) if p == 1]

                # security code, should be removed
                if len(trainset) == 0:
                    print("\n\nNo samples labeled as true, please check the threshold or the gradient analysis.\n\n")
                    trainset.append({"instruction": ds[0]['prompt'], "input": "", "output": ds[0]['gen']})


        with open(os.path.join(save_dir, 'sft_trainset.json'), 'w') as f:
            json.dump(trainset, f, indent=4)

        return trainset



def grpo_train(cfg, ds):
    """
    Docstring for grpo_train
    
    param cfg: grpo cfg
    """

    # Use an instruct-tuned Qwen checkpoint
    if cfg.step == 0:
        model_dir = cfg.grpo_init_model
    else:
        model_dir = cfg.grpo_input_model

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # For chat/instruct models, TRL typically expects your dataset "prompt" to already be a string prompt.
    # If you have structured messages, you can apply chat template before training.

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
        # device_map="auto",
    )


    # ds cfg
    with open(cfg.ds_cfg, 'r') as f:
        ds_cfg = json.load(f)

        ds_cfg["train_micro_batch_size_per_gpu"] = cfg.grpo_batch_size_per_device
        ds_cfg["gradient_accumulation_steps"] = cfg.grpo_accumulation_steps
    
    

    # GRPOConfig: key knobs are num_generations (group size), num_iterations (policy updates per batch),
    # beta (KL penalty weight), epsilon (clip range), plus sampling params.
    # See TRL GRPO docs for details. :contentReference[oaicite:1]{index=1}
    grpo_cfg = GRPOConfig(
        output_dir=cfg.grpo_output_dir,
        # remove max_steps if using epoch
        # max_steps=cfg.grpo_max_steps,
        num_train_epochs=cfg.grpo_epochs,

        learning_rate=cfg.grpo_lr,
        per_device_train_batch_size=cfg.grpo_batch_size_per_device,
        gradient_accumulation_steps=cfg.grpo_accumulation_steps,

        num_generations=cfg.grpo_rollout_n,
        num_iterations=cfg.grpo_num_iterations,
        epsilon=cfg.grpo_epsilon,
        beta=cfg.grpo_beta,

        # max_prompt_length=cfg.grpo_max_prompt_length,
        max_completion_length=cfg.grpo_max_completion_length,

        logging_steps=cfg.grpo_logging_steps,
        save_steps=cfg.grpo_save_steps,
        save_total_limit=cfg.grpo_save_total_limit,

        temperature=cfg.grpo_temperature,
        # vllm for generation
        # not avaiable, have to reduce 1 gpu for training
        use_vllm=True,
        vllm_tensor_parallel_size=4,
        vllm_gpu_memory_utilization=0.7,
        vllm_mode="colocate",

        # deepspeed
        deepspeed=ds_cfg,

        # precision
        bf16=True,
    )


    # --- GRPO-specific ---
                    # group size G  :contentReference[oaicite:2]{index=2}
                    # μ (updates per generation batch) :contentReference[oaicite:3]{index=3}
                            # PPO-style clip ε (when num_iterations>1) :contentReference[oaicite:4]{index=4}
                            # KL coef β (0.0 is common default) :contentReference[oaicite:5]{index=5}
                # optional scaling strategy :contentReference[oaicite:6]{index=6}

    # --- lengths ---

    # --- generation (sampling) ---
     


    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_func,  # replace with your real reward / verifier / judge
        train_dataset=ds,
        args=grpo_cfg,
        processing_class=tokenizer,    # TRL uses this for tokenization
    )

    trainer.train()
    trainer.save_model()



def add_grpo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    #env agrs
    parser.add_argument("--step", type=int, default=0, help="Current training step, used for naming and loading/saving.")
    parser.add_argument("--labeling_method", type=str, default="trace", help="Labeling method for reward hacking, e.g. trace or gradient.")
    parser.add_argument("--cluster_method", type=str, default="kmeans", help="Clustering method for gradient labeling, e.g. kmeans or soft_prob.")
    parser.add_argument("--cluster_semantics", type=str, default="majority",
                        choices=["majority", "gpt", "baseline_centroids"],
                        help="How to map cluster id to true/false labels.")
    parser.add_argument("--baseline_km_centers", type=str, default=None,
                        help="Path to baseline KMeans model for baseline-centroid semantic mapping.")
    parser.add_argument("--baseline_true_cluster", type=int, default=1,
                        help="Which baseline cluster index corresponds to true class.")
    parser.add_argument("--cluster_soft_prob_temp", type=float, default=1.0,
                        help="Temperature for soft probability from KMeans distances.")
    parser.add_argument("--cluster_soft_prob_n_runs", type=int, default=8,
                        help="Number of KMeans restarts (different seeds) for robust soft_prob.")
    parser.add_argument("--cluster_gpt_eval_prompt", type=str, default=None,
                        help="Optional prompt template for GPT semantic mapping with {question} and {response}.")
    parser.add_argument("--trace_threshold", type=float, default=0.2904761904761905, help="Threshold for trace labeling, only used when labeling_method is trace.")
    parser.add_argument("--train_type", type=str, default="dpo", help="Train type for reward hacking, e.g. dpo or sft.")
    parser.add_argument("--dataset", type=str, default="arlsat", help="Dataset name for training, e.g. arlsat.")
    
    # model
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to use for each step (after excluding grpo training samples).")
    parser.add_argument("--n_grpo_training_samples", type=int, default=16, help="Number of samples to use for GRPO training in each step.")
    parser.add_argument("--grpo_sample", action="store_true", help="Whether to run sampling after GRPO training.")
    parser.add_argument("--grpo_init_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--grpo_input_model", default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s0/grpo", help="Model path for grpo training, used when step > 0.")

    # ----- IO / run control -----
    parser.add_argument("--grpo_output_dir", type=str, 
                        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s0/grpo",
                        help="Output directory for TRL GRPO checkpoints/logs.")
    parser.add_argument("--grpo_max_steps", type=int, default=None,
                        help="If set, use max_steps for GRPO (overrides epochs).")
    parser.add_argument("--grpo_epochs", type=float, default=1.0,
                        help="Number of GRPO epochs (ignored if --grpo_max_steps is set).")

    # ----- batch / optimization -----
    parser.add_argument("--grpo_lr", type=float, default=1e-6,
                        help="Learning rate (verl: actor.optim.lr).")
    parser.add_argument("--grpo_batch_size_per_device", type=int, default=2,
                        help="Per-device batch size (verl: ppo_micro_batch_size_per_gpu).")
    parser.add_argument("--grpo_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps. "
                             "To match verl global train_batch_size: GAS=train_batch_size/(world_size*per_device_bsz).")

    # ----- GRPO core -----
    parser.add_argument("--grpo_rollout_n", type=int, default=8,
                        help="Completions per prompt (verl: rollout.n).")
    parser.add_argument("--grpo_num_iterations", type=int, default=1,
                        help="Policy updates per rollout batch (TRL mu).")
    parser.add_argument("--grpo_epsilon", type=float, default=0.2,
                        help="PPO-style clip range epsilon (mainly used if num_iterations>1).")
    parser.add_argument("--grpo_beta", type=float, default=0.01,
                        help="KL coefficient beta (closest to verl: kl_loss_coef).")

    # ----- lengths -----
    parser.add_argument("--grpo_max_prompt_length", type=int, default=3072,
                        help="Max prompt length (tokens).")
    parser.add_argument("--grpo_max_completion_length", type=int, default=3072,
                        help="Max completion length / max_new_tokens.")

    # ----- generation sampling -----
    parser.add_argument("--grpo_temperature", type=float, default=0.7,
                        help="Sampling temperature for GRPO rollouts.")
    parser.add_argument("--grpo_top_p", type=float, default=0.9,
                        help="Top-p nucleus sampling for GRPO rollouts.")
    parser.add_argument("--sample_batch_size", default=4)

    # ----- logging / checkpointing -----
    # # Training length (choose one style: steps or epochs; you currently set both)
    # parser.add_argument("--grpo_steps_per_iter", type=int, default=200,
    #                     help="Max training steps for GRPO in each outer iteration.")
    parser.add_argument("--grpo_logging_steps", type=int, default=2,
                        help="Log every N steps.")
    parser.add_argument("--grpo_save_steps", type=int, default=2,
                        help="Save every N steps.")
    parser.add_argument("--grpo_save_total_limit", default=1)

    # ----- memory / distributed -----
    parser.add_argument("--grpo_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (verl: enable_gradient_checkpointing=True).")

    # FSDP (approx alignable)
    parser.add_argument("--grpo_use_fsdp", action="store_true",
                        help="Enable FSDP in Trainer.")
    
    # hf fsdp
    # parser.add_argument("--grpo_fsdp", type=str, default="full_shard auto_wrap",
    #                     help='FSDP mode string, e.g. "full_shard auto_wrap".')
    # parser.add_argument("--grpo_fsdp_layer_cls", type=str, default="Qwen3DecoderLayer",
    #                     help="Transformer layer class to wrap for auto_wrap "
    #                          "(e.g., Qwen2DecoderLayer or Qwen3DecoderLayer).")
    # parser.add_argument("--grpo_fsdp_offload_params", action="store_true",
    #                     help="FSDP param CPU offload (closest to verl: fsdp_config.param_offload=True).")
    # parser.add_argument("--grpo_fsdp_state_dict_type", type=str, default="FULL_STATE_DICT",
    #                     choices=["FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"],
    #                     help="FSDP state dict type for saving. FULL is easiest for vLLM + reload.")
    # parser.add_argument("--grpo_fsdp_use_orig_params", action="store_true",
    #                     help="Set FSDP use_orig_params=True (recommended on newer PyTorch).")

    # deepspeed
    parser.add_argument("--ds_cfg", 
                        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/ds_zero3.json")

    return parser


# Example usage:
# parser = argparse.ArgumentParser()
# add_grpo_args(parser)
# cfg = parser.parse_args()


# grpo training pipeline
# for grpo training, using accelerate launch to initiate
# for sampling + labeling, using python -m and setting --grpo_sample

def grpo_trainer(grpo_cfg):
    parser = argparse.ArgumentParser()
    add_grpo_args(parser=parser)
    grpo_cfg = parser.parse_args()
    
    # basic setting
    step = grpo_cfg.step
    n_samples = grpo_cfg.n_samples
    n_grpo_samples = grpo_cfg.n_grpo_training_samples

    output_dir = f"{grpo_cfg.grpo_output_dir}/{grpo_cfg.labeling_method}/s{step}"
    model_name = f"{grpo_cfg.grpo_output_dir}/{grpo_cfg.labeling_method}/s{step}/grpo"
    grpo_cfg.grpo_output_dir = model_name

    # grpo training
    # load data 
    print(f"Current step: {step}, labeling_method: {grpo_cfg.labeling_method} grpo_sample: {grpo_cfg.grpo_sample}\n")
    print(f"\nCurrent dataset: {grpo_cfg.dataset}, train type: {grpo_cfg.train_type}\n")
    if not grpo_cfg.grpo_sample:
        
        ds_ = load_data(dataset=grpo_cfg.dataset)[step * (n_grpo_samples+n_samples): step * (n_grpo_samples+n_samples) + n_grpo_samples]
        ds = Dataset.from_list(ds_)

        if os.path.exists(model_name+'/model-00001-of-00002.safetensors'):
            print(f"Model dir {model_name} already exists, skip training.")
        else:
            print(f"Start GRPO training for step {step}...")
            grpo_train(cfg=grpo_cfg, ds=ds)

    # sampling responses
    else: 
        ds = load_data(dataset=grpo_cfg.dataset)[step * (n_grpo_samples+n_samples) + n_grpo_samples: (step+1) * (n_grpo_samples+n_samples)]
        ds = Dataset.from_list(ds)

        # sample_batch_size = 4
        # sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        if os.path.exists(os.path.join(output_dir, 'grpo_samples', 'samples.json')):
            print(f"Samples already exist for step {step}, skip sampling.\n\n")
            with open(os.path.join(output_dir, 'grpo_samples', 'samples.json'), 'r') as f:
                samples = json.load(f)
        else:
            print(f"Start sampling responses for step {step}...\n\n")
            sample_batch_size = 4
            samples = sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        # # labeling responses with gradient method


        if grpo_cfg.labeling_method == "gradient":
            print(f"Start gradient labeling for step {step}...\n\n")

            if os.path.exists(os.path.join(output_dir, 'gradient', 'dpo_trainset.json')):
                print(f"Gradient labeling already exists for step {step}, skip labeling.\n\n")
                # with open(os.path.join(output_dir, 'gradient', 'dpo_trainset.json'), 'r') as f:
                #     trainset = json.load(f)
            else:
                trainset = gradient_labeling(output_dir=output_dir, ds=samples, train_type=grpo_cfg.train_type,
                                             method="cluster", cluster_method=grpo_cfg.cluster_method,
                                             cluster_semantics=grpo_cfg.cluster_semantics,
                                             baseline_km_centers=grpo_cfg.baseline_km_centers,
                                             baseline_true_cluster=grpo_cfg.baseline_true_cluster,
                                             cluster_soft_prob_temp=grpo_cfg.cluster_soft_prob_temp,
                                             cluster_soft_prob_n_runs=grpo_cfg.cluster_soft_prob_n_runs,
                                             gpt_eval_prompt=grpo_cfg.cluster_gpt_eval_prompt)
        elif grpo_cfg.labeling_method == "trace":        
            # trace method for labeling
            print(f"Start trace labeling for step {step}...\n\n")

            if os.path.exists(os.path.join(output_dir, 'trace', 'dpo_trainset.json')):
                print(f"Trace labeling already exists for step {step}, skip labeling.\n\n")
                # with open(os.path.join(output_dir, 'trace', 'dpo_trainset.json'), 'r') as f:
                #     trainset = json.load(f)
            else:
                trainset = trace_labeling(current_dir=output_dir, ds=samples, train_type=grpo_cfg.train_type, threshold=grpo_cfg.trace_threshold)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_grpo_args(parser=parser)
    grpo_cfg = parser.parse_args()
    
    # basic setting
    step = grpo_cfg.step
    n_samples = grpo_cfg.n_samples
    n_grpo_samples = grpo_cfg.n_grpo_training_samples

    output_dir = f"{grpo_cfg.grpo_output_dir}/{grpo_cfg.labeling_method}/s{step}"
    model_name = f"{grpo_cfg.grpo_output_dir}/{grpo_cfg.labeling_method}/s{step}/grpo"
    grpo_cfg.grpo_output_dir = model_name
    
    if step == 0:
        grpo_cfg.grpo_input_model = grpo_cfg.grpo_init_model


    # grpo training
    # load data 
    print(f"Current step: {step}, labeling_method: {grpo_cfg.labeling_method} grpo_sample: {grpo_cfg.grpo_sample}\n")
    print(f"\nCurrent dataset: {grpo_cfg.dataset}, train type: {grpo_cfg.train_type}, grpo input model: {grpo_cfg.grpo_input_model}\n")
    if not grpo_cfg.grpo_sample:
        
        ds_ = load_data(dataset=grpo_cfg.dataset)[step * (n_grpo_samples+n_samples): step * (n_grpo_samples+n_samples) + n_grpo_samples]
        ds = Dataset.from_list(ds_)

        if os.path.exists(model_name+'/model-00001-of-00002.safetensors'):
            print(f"Model dir {model_name} already exists, skip training.\n")
        else:
            print(f"Start GRPO training for step {step}...\n\n")
            grpo_train(cfg=grpo_cfg, ds=ds)

    # sampling responses
    else: 
        ds = load_data(dataset=grpo_cfg.dataset)[step * (n_grpo_samples+n_samples) + n_grpo_samples: (step+1) * (n_grpo_samples+n_samples)]
        ds = Dataset.from_list(ds)

        # sample_batch_size = 4
        # sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        if os.path.exists(os.path.join(output_dir, 'grpo_samples', 'samples.json')):
            print(f"Samples already exist for step {step}, skip sampling.\n\n")
            with open(os.path.join(output_dir, 'grpo_samples', 'samples.json'), 'r') as f:
                samples = json.load(f)
        else:
            print(f"Start sampling responses for step {step}...\n\n")
            sample_batch_size = grpo_cfg.sample_batch_size
            samples = sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        # # labeling responses with gradient method

        if grpo_cfg.labeling_method == "gradient":
            print(f"Start gradient labeling for step {step}...\n\n")

            if os.path.exists(os.path.join(output_dir, 'gradient', 'dpo_trainset.json')):
                print(f"Gradient labeling already exists for step {step}, skip labeling.\n\n")
                # with open(os.path.join(output_dir, 'gradient', 'dpo_trainset.json'), 'r') as f:
                #     trainset = json.load(f)
            else:
                trainset = gradient_labeling(output_dir=output_dir, ds=samples, train_type=grpo_cfg.train_type,
                                             method="cluster", cluster_method=grpo_cfg.cluster_method,
                                             cluster_semantics=grpo_cfg.cluster_semantics,
                                             baseline_km_centers=grpo_cfg.baseline_km_centers,
                                             baseline_true_cluster=grpo_cfg.baseline_true_cluster,
                                             cluster_soft_prob_temp=grpo_cfg.cluster_soft_prob_temp,
                                             cluster_soft_prob_n_runs=grpo_cfg.cluster_soft_prob_n_runs,
                                             gpt_eval_prompt=grpo_cfg.cluster_gpt_eval_prompt)
        elif grpo_cfg.labeling_method == "trace":        
            # trace method for labeling
            print(f"Start trace labeling for step {step}...\n\n")

            if os.path.exists(os.path.join(output_dir, 'trace', 'dpo_trainset.json')):
                print(f"Trace labeling already exists for step {step}, skip labeling.\n\n")
                # with open(os.path.join(output_dir, 'trace', 'dpo_trainset.json'), 'r') as f:
                #     trainset = json.load(f)
            else:
                trainset = trace_labeling(current_dir=output_dir, ds=samples, train_type=grpo_cfg.train_type, threshold=grpo_cfg.trace_threshold)
