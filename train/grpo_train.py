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
from train.labeling import (
    gradient_analysis,
    trace_analysis,
    create_trace_runtime,
    close_trace_runtime,
)
from train.global_rollout_labeling import (
    flatten_rollout_records,
    build_rollout_score_sidecar,
    rank_rollouts_by_sample,
    build_sft_top1_trainset_from_ranked,
    build_dpo_trainset_from_ranked,
)
from train.load_data import load_data


random.seed(224)
SEED=224

MAX_TOKEN=3072

# bootstrap.py

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"\nFailed to load json {path}: {e}\n")
        return None


def _dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _sampling_partial_paths(save_dir):
    return (
        os.path.join(save_dir, "samples.partial.json"),
        os.path.join(save_dir, "all_samples.partial.json"),
    )


def _save_sampling_partial(save_dir, records, all_records, processed=0):
    samples_partial_path, all_partial_path = _sampling_partial_paths(save_dir)
    _dump_json(samples_partial_path, records)
    _dump_json(all_partial_path, all_records)
    # save intermediate
    all_samples_path = all_partial_path.replace(".json", f'_idx{processed}.json')
    _dump_json(all_samples_path, all_records)


def _load_sampling_partial(save_dir, ds_len):
    samples_partial_path, all_partial_path = _sampling_partial_paths(save_dir)
    records = _load_json_if_exists(samples_partial_path)
    all_records = _load_json_if_exists(all_partial_path)

    if not isinstance(all_records, list):
        return [], [], 0
    if len(all_records) > ds_len:
        print(f"\nSampling partial progress length {len(all_records)} > current ds length {ds_len}; ignoring partial files.\n")
        return [], [], 0

    if not isinstance(records, list):
        records = []

    return records, all_records, len(all_records)


def _label_resume_tag(save_every_items):
    if save_every_items is None:
        return None
    if int(save_every_items) <= 0:
        return "save_every_off"
    return f"save_every_{int(save_every_items)}"


def _label_resume_paths(save_dir, final_name, save_every_items=None):
    tag = _label_resume_tag(save_every_items)
    if tag is None:
        partial_name = final_name.replace(".json", ".partial.json")
        meta_name = final_name.replace(".json", ".resume.json")
    else:
        partial_name = final_name.replace(".json", f".{tag}.partial.json")
        meta_name = final_name.replace(".json", f".{tag}.resume.json")
    return (
        os.path.join(save_dir, final_name),
        os.path.join(save_dir, partial_name),
        os.path.join(save_dir, meta_name),
    )


def _load_label_partial(save_dir, final_name, total_len, save_every_items=None):
    final_path, tagged_partial_path, tagged_meta_path = _label_resume_paths(
        save_dir, final_name, save_every_items=save_every_items
    )
    if os.path.exists(final_path):
        data = _load_json_if_exists(final_path)
        if isinstance(data, list):
            return {"done": True, "next_index": total_len, "trainset": data}

    candidate_paths = [(tagged_partial_path, tagged_meta_path)]
    _, legacy_partial_path, legacy_meta_path = _label_resume_paths(save_dir, final_name)
    if (legacy_partial_path, legacy_meta_path) != (tagged_partial_path, tagged_meta_path):
        candidate_paths.append((legacy_partial_path, legacy_meta_path))

    partial_path = None
    meta = None
    partial = None
    for cand_partial_path, cand_meta_path in candidate_paths:
        cand_meta = _load_json_if_exists(cand_meta_path)
        cand_partial = _load_json_if_exists(cand_partial_path)
        if isinstance(cand_meta, dict) and isinstance(cand_partial, list):
            partial_path = cand_partial_path
            meta = cand_meta
            partial = cand_partial
            break

    if not isinstance(meta, dict) or not isinstance(partial, list):
        return {"done": False, "next_index": 0, "trainset": []}

    next_index = int(meta.get("next_index", 0))
    if next_index < 0 or next_index > total_len:
        print(f"\nInvalid labeling resume next_index={next_index} for total={total_len}; restarting labeling.\n")
        return {"done": False, "next_index": 0, "trainset": []}

    print(f"\nResuming labeling from index {next_index} / {total_len} using {partial_path}\n")
    return {"done": False, "next_index": next_index, "trainset": partial}


def _save_label_partial(save_dir, final_name, trainset, next_index, total_len, save_every_items=None):
    _, partial_path, meta_path = _label_resume_paths(save_dir, final_name, save_every_items=save_every_items)
    _dump_json(partial_path, trainset)
    _dump_json(meta_path, {"next_index": int(next_index), "total_len": int(total_len)})


def _save_label_final(save_dir, final_name, trainset):
    final_path, _, _ = _label_resume_paths(save_dir, final_name)
    _dump_json(final_path, trainset)


def _maybe_save_label_partial(save_dir, final_name, trainset, processed, total_len, save_every_items):
    if save_every_items > 0 and (processed % save_every_items == 0 or processed == total_len):
        _save_label_partial(
            save_dir,
            final_name,
            trainset,
            processed,
            total_len,
            save_every_items=save_every_items,
        )


def sample_responses(
    model_name,
    ds: List[str],
    output_dir,
    k: int=8,
    k_max: int=32,
    batch_size=16,
    temp: float=0.7,
    top_p: float=0.9,
    max_token: int=MAX_TOKEN,
    resume: bool=True,
    save_every_prompts: int=8,
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

    final_samples_path = os.path.join(save_dir, 'samples.json')
    final_all_samples_path = os.path.join(save_dir, 'all_samples.json')
    if resume and os.path.exists(final_samples_path) and os.path.exists(final_all_samples_path):
        loaded = _load_json_if_exists(final_samples_path)
        if isinstance(loaded, list):
            print(f"\nFound completed sampling outputs in {save_dir}; loading existing samples.\n")
            return loaded

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
    start_i = 0

    if resume:
        records, all_records, start_i = _load_sampling_partial(save_dir, ds_len)
        if start_i > 0:
            print(f"\nResuming sampling from prompt index {start_i} / {ds_len}\n")

    if ds_len == 0:
        _dump_json(os.path.join(save_dir, 'samples.json'), [])
        _dump_json(os.path.join(save_dir, 'all_samples.json'), [])
        return []

    for i in tqdm(range(start_i, ds_len), desc="Sampling responses"):
        example = ds[i]
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
        
        processed = i + 1
        if save_every_prompts > 0 and (processed % save_every_prompts == 0 or processed == ds_len):
            print(f"Saving intermediate sampling results for {i}-th prompt...")
            _save_sampling_partial(save_dir, records, all_records, processed=processed)

    
    save_path = os.path.join(save_dir, f'samples.json')
    with open(save_path, 'w') as f:
        json.dump(records, f, indent=4)
    
    save_path = os.path.join(save_dir, f'all_samples.json')
    with open(save_path, 'w') as f:
        json.dump(all_records, f, indent=4)

    samples_partial_path, all_partial_path = _sampling_partial_paths(save_dir)
    for p in [samples_partial_path, all_partial_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print(f"\nFailed to remove sampling partial file {p}: {e}\n")

    return records


def _label_trainset_filename(train_type: str):
    return "dpo_trainset.json" if train_type == "dpo" else "sft_trainset.json"


def _sampling_outputs_complete(output_dir, expected_len):
    samples_path = os.path.join(output_dir, 'grpo_samples', 'samples.json')
    all_samples_path = os.path.join(output_dir, 'grpo_samples', 'all_samples.json')
    if not (os.path.exists(samples_path) and os.path.exists(all_samples_path)):
        return False
    all_samples = _load_json_if_exists(all_samples_path)
    return isinstance(all_samples, list) and len(all_samples) == expected_len


def _load_cached_gradient_matrix(gradient_path: str) -> torch.Tensor:
    if not os.path.exists(gradient_path):
        raise FileNotFoundError(f"Gradient cache not found: {gradient_path}")
    with open(gradient_path, "rb") as f:
        payload = torch.load(f, map_location="cpu")

    sketches = payload["sketches"] if isinstance(payload, dict) and "sketches" in payload else payload
    if isinstance(sketches, torch.Tensor):
        grad = sketches.to(dtype=torch.float32, device="cpu")
    else:
        grad = torch.as_tensor(sketches, dtype=torch.float32, device="cpu")

    if grad.ndim != 2:
        raise ValueError(f"Expected 2D gradient matrix, got shape={tuple(grad.shape)}")
    return grad


def _nearest_neighbor_transfer_scores(grad: torch.Tensor) -> List[float]:
    if grad.shape[0] < 2:
        raise ValueError("Need at least 2 rollouts to compute nearest-neighbor transfer scores.")
    grad_n = torch.nn.functional.normalize(grad, p=2, dim=1)
    dmat = torch.cdist(grad_n, grad_n, p=2)
    dmat.fill_diagonal_(float("inf"))
    scores = torch.min(dmat, dim=1).values
    return [float(s) for s in scores]


def trace_labeling(current_dir, model_name=None, ds=None, train_type="dpo", threshold=0.2904761904761905,
                   resume: bool=True, save_every_items: int=16):
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
        final_name = 'dpo_trainset.json'
        start_i = 0
        if resume:
            state = _load_label_partial(save_dir, final_name, len(ds), save_every_items=save_every_items)
            if state["done"]:
                print(f"\nTrace labeling already completed; loading {final_name}\n")
                return state["trainset"]
            start_i = state["next_index"]
            trainset = state["trainset"]

        trace_llm = None
        trace_sampler = None
        try:
            if start_i < len(ds):
                trace_llm, trace_sampler = create_trace_runtime(
                    model_name=model_name,
                    n_gpu=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.85,
                    max_token=2048,
                    temperature=0.0,
                )

            for i in range(start_i, len(ds)):
                record = ds[i]
                prompt = record['prompt']
                label = record['label']
                gen = record['gen']
                
                # if correct responses < 2
                if len(gen) < 2:
                    _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                    continue

                processed_gen = [{"prompt": prompt, "gen": g, "label": label} for g in gen]
                
                # os.makedirs(f'{save_dir}/idx{i}', exist_ok=True)
                pred, trace_score = trace_analysis(processed_gen, model_name, 
                                      save_dir=f'{save_dir}/idx{i}', threshold=threshold, saving=True,
                                      trace_llm=trace_llm, trace_sampler=trace_sampler,
                                      return_trace_score=True)
                # print(pred)
                # based on predicion 1/0, get index and split gen into true set and false set
                

                chosen_idx = max(range(len(trace_score)), key=lambda idx: trace_score[idx])
                rejected_idx = min(range(len(trace_score)), key=lambda idx: trace_score[idx])
                sample = {"prompt": prompt, "chosen": gen[chosen_idx], "rejected": gen[rejected_idx], "label": label}
                trainset.append(sample)
                
                # true_idx = [i for i, p in enumerate(pred) if p == 1]
                # false_idx = [i for i, p in enumerate(pred) if p == 0]
                # if true_idx and false_idx:
                #     # for dpo, only get 1 pair of pos and neg samples
                #     sample = {"prompt": prompt, "chosen": gen[true_idx[0]], "rejected": gen[false_idx[0]], "label": label}
                #     trainset.append(sample)

                _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                print(f'\n\nSaving trace intermediate results, idx={i+1}...\n\n')
        finally:
            close_trace_runtime(trace_llm)

        _save_label_final(save_dir, final_name, trainset)

        return trainset
    
    elif train_type == "sft":
        pred, _ = trace_analysis(ds, model_name, save_dir=f'{save_dir}', 
                              threshold=threshold, saving=True, return_trace_score=True)
        
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

    elif train_type == "sft_top1":
        final_name = 'sft_trainset.json'
        start_i = 0
        if resume:
            state = _load_label_partial(save_dir, final_name, len(ds), save_every_items=save_every_items)
            if state["done"]:
                print(f"\nTrace labeling already completed; loading {final_name}\n")
                return state["trainset"]
            start_i = state["next_index"]
            trainset = state["trainset"]

        trace_llm = None
        trace_sampler = None
        try:
            if start_i < len(ds):
                trace_llm, trace_sampler = create_trace_runtime(
                    model_name=model_name,
                    n_gpu=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.85,
                    max_token=2048,
                    temperature=0.0,
                )

            for i in range(start_i, len(ds)):
                record = ds[i]
                prompt = record['prompt']
                label = record['label']
                gen = record['gen']

                if len(gen) == 0:
                    _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                    continue

                processed_gen = [{"prompt": prompt, "gen": g, "label": label} for g in gen]
                save_dir_i = f'{save_dir}/idx{i}'
                os.makedirs(save_dir_i, exist_ok=True)
                pred, _ = trace_analysis(
                    processed_gen, model_name, save_dir=save_dir_i, threshold=threshold,
                    trace_llm=trace_llm, trace_sampler=trace_sampler,
                    return_trace_score=True,
                )

                trace_scores = None
                trace_path = os.path.join(save_dir_i, 'trace.json')
                if os.path.exists(trace_path):
                    with open(trace_path, 'r') as f:
                        trace_scores = json.load(f).get('all_trace', None)

                true_idx = [j for j, p in enumerate(pred) if p == 1]
                if len(true_idx) == 0:
                    _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                    continue

                # Lower trace score means more positive according to the thresholding rule.
                if trace_scores is not None and len(trace_scores) == len(pred):
                    chosen_idx = min(true_idx, key=lambda j: trace_scores[j])
                else:
                    chosen_idx = true_idx[0]

                trainset.append({"instruction": prompt, "input": "", "output": gen[chosen_idx]})

                _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
        finally:
            close_trace_runtime(trace_llm)

        _save_label_final(save_dir, final_name, trainset)

        return trainset

def gradient_labeling(output_dir, model_name=None, ds=None, train_type="dpo", 
                      get_gradient=True, load_pi=True,
                      method="cluster", cluster_method="kmeans",
                      cluster_scope="local",
                      cluster_semantics="majority",
                      baseline_km_centers=None,
                      baseline_true_cluster=1,
                      cluster_soft_prob_temp=1.0,
                      cluster_soft_prob_n_runs=8,
                      cluster_soft_prob_single_seed=False,
                      gpt_eval_prompt=None,
                      resume: bool=True, save_every_items: int=8,
                      sft_top_k: int=1000): 
    """
    
    output: training set based on train_type
    """
    print(
        f"\nCurrent gradient labeling method: {method}, "
        f"Cluster method: {cluster_method}, Cluster scope: {cluster_scope}, train type: {train_type}\n"
    )

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

    if cluster_scope == "two_stage_global" and train_type not in {"dpo", "sft_top1"}:
        raise ValueError("cluster_scope=two_stage_global is currently supported only for train_type=dpo or sft_top1.")
    if cluster_scope == "transfer_local" and train_type != "sft_top1":
        raise ValueError("cluster_scope=transfer_local is currently supported only for train_type=sft_top1.")

    trainset = []
    if train_type == "sft_top1" and cluster_scope == "transfer_local":
        final_name = "sft_trainset.json"
        start_i = 0
        if resume:
            state = _load_label_partial(save_dir, final_name, len(ds), save_every_items=save_every_items)
            if state["done"]:
                print(f"\nGradient labeling already completed; loading {final_name}\n")
                return state["trainset"]
            start_i = state["next_index"]
            trainset = state["trainset"]

        if start_i != 0 or len(trainset) != 0:
            print("\ncluster_scope=transfer_local ignores partial SFT-top1 resume and recomputes all samples.\n")
            start_i = 0
            trainset = []

        score_rows = []
        selected_samples = []
        skipped_samples = []

        for i in range(len(ds)):
            record = ds[i]
            prompt = record.get("prompt", "")
            label = record.get("label", "")
            pid = record.get("pid", i)
            gen = record.get("gen", [])
            if isinstance(gen, str):
                gen = [gen]
            if not isinstance(gen, list):
                skipped_samples.append(
                    {"sample_idx": i, "pid": pid, "reason": "invalid_gen_type", "n_gen": 0}
                )
                continue

            valid_rollouts = [(j, g) for j, g in enumerate(gen) if isinstance(g, str)]
            if len(valid_rollouts) < 2:
                skipped_samples.append(
                    {"sample_idx": i, "pid": pid, "reason": "insufficient_rollouts", "n_gen": len(valid_rollouts)}
                )
                continue

            processed_gen = [{"input": prompt, "output": g} for _, g in valid_rollouts]
            local_dir = os.path.join(save_dir, f"idx{i}")
            os.makedirs(local_dir, exist_ok=True)

            try:
                gradient_analysis(
                    analyzer=analyzer,
                    model_name=model_name,
                    ds=processed_gen,
                    save_dir=local_dir,
                    method=method,
                    cluster_method=cluster_method,
                    baseline_km_centers=baseline_km_centers,
                    cluster_semantics="majority",
                    baseline_true_cluster=baseline_true_cluster,
                    gpt_eval_prompt=gpt_eval_prompt,
                    cluster_scope="local",
                    get_gradient=get_gradient,
                    load_pi=load_pi,
                    test_ratio=0.7,
                    use_pca=False,
                    use_svd=False,
                    use_t_sne=False,
                    normalized=True,
                    return_detail=False,
                    soft_prob_temp=cluster_soft_prob_temp,
                    soft_prob_n_runs=cluster_soft_prob_n_runs,
                    soft_prob_single_seed=cluster_soft_prob_single_seed,
                )
            except Exception as e:
                skipped_samples.append(
                    {
                        "sample_idx": i,
                        "pid": pid,
                        "reason": "local_gradient_analysis_failed",
                        "error": str(e),
                        "n_gen": len(valid_rollouts),
                    }
                )
                continue

            gradient_path = os.path.join(local_dir, "ds_gradient")
            try:
                grad = _load_cached_gradient_matrix(gradient_path)
            except Exception as e:
                skipped_samples.append(
                    {
                        "sample_idx": i,
                        "pid": pid,
                        "reason": "load_gradient_failed",
                        "error": str(e),
                        "n_gen": len(valid_rollouts),
                    }
                )
                continue

            if grad.shape[0] != len(valid_rollouts):
                skipped_samples.append(
                    {
                        "sample_idx": i,
                        "pid": pid,
                        "reason": "gradient_rollout_length_mismatch",
                        "gradient_rows": int(grad.shape[0]),
                        "n_gen": len(valid_rollouts),
                    }
                )
                continue

            try:
                transfer_scores = _nearest_neighbor_transfer_scores(grad)
            except Exception as e:
                skipped_samples.append(
                    {
                        "sample_idx": i,
                        "pid": pid,
                        "reason": "transfer_score_failed",
                        "error": str(e),
                        "n_gen": len(valid_rollouts),
                    }
                )
                continue

            low_local_idx = min(range(len(transfer_scores)), key=lambda idx: transfer_scores[idx])
            high_local_idx = max(range(len(transfer_scores)), key=lambda idx: transfer_scores[idx])

            for local_idx, (rollout_idx, output) in enumerate(valid_rollouts):
                score_rows.append(
                    {
                        "pid": pid,
                        "sample_idx": i,
                        "rollout_idx": int(rollout_idx),
                        "prompt": prompt,
                        "label": label,
                        "output": output,
                        "transfer_score": float(transfer_scores[local_idx]),
                    }
                )

            low_rollout_idx, low_output = valid_rollouts[low_local_idx]
            high_rollout_idx, high_output = valid_rollouts[high_local_idx]
            selected_samples.append(
                {
                    "pid": pid,
                    "sample_idx": i,
                    "prompt": prompt,
                    "label": label,
                    "n_valid_rollouts": len(valid_rollouts),
                    "lowest_transfer": {
                        "rollout_idx": int(low_rollout_idx),
                        "output": low_output,
                        "transfer_score": float(transfer_scores[low_local_idx]),
                    },
                    "highest_transfer": {
                        "rollout_idx": int(high_rollout_idx),
                        "output": high_output,
                        "transfer_score": float(transfer_scores[high_local_idx]),
                    },
                }
            )

        _dump_json(os.path.join(save_dir, "transfer_rollout_scores.json"), score_rows)
        _dump_json(
            os.path.join(save_dir, "transfer_selected_low_high_per_sample.json"),
            {
                "selected_samples": selected_samples,
                "skipped_samples": skipped_samples,
                "n_selected_samples": len(selected_samples),
                "n_skipped_samples": len(skipped_samples),
            },
        )

        trainset = []
        for row in selected_samples:
            lowest = row["lowest_transfer"]
            trainset.append(
                {
                    "instruction": row.get("prompt", ""),
                    "input": "",
                    "output": lowest["output"],
                    "transfer_score": float(lowest["transfer_score"]),
                    "pid": row.get("pid", row.get("sample_idx", 0)),
                    "rollout_idx": int(lowest["rollout_idx"]),
                }
            )

        print(
            f"\nTransfer-local SFT-top1 labeling complete: {len(trainset)} samples selected, "
            f"{len(skipped_samples)} samples skipped.\n"
        )
        _save_label_final(save_dir, final_name, trainset)
        return trainset

    if train_type == "dpo":
        final_name = 'dpo_trainset.json'
        start_i = 0
        if resume:
            state = _load_label_partial(save_dir, final_name, len(ds), save_every_items=save_every_items)
            if state["done"]:
                print(f"\nGradient labeling already completed; loading {final_name}\n")
                return state["trainset"]
            start_i = state["next_index"]
            trainset = state["trainset"]

        if cluster_scope == "two_stage_global":
            if method != "cluster" or cluster_method != "soft_prob":
                raise ValueError(
                    "cluster_scope=two_stage_global currently supports only method=cluster and cluster_method=soft_prob."
                )

            if start_i != 0 or len(trainset) != 0:
                print("\ncluster_scope=two_stage_global ignores partial DPO resume and recomputes all samples.\n")
                start_i = 0
                trainset = []

            selected_rollouts = []
            skipped_samples = []
            for i in range(len(ds)):
                record = ds[i]
                prompt = record.get("prompt", "")
                label = record.get("label", "")
                pid = record.get("pid", i)
                gen = record.get("gen", [])
                if isinstance(gen, str):
                    gen = [gen]
                if not isinstance(gen, list):
                    skipped_samples.append(
                        {"sample_idx": i, "pid": pid, "reason": "invalid_gen_type", "n_gen": 0}
                    )
                    continue

                valid_rollouts = [(j, g) for j, g in enumerate(gen) if isinstance(g, str)]
                if len(valid_rollouts) < 2:
                    skipped_samples.append(
                        {"sample_idx": i, "pid": pid, "reason": "insufficient_rollouts", "n_gen": len(valid_rollouts)}
                    )
                    continue

                processed_gen = [{"input": prompt, "output": g} for _, g in valid_rollouts]
                local_dir = os.path.join(save_dir, f"idx{i}")
                os.makedirs(local_dir, exist_ok=True)
                local_res = gradient_analysis(
                    analyzer=analyzer,
                    model_name=model_name,
                    ds=processed_gen,
                    save_dir=local_dir,
                    method="cluster",
                    cluster_method=cluster_method,
                    baseline_km_centers=baseline_km_centers,
                    cluster_semantics="majority",
                    baseline_true_cluster=baseline_true_cluster,
                    gpt_eval_prompt=gpt_eval_prompt,
                    cluster_scope="local",
                    get_gradient=get_gradient,
                    load_pi=load_pi,
                    test_ratio=0.7,
                    use_pca=False,
                    use_svd=False,
                    use_t_sne=False,
                    normalized=True,
                    return_detail=True,
                    soft_prob_temp=cluster_soft_prob_temp,
                    soft_prob_n_runs=cluster_soft_prob_n_runs,
                    soft_prob_single_seed=cluster_soft_prob_single_seed,
                )
                if local_res is None or not isinstance(local_res, dict):
                    skipped_samples.append(
                        {"sample_idx": i, "pid": pid, "reason": "local_clustering_failed", "n_gen": len(valid_rollouts)}
                    )
                    continue

                pred = local_res.get("result", [])
                true_probs = local_res.get("true_probs", [])
                if len(pred) != len(valid_rollouts) or len(true_probs) != len(valid_rollouts):
                    skipped_samples.append(
                        {
                            "sample_idx": i,
                            "pid": pid,
                            "reason": "local_result_length_mismatch",
                            "pred_len": len(pred),
                            "true_prob_len": len(true_probs),
                            "n_gen": len(valid_rollouts),
                        }
                    )
                    continue

                true_idx = [j for j, p in enumerate(pred) if int(p) == 1]
                false_idx = [j for j, p in enumerate(pred) if int(p) == 0]
                if len(true_idx) == 0 or len(false_idx) == 0:
                    skipped_samples.append(
                        {
                            "sample_idx": i,
                            "pid": pid,
                            "reason": "missing_local_true_or_false_cluster",
                            "n_true": len(true_idx),
                            "n_false": len(false_idx),
                        }
                    )
                    continue

                pos_local_idx = max(true_idx, key=lambda idx: float(true_probs[idx]))
                neg_local_idx = min(false_idx, key=lambda idx: float(true_probs[idx]))
                pos_rollout_idx, pos_output = valid_rollouts[pos_local_idx]
                neg_rollout_idx, neg_output = valid_rollouts[neg_local_idx]

                selected_rollouts.append(
                    {
                        "pid": pid,
                        "sample_idx": i,
                        "rollout_idx": int(pos_rollout_idx),
                        "prompt": prompt,
                        "label": label,
                        "output": pos_output,
                        "local_role": "local_true",
                        "local_true_prob": float(true_probs[pos_local_idx]),
                    }
                )
                selected_rollouts.append(
                    {
                        "pid": pid,
                        "sample_idx": i,
                        "rollout_idx": int(neg_rollout_idx),
                        "prompt": prompt,
                        "label": label,
                        "output": neg_output,
                        "local_role": "local_false",
                        "local_true_prob": float(true_probs[neg_local_idx]),
                    }
                )

            _dump_json(
                os.path.join(save_dir, "two_stage_selected_rollouts.json"),
                {
                    "selected_rollouts": selected_rollouts,
                    "skipped_samples": skipped_samples,
                    "n_selected_rollouts": len(selected_rollouts),
                    "n_selected_samples": len(selected_rollouts) // 2,
                    "n_skipped_samples": len(skipped_samples),
                },
            )

            if len(selected_rollouts) == 0:
                print("\nTwo-stage global labeling selected no rollouts. Writing empty DPO trainset.\n")
                _dump_json(os.path.join(save_dir, "two_stage_global_rollout_scores.json"), [])
                _save_label_final(save_dir, final_name, [])
                return []

            stage2_ds = [{"input": row["prompt"], "output": row["output"]} for row in selected_rollouts]
            stage2_meta = [
                {
                    "pid": row["pid"],
                    "sample_idx": row["sample_idx"],
                    "rollout_idx": row["rollout_idx"],
                    "prompt": row["prompt"],
                    "label": row["label"],
                    "output": row["output"],
                    "local_role": row["local_role"],
                    "local_true_prob": row["local_true_prob"],
                }
                for row in selected_rollouts
            ]

            two_stage_global_dir = os.path.join(save_dir, "two_stage_global")
            os.makedirs(two_stage_global_dir, exist_ok=True)
            cluster_res = gradient_analysis(
                analyzer=analyzer,
                model_name=model_name,
                ds=stage2_ds,
                save_dir=two_stage_global_dir,
                method="cluster",
                cluster_method=cluster_method,
                baseline_km_centers=baseline_km_centers,
                cluster_semantics="gpt",
                baseline_true_cluster=baseline_true_cluster,
                gpt_eval_prompt=gpt_eval_prompt,
                cluster_scope="global",
                cluster_preprocess="residualize_response_length",
                get_gradient=get_gradient,
                load_pi=load_pi,
                test_ratio=0.7,
                use_pca=False,
                use_svd=False,
                use_t_sne=False,
                normalized=True,
                return_detail=True,
                soft_prob_temp=cluster_soft_prob_temp,
                soft_prob_n_runs=cluster_soft_prob_n_runs,
                soft_prob_single_seed=cluster_soft_prob_single_seed,
            )
            if cluster_res is None or not isinstance(cluster_res, dict):
                raise RuntimeError("Two-stage global soft_prob clustering returned no detailed result.")

            pred = cluster_res.get("result", [])
            true_probs = cluster_res.get("true_probs", [])
            cluster_labels = cluster_res.get("cluster_labels", [])
            sidecar_rows = build_rollout_score_sidecar(
                meta=stage2_meta,
                pred=pred,
                true_probs=true_probs,
                cluster_labels=cluster_labels,
            )
            _dump_json(os.path.join(save_dir, "two_stage_global_rollout_scores.json"), sidecar_rows)

            grouped = {}
            for row in sidecar_rows:
                sample_idx = int(row["sample_idx"])
                grouped.setdefault(sample_idx, []).append(row)

            trainset = []
            skipped_pairing = 0
            for sample_idx, rows in grouped.items():
                if len(rows) != 2:
                    skipped_pairing += 1
                    continue
                if int(rows[0]["cluster_label"]) == int(rows[1]["cluster_label"]):
                    skipped_pairing += 1
                    continue

                pos_rows = [r for r in rows if int(r["pred"]) == 1]
                neg_rows = [r for r in rows if int(r["pred"]) == 0]
                if len(pos_rows) != 1 or len(neg_rows) != 1:
                    skipped_pairing += 1
                    continue

                chosen = pos_rows[0]
                rejected = neg_rows[0]
                trainset.append(
                    {
                        "prompt": chosen.get("prompt", ds[sample_idx].get("prompt", "")),
                        "chosen": chosen["output"],
                        "rejected": rejected["output"],
                        "label": chosen.get("label", ds[sample_idx].get("label", "")),
                        "pid": chosen.get("pid", ds[sample_idx].get("pid", sample_idx)),
                        "chosen_true_prob": float(chosen["true_prob"]),
                        "rejected_true_prob": float(rejected["true_prob"]),
                        "chosen_rollout_idx": int(chosen["rollout_idx"]),
                        "rejected_rollout_idx": int(rejected["rollout_idx"]),
                    }
                )

            print(
                f"\nTwo-stage global DPO labeling complete: {len(trainset)} pairs built, "
                f"{skipped_pairing} samples skipped during final pairing.\n"
            )
            _save_label_final(save_dir, final_name, trainset)
            return trainset

        if cluster_scope == "global":
            if method != "cluster" or cluster_method != "soft_prob":
                raise ValueError(
                    "cluster_scope=global currently supports only method=cluster and cluster_method=soft_prob."
                )

            flattened, meta = flatten_rollout_records(ds)
            if len(flattened) == 0:
                print("\nNo rollouts found for global labeling. Writing empty DPO trainset.\n")
                _save_label_final(save_dir, final_name, [])
                _dump_json(os.path.join(save_dir, "global_rollout_scores.json"), [])
                return []

            global_dir = os.path.join(save_dir, "global")
            os.makedirs(global_dir, exist_ok=True)
            cluster_res = gradient_analysis(
                analyzer=analyzer,
                model_name=model_name,
                ds=flattened,
                save_dir=global_dir,
                method="cluster",
                cluster_method=cluster_method,
                baseline_km_centers=baseline_km_centers,
                cluster_semantics=cluster_semantics,
                baseline_true_cluster=baseline_true_cluster,
                gpt_eval_prompt=gpt_eval_prompt,
                cluster_scope=cluster_scope,
                get_gradient=get_gradient,
                load_pi=load_pi,
                test_ratio=0.7,
                use_pca=False,
                use_svd=False,
                use_t_sne=False,
                normalized=True,
                return_detail=True,
                soft_prob_temp=cluster_soft_prob_temp,
                soft_prob_n_runs=cluster_soft_prob_n_runs,
                soft_prob_single_seed=cluster_soft_prob_single_seed,
            )
            if cluster_res is None or not isinstance(cluster_res, dict):
                raise RuntimeError("Global soft_prob clustering returned no detailed result.")

            pred = cluster_res.get("result", [])
            true_probs = cluster_res.get("true_probs", [])
            cluster_labels = cluster_res.get("cluster_labels", [])
            sidecar_rows = build_rollout_score_sidecar(
                meta=meta,
                pred=pred,
                true_probs=true_probs,
                cluster_labels=cluster_labels,
            )
            _dump_json(os.path.join(save_dir, "global_rollout_scores.json"), sidecar_rows)

            ranked_by_sample = rank_rollouts_by_sample(sidecar_rows)
            trainset, skipped = build_dpo_trainset_from_ranked(
                ds=ds,
                ranked_by_sample=ranked_by_sample,
                true_prob_threshold=0.5,
            )
            print(
                f"\nGlobal DPO labeling complete: {len(trainset)} pairs built, "
                f"{skipped} samples skipped (missing true/false split or insufficient rollouts).\n"
            )
            _save_label_final(save_dir, final_name, trainset)
            return trainset

        for i in range(start_i, len(ds)):
            record = ds[i]
            prompt = record['prompt']
            label = record['label']
            gen = record['gen']

            # if no correct responses
            if len(gen) < 2:
                _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
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
                                                cluster_scope=cluster_scope,
                                                get_gradient=get_gradient, load_pi=load_pi,
                                                test_ratio=0.7, 
                                                use_pca=False, use_svd=False, use_t_sne=False, 
                                                normalized=True,
                                                return_detail=(cluster_method == "soft_prob"),
                                                soft_prob_temp=cluster_soft_prob_temp,
                                                soft_prob_n_runs=cluster_soft_prob_n_runs,
                                                soft_prob_single_seed=cluster_soft_prob_single_seed)
                if cluster_res is None:
                    _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
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

            _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)


        if len(trainset) == 0:
            print("\nNo valid DPO pairs found after clustering. Using skip policy (empty trainset).\n")

        _save_label_final(save_dir, final_name, trainset)
        
        return trainset

    elif train_type == "sft":

        processed_gen = [{"input": record['prompt'], "output": record['gen']} for record in ds]
        pred = []
        true_probs = []

        os.makedirs(f'{save_dir}', exist_ok=True)
        if method == "svm":
            pred = gradient_analysis(analyzer=analyzer, model_name=model_name,
                                    ds=processed_gen, save_dir=f'{save_dir}',
                                    method="svm",
                                    get_gradient=get_gradient, test_ratio=0.7,
                                    use_pca=False, use_svd=False, use_t_sne=False,
                                    normalized=True)
            true_probs = [float(int(p)) for p in pred]

        elif method == "cluster":
            if len(processed_gen) == 1:
                pred = [1]
                true_probs = [1.0]
            elif len(processed_gen) == 0:
                pred = []
                true_probs = []
            else:
                sft_cluster_method = "soft_prob" 
                sft_cluster_semantics = "gpt" if cluster_semantics == "majority" else cluster_semantics
                if sft_cluster_method != cluster_method:
                    print(f"\nSFT mode: overriding cluster_method={cluster_method} -> {sft_cluster_method}\n")
                if sft_cluster_semantics != cluster_semantics:
                    print(f"\nSFT mode: overriding cluster_semantics={cluster_semantics} -> {sft_cluster_semantics}\n")

                print(f'\n\nRunning gradient analysis on mode: {train_type}\n\n')
                cluster_res = gradient_analysis(analyzer=analyzer, model_name=model_name,
                                                ds=processed_gen, save_dir=f'{save_dir}',
                                                method="cluster", cluster_method=sft_cluster_method,
                                                baseline_km_centers=baseline_km_centers,
                                                cluster_semantics=sft_cluster_semantics,
                                                baseline_true_cluster=baseline_true_cluster,
                                                gpt_eval_prompt=gpt_eval_prompt,
                                                cluster_scope=cluster_scope,
                                                get_gradient=get_gradient, load_pi=load_pi,
                                                test_ratio=0.7,
                                                use_pca=False, use_svd=False, use_t_sne=False,
                                                normalized=True,
                                                return_detail=True,
                                                soft_prob_temp=cluster_soft_prob_temp,
                                                soft_prob_n_runs=cluster_soft_prob_n_runs,
                                                soft_prob_single_seed=cluster_soft_prob_single_seed)
                if cluster_res is None:
                    pred = []
                    true_probs = []
                elif isinstance(cluster_res, dict):
                    pred = cluster_res.get("result", [])
                    true_probs = cluster_res.get("true_probs", [])
                else:
                    pred = cluster_res
                    true_probs = [float(int(p)) for p in pred]
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

        if true_probs is None or len(true_probs) != len(pred):
            true_probs = [float(int(p)) for p in pred]

        ranked = [
            (idx, float(prob))
            for idx, (p, prob) in enumerate(zip(pred, true_probs))
            if int(p) == 1
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        if int(sft_top_k) > 0:
            ranked = ranked[:int(sft_top_k)]

        trainset = [
            {"instruction": ds[idx]['prompt'], "input": "", "output": ds[idx]['gen'], "true_prob": prob}
            for idx, prob in ranked
        ]

        # security code, should be removed
        if len(trainset) == 0 and len(ds) > 0:
            print("\n\nNo samples labeled as true, please check the threshold or the gradient analysis.\n\n")
            fallback_prob = float(true_probs[0]) if len(true_probs) > 0 else 1.0
            trainset.append({"instruction": ds[0]['prompt'], "input": "", "output": ds[0]['gen'], "true_prob": fallback_prob})

        with open(os.path.join(save_dir, 'sft_trainset.json'), 'w') as f:
            json.dump(trainset, f, indent=4)

        return trainset

    elif train_type == "sft_top1":
        final_name = 'sft_trainset.json'
        start_i = 0
        if resume:
            state = _load_label_partial(save_dir, final_name, len(ds), save_every_items=save_every_items)
            if state["done"]:
                print(f"\nGradient labeling already completed; loading {final_name}\n")
                return state["trainset"]
            start_i = state["next_index"]
            trainset = state["trainset"]

        if cluster_scope == "two_stage_global":
            if method != "cluster" or cluster_method != "soft_prob":
                raise ValueError(
                    "cluster_scope=two_stage_global currently supports only method=cluster and cluster_method=soft_prob."
                )

            if start_i != 0 or len(trainset) != 0:
                print("\ncluster_scope=two_stage_global ignores partial SFT-top1 resume and recomputes all samples.\n")
                start_i = 0
                trainset = []

            selected_rollouts = []
            skipped_samples = []
            for i in range(len(ds)):
                record = ds[i]
                prompt = record.get("prompt", "")
                label = record.get("label", "")
                pid = record.get("pid", i)
                gen = record.get("gen", [])
                if isinstance(gen, str):
                    gen = [gen]
                if not isinstance(gen, list):
                    skipped_samples.append(
                        {"sample_idx": i, "pid": pid, "reason": "invalid_gen_type", "n_gen": 0}
                    )
                    continue

                valid_rollouts = [(j, g) for j, g in enumerate(gen) if isinstance(g, str)]
                if len(valid_rollouts) < 2:
                    skipped_samples.append(
                        {"sample_idx": i, "pid": pid, "reason": "insufficient_rollouts", "n_gen": len(valid_rollouts)}
                    )
                    continue

                processed_gen = [{"input": prompt, "output": g} for _, g in valid_rollouts]
                local_dir = os.path.join(save_dir, f"idx{i}")
                os.makedirs(local_dir, exist_ok=True)
                local_res = gradient_analysis(
                    analyzer=analyzer,
                    model_name=model_name,
                    ds=processed_gen,
                    save_dir=local_dir,
                    method="cluster",
                    cluster_method=cluster_method,
                    baseline_km_centers=baseline_km_centers,
                    cluster_semantics="majority",
                    baseline_true_cluster=baseline_true_cluster,
                    gpt_eval_prompt=gpt_eval_prompt,
                    cluster_scope="local",
                    get_gradient=get_gradient,
                    load_pi=load_pi,
                    test_ratio=0.7,
                    use_pca=False,
                    use_svd=False,
                    use_t_sne=False,
                    normalized=True,
                    return_detail=True,
                    soft_prob_temp=cluster_soft_prob_temp,
                    soft_prob_n_runs=cluster_soft_prob_n_runs,
                    soft_prob_single_seed=cluster_soft_prob_single_seed,
                )
                if local_res is None or not isinstance(local_res, dict):
                    skipped_samples.append(
                        {"sample_idx": i, "pid": pid, "reason": "local_clustering_failed", "n_gen": len(valid_rollouts)}
                    )
                    continue

                pred = local_res.get("result", [])
                true_probs = local_res.get("true_probs", [])
                if len(pred) != len(valid_rollouts) or len(true_probs) != len(valid_rollouts):
                    skipped_samples.append(
                        {
                            "sample_idx": i,
                            "pid": pid,
                            "reason": "local_result_length_mismatch",
                            "pred_len": len(pred),
                            "true_prob_len": len(true_probs),
                            "n_gen": len(valid_rollouts),
                        }
                    )
                    continue

                true_idx = [j for j, p in enumerate(pred) if int(p) == 1]
                false_idx = [j for j, p in enumerate(pred) if int(p) == 0]
                if len(true_idx) == 0 or len(false_idx) == 0:
                    skipped_samples.append(
                        {
                            "sample_idx": i,
                            "pid": pid,
                            "reason": "missing_local_true_or_false_cluster",
                            "n_true": len(true_idx),
                            "n_false": len(false_idx),
                        }
                    )
                    continue

                pos_local_idx = max(true_idx, key=lambda idx: float(true_probs[idx]))
                neg_local_idx = min(false_idx, key=lambda idx: float(true_probs[idx]))
                pos_rollout_idx, pos_output = valid_rollouts[pos_local_idx]
                neg_rollout_idx, neg_output = valid_rollouts[neg_local_idx]

                selected_rollouts.append(
                    {
                        "pid": pid,
                        "sample_idx": i,
                        "rollout_idx": int(pos_rollout_idx),
                        "prompt": prompt,
                        "label": label,
                        "output": pos_output,
                        "local_role": "local_true",
                        "local_true_prob": float(true_probs[pos_local_idx]),
                    }
                )
                selected_rollouts.append(
                    {
                        "pid": pid,
                        "sample_idx": i,
                        "rollout_idx": int(neg_rollout_idx),
                        "prompt": prompt,
                        "label": label,
                        "output": neg_output,
                        "local_role": "local_false",
                        "local_true_prob": float(true_probs[neg_local_idx]),
                    }
                )

            _dump_json(
                os.path.join(save_dir, "two_stage_selected_rollouts.json"),
                {
                    "selected_rollouts": selected_rollouts,
                    "skipped_samples": skipped_samples,
                    "n_selected_rollouts": len(selected_rollouts),
                    "n_selected_samples": len(selected_rollouts) // 2,
                    "n_skipped_samples": len(skipped_samples),
                },
            )

            if len(selected_rollouts) == 0:
                print("\nTwo-stage global SFT-top1 labeling selected no rollouts. Writing empty SFT trainset.\n")
                _dump_json(os.path.join(save_dir, "two_stage_global_rollout_scores.json"), [])
                _save_label_final(save_dir, final_name, [])
                return []

            stage2_ds = [{"input": row["prompt"], "output": row["output"]} for row in selected_rollouts]
            stage2_meta = [
                {
                    "pid": row["pid"],
                    "sample_idx": row["sample_idx"],
                    "rollout_idx": row["rollout_idx"],
                    "prompt": row["prompt"],
                    "label": row["label"],
                    "output": row["output"],
                    "local_role": row["local_role"],
                    "local_true_prob": row["local_true_prob"],
                }
                for row in selected_rollouts
            ]

            two_stage_global_dir = os.path.join(save_dir, "two_stage_global")
            os.makedirs(two_stage_global_dir, exist_ok=True)
            cluster_res = gradient_analysis(
                analyzer=analyzer,
                model_name=model_name,
                ds=stage2_ds,
                save_dir=two_stage_global_dir,
                method="cluster",
                cluster_method=cluster_method,
                baseline_km_centers=baseline_km_centers,
                cluster_semantics="gpt",
                baseline_true_cluster=baseline_true_cluster,
                gpt_eval_prompt=gpt_eval_prompt,
                cluster_scope="global",
                cluster_preprocess="residualize_response_length",
                get_gradient=get_gradient,
                load_pi=load_pi,
                test_ratio=0.7,
                use_pca=False,
                use_svd=False,
                use_t_sne=False,
                normalized=True,
                return_detail=True,
                soft_prob_temp=cluster_soft_prob_temp,
                soft_prob_n_runs=cluster_soft_prob_n_runs,
                soft_prob_single_seed=cluster_soft_prob_single_seed,
            )
            if cluster_res is None or not isinstance(cluster_res, dict):
                raise RuntimeError("Two-stage global soft_prob clustering returned no detailed result.")

            pred = cluster_res.get("result", [])
            true_probs = cluster_res.get("true_probs", [])
            cluster_labels = cluster_res.get("cluster_labels", [])
            sidecar_rows = build_rollout_score_sidecar(
                meta=stage2_meta,
                pred=pred,
                true_probs=true_probs,
                cluster_labels=cluster_labels,
            )
            _dump_json(os.path.join(save_dir, "two_stage_global_rollout_scores.json"), sidecar_rows)

            best_pos_by_sample = {}
            for row in sidecar_rows:
                if int(row["pred"]) != 1:
                    continue
                sample_idx = int(row["sample_idx"])
                if (
                    sample_idx not in best_pos_by_sample
                    or float(row["true_prob"]) > float(best_pos_by_sample[sample_idx]["true_prob"])
                ):
                    best_pos_by_sample[sample_idx] = row

            trainset = []
            for sample_idx in sorted(best_pos_by_sample.keys()):
                row = best_pos_by_sample[sample_idx]
                trainset.append(
                    {
                        "instruction": row.get("prompt", ds[sample_idx].get("prompt", "")),
                        "input": "",
                        "output": row["output"],
                        "true_prob": float(row["true_prob"]),
                        "pid": row.get("pid", ds[sample_idx].get("pid", sample_idx)),
                        "rollout_idx": int(row["rollout_idx"]),
                    }
                )

            selected_sample_count = len({int(r["sample_idx"]) for r in selected_rollouts})
            skipped_no_positive = max(0, selected_sample_count - len(trainset))
            print(
                f"\nTwo-stage global SFT-top1 labeling complete: {len(trainset)} samples selected, "
                f"{skipped_no_positive} samples skipped (no positive rollout after global clustering).\n"
            )
            _save_label_final(save_dir, final_name, trainset)
            return trainset

        if cluster_scope == "global":
            if method != "cluster" or cluster_method != "soft_prob":
                raise ValueError(
                    "cluster_scope=global currently supports only method=cluster and cluster_method=soft_prob."
                )

            flattened, meta = flatten_rollout_records(ds)
            if len(flattened) == 0:
                print("\nNo rollouts found for global labeling. Writing empty SFT trainset.\n")
                _save_label_final(save_dir, final_name, [])
                _dump_json(os.path.join(save_dir, "global_rollout_scores.json"), [])
                return []

            global_dir = os.path.join(save_dir, "global")
            os.makedirs(global_dir, exist_ok=True)
            cluster_res = gradient_analysis(
                analyzer=analyzer,
                model_name=model_name,
                ds=flattened,
                save_dir=global_dir,
                method="cluster",
                cluster_method=cluster_method,
                baseline_km_centers=baseline_km_centers,
                cluster_semantics=cluster_semantics,
                baseline_true_cluster=baseline_true_cluster,
                gpt_eval_prompt=gpt_eval_prompt,
                cluster_scope=cluster_scope,
                get_gradient=get_gradient,
                load_pi=load_pi,
                test_ratio=0.7,
                use_pca=False,
                use_svd=False,
                use_t_sne=False,
                normalized=True,
                return_detail=True,
                soft_prob_temp=cluster_soft_prob_temp,
                soft_prob_n_runs=cluster_soft_prob_n_runs,
                soft_prob_single_seed=cluster_soft_prob_single_seed,
            )
            if cluster_res is None or not isinstance(cluster_res, dict):
                raise RuntimeError("Global soft_prob clustering returned no detailed result.")

            pred = cluster_res.get("result", [])
            true_probs = cluster_res.get("true_probs", [])
            cluster_labels = cluster_res.get("cluster_labels", [])
            sidecar_rows = build_rollout_score_sidecar(
                meta=meta,
                pred=pred,
                true_probs=true_probs,
                cluster_labels=cluster_labels,
            )
            _dump_json(os.path.join(save_dir, "global_rollout_scores.json"), sidecar_rows)

            ranked_by_sample = rank_rollouts_by_sample(sidecar_rows)
            trainset = build_sft_top1_trainset_from_ranked(ds=ds, ranked_by_sample=ranked_by_sample)
            print(
                f"\nGlobal SFT-top1 labeling complete: {len(trainset)} samples with selected rollouts.\n"
            )
            _save_label_final(save_dir, final_name, trainset)
            return trainset

        for i in range(start_i, len(ds)):
            record = ds[i]
            prompt = record['prompt']
            label = record['label']
            gen = record['gen']

            if len(gen) < 2:
                _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                continue

            processed_gen = [{"input": prompt, "output": g} for g in gen]
            true_probs = None

            os.makedirs(f'{save_dir}/idx{i}', exist_ok=True)
            if method == "svm":
                pred = gradient_analysis(analyzer=analyzer, model_name=model_name,
                                        ds=processed_gen, save_dir=f'{save_dir}/idx{i}',
                                        method="svm",
                                        get_gradient=get_gradient, test_ratio=0.7,
                                        use_pca=False, use_svd=False, use_t_sne=False,
                                        normalized=True)
            elif method == "cluster":
                cluster_res = gradient_analysis(analyzer=analyzer, model_name=model_name,
                                                ds=processed_gen, save_dir=f'{save_dir}/idx{i}',
                                                method="cluster", cluster_method=cluster_method,
                                                baseline_km_centers=baseline_km_centers,
                                                cluster_semantics=cluster_semantics,
                                                baseline_true_cluster=baseline_true_cluster,
                                                gpt_eval_prompt=gpt_eval_prompt,
                                                cluster_scope=cluster_scope,
                                                get_gradient=get_gradient, load_pi=load_pi,
                                                test_ratio=0.7,
                                                use_pca=False, use_svd=False, use_t_sne=False,
                                                normalized=True,
                                                return_detail=(cluster_method == "soft_prob"),
                                                soft_prob_temp=cluster_soft_prob_temp,
                                                soft_prob_n_runs=cluster_soft_prob_n_runs,
                                                soft_prob_single_seed=cluster_soft_prob_single_seed)
                if cluster_res is None:
                    _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                    continue
                if cluster_method == "soft_prob" and isinstance(cluster_res, dict):
                    pred = cluster_res["result"]
                    true_probs = cluster_res.get("true_probs", None)
                else:
                    pred = cluster_res
                    num_1 = sum(pred)
                    num_0 = len(pred) - num_1
                    if num_1 < num_0:
                        pred = [0 if p == 1 else 1 for p in pred]
            else:
                raise NotImplementedError(f"Method {method} not implemented.")

            true_idx = [j for j, p in enumerate(pred) if p == 1]
            if len(true_idx) == 0:
                _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)
                continue

            if true_probs is not None and len(true_probs) == len(pred):
                chosen_idx = max(true_idx, key=lambda j: true_probs[j])
            else:
                chosen_idx = true_idx[0]

            trainset.append({"instruction": prompt, "input": "", "output": gen[chosen_idx]})

            _maybe_save_label_partial(save_dir, final_name, trainset, i + 1, len(ds), save_every_items)

        _save_label_final(save_dir, final_name, trainset)

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
        vllm_gpu_memory_utilization=0.6,
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

    resume_from_checkpoint = cfg.grpo_resume_from_checkpoint or None
    if resume_from_checkpoint:
        print(f"Resuming GRPO from checkpoint: {resume_from_checkpoint}\n")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()



def add_grpo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    #env agrs
    parser.add_argument("--step", type=int, default=0, help="Current training step, used for naming and loading/saving.")
    parser.add_argument("--labeling_method", type=str, default="trace", help="Labeling method for reward hacking, e.g. trace or gradient.")
    parser.add_argument("--cluster_method", type=str, default="kmeans", help="Clustering method for gradient labeling, e.g. kmeans or soft_prob.")
    parser.add_argument("--cluster_scope", type=str, default="local", choices=["local", "global", "two_stage_global", "transfer_local"],
                        help="Cluster each sample separately (local), all rollouts together (global), run two-stage local->global clustering, or use local transfer-distance scoring.")
    parser.add_argument("--cluster_semantics", type=str, default="majority",
                        choices=["majority", "gpt", "gpt_top5_avg", "baseline_centroids", "rft_arlsat_gpt_top5"],
                        help="How to map cluster id to true/false labels.")
    parser.add_argument("--baseline_km_centers", type=str, default=None,
                        help="Path to baseline KMeans model for baseline-centroid semantic mapping.")
    parser.add_argument("--baseline_true_cluster", type=int, default=1,
                        help="Which baseline cluster index corresponds to true class.")
    parser.add_argument("--cluster_soft_prob_temp", type=float, default=1.0,
                        help="Temperature for soft probability from KMeans distances.")
    parser.add_argument("--cluster_soft_prob_n_runs", type=int, default=8,
                        help="Number of KMeans restarts (different seeds) for robust soft_prob.")
    parser.add_argument("--cluster_soft_prob_single_seed", action="store_false",
                        help="If set, run soft_prob KMeans only once with seed=224 (skip multi-seed selection).")
    parser.add_argument("--cluster_gpt_eval_prompt", type=str, default=None,
                        help="Optional prompt template for GPT semantic mapping with {question} and {response}.")
    parser.add_argument("--trace_threshold", type=float, default=0.2904761904761905, help="Threshold for trace labeling, only used when labeling_method is trace.")
    parser.add_argument("--train_type", type=str, default="dpo", help="Train type for reward hacking, e.g. dpo, sft, or sft_top1.")
    parser.add_argument("--dataset", type=str, default="arlsat", help="Dataset name for training, e.g. arlsat.")
    
    parser.add_argument("--load_pi", action="store_true")
    # special args
    parser.add_argument("--grpo_sample_only", action="store_true")
    parser.add_argument("--skip_sample", action="store_true")

    # model
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to use for each step (after excluding grpo training samples).")
    parser.add_argument("--n_grpo_training_samples", type=int, default=16, help="Number of samples to use for GRPO training in each step.")
    parser.add_argument("--grpo_sample", action="store_true", help="Whether to run sampling after GRPO training.")
    parser.add_argument("--sft_top1", action="store_true",
                        help="Label sampled rollouts into SFT trainset by keeping one best positive completion per prompt.")
    parser.add_argument("--sft_top_k", type=int, default=1000,
                        help="For train_type=sft gradient labeling, keep top-k positive samples by true_prob. <=0 keeps all positives.")
    parser.add_argument("--full_dataset_each_step", action="store_true",
                        help="When sampling/labeling, use the full dataset each step instead of n_samples slicing.")
    parser.add_argument("--sample_model_dir", type=str, default="",
                        help="If set with --grpo_sample, sample/label directly from this model path instead of <output>/.../grpo.")
    parser.add_argument("--grpo_init_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--grpo_input_model", default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s0/grpo", help="Model path for grpo training, used when step > 0.")

    # ----- IO / run control -----
    parser.add_argument("--grpo_output_dir", type=str, 
                        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s0/grpo",
                        help="Output directory for TRL GRPO checkpoints/logs.")
    parser.add_argument("--grpo_resume_from_checkpoint", type=str, default="",
                        help="Resume TRL GRPO training from this checkpoint directory.")
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
    parser.add_argument("--sample_batch_size", type=int, default=4)

    # ----- logging / checkpointing -----
    # # Training length (choose one style: steps or epochs; you currently set both)
    # parser.add_argument("--grpo_steps_per_iter", type=int, default=200,
    #                     help="Max training steps for GRPO in each outer iteration.")
    parser.add_argument("--grpo_logging_steps", type=int, default=2,
                        help="Log every N steps.")
    parser.add_argument("--grpo_save_steps", type=int, default=2,
                        help="Save every N steps.")
    parser.add_argument("--grpo_save_total_limit", type=int, default=1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_grpo_args(parser=parser)
    grpo_cfg = parser.parse_args()

    if grpo_cfg.sft_top1:
        grpo_cfg.train_type = "sft_top1"
    
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
        if not grpo_cfg.grpo_resume_from_checkpoint:
            latest_checkpoint = _find_latest_resumable_checkpoint(model_name)
            if latest_checkpoint:
                grpo_cfg.grpo_resume_from_checkpoint = latest_checkpoint
                print(f"Detected existing GRPO checkpoint, auto-resuming from {latest_checkpoint}.\n")

        if os.path.exists(model_name+'/model-00001-of-00002.safetensors'):
            print(f"Model dir {model_name} already exists, skip training.\n")
        else:
            print(f"Start GRPO training for step {step}...\n\n")
            grpo_train(cfg=grpo_cfg, ds=ds)

    # sampling responses
    else: 
        if grpo_cfg.full_dataset_each_step:
            ds = load_data(dataset=grpo_cfg.dataset)
            print(f"\nUsing full dataset for sampling/labeling at step {step}: {len(ds)} samples\n")

            if grpo_cfg.grpo_sample_only:
                if grpo_cfg.labeling_method == "sample-1000":
                    ds = ds[-1000: -800]
                elif grpo_cfg.labeling_method == "sample-800":
                    ds = ds[-800: -600]
                elif grpo_cfg.cluster_semantics == "baseline":
                    ds = ds[-600: -400]
                elif grpo_cfg.cluster_semantics == "gpt":
                    ds = ds[-400: -200]
                else:
                    ds = ds[-200: ]

        else:
            ds = load_data(dataset=grpo_cfg.dataset)[step * (n_grpo_samples+n_samples) + n_grpo_samples: (step+1) * (n_grpo_samples+n_samples)]
        ds = Dataset.from_list(ds)
        sample_model_name = grpo_cfg.sample_model_dir if grpo_cfg.sample_model_dir else model_name
        label_trainset_file = _label_trainset_filename(grpo_cfg.train_type)

        if grpo_cfg.grpo_sample_only:
            sample_batch_size = grpo_cfg.sample_batch_size
            print(f"Sampling from model: {sample_model_name}\n")
            samples = sample_responses(model_name=sample_model_name, ds=ds, 
                                       batch_size=sample_batch_size, 
                                       output_dir=output_dir, resume=True)
            
            input("\n\n\n\n\nfinished sampling:") 

            

        # sample_batch_size = 4
        # sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

        if grpo_cfg.skip_sample:
            save_path = os.path.join(output_dir, 'grpo_samples', f'samples.json')
            print(save_path)
            with open(save_path, 'r') as f:
                samples = json.load(f)
            print(f'\n\nLoaded samples len: {len(samples)}')
        else:
            if _sampling_outputs_complete(output_dir, len(ds)):
                print(f"Completed samples already exist for step {step}, loading cached sampling outputs.\n\n")
            else:
                print(f"Start/resume sampling responses for step {step}...\n\n")
            sample_batch_size = grpo_cfg.sample_batch_size
            print(f"Sampling from model: {sample_model_name}\n")
            samples = sample_responses(model_name=sample_model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir, resume=True)

        # # labeling responses with gradient method

        if grpo_cfg.train_type == "sft":
            grpo_cfg.load_pi = False
        else:
            grpo_cfg.load_pi = True


        if grpo_cfg.labeling_method == "gradient":
            print(f"Start gradient labeling for step {step}...\n\n")
            if grpo_cfg.train_type == "sft" and os.path.exists(os.path.join(output_dir, 'gradient', label_trainset_file)):
                print(f"Gradient labeling already exists for step {step}, skip labeling.\n\n")
            else:
                if os.path.exists(os.path.join(output_dir, 'gradient', label_trainset_file)):
                    print(f"Gradient labeling final file exists for step {step}; resume-aware labeling will validate/load it.\n\n")
                trainset = gradient_labeling(output_dir=output_dir, model_name=sample_model_name, ds=samples, train_type=grpo_cfg.train_type,
                                             method="cluster", cluster_method=grpo_cfg.cluster_method,
                                             cluster_scope=grpo_cfg.cluster_scope,
                                             cluster_semantics=grpo_cfg.cluster_semantics,
                                             baseline_km_centers=grpo_cfg.baseline_km_centers,
                                             baseline_true_cluster=grpo_cfg.baseline_true_cluster,
                                             cluster_soft_prob_temp=grpo_cfg.cluster_soft_prob_temp,
                                             cluster_soft_prob_n_runs=grpo_cfg.cluster_soft_prob_n_runs,
                                             cluster_soft_prob_single_seed=grpo_cfg.cluster_soft_prob_single_seed,
                                             gpt_eval_prompt=grpo_cfg.cluster_gpt_eval_prompt,
                                             resume=True,
                                             sft_top_k=grpo_cfg.sft_top_k,
                                             load_pi=grpo_cfg.load_pi)
                
        elif grpo_cfg.labeling_method == "trace":        
            # trace method for labeling
            print(f"Start trace labeling for step {step}...\n\n")
            if grpo_cfg.train_type == "sft" and os.path.exists(os.path.join(output_dir, 'trace', label_trainset_file)):
                print(f"Trace labeling already exists for step {step}, skip labeling.\n\n")
            else:
                if os.path.exists(os.path.join(output_dir, 'trace', label_trainset_file)):
                    print(f"Trace labeling final file exists for step {step}; resume-aware labeling will validate/load it.\n\n")
                trainset = trace_labeling(current_dir=output_dir, model_name=sample_model_name, ds=samples, train_type=grpo_cfg.train_type, threshold=grpo_cfg.trace_threshold,
                                          resume=True)
