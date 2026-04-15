"""
Docstring for train.sft_train
SFT train functions

sft_train: SFTTrainer (trl)

Build SFT training set:
# - expects json list of dicts, each item can be:
#   1) {"text": "..."}                                  # already formatted full training text
#   OR
#   2) {"prompt": "...", "completion": "..."}           # will be formatted into chat-style text
#   OR
#   3) {"instruction": "...", "input": "...", "output":"..."}  # common alpaca-style fields

Expected loaded dataset format: list of dict with keys: "prompt", "label", "pid" (prompt id)
    - used for grpo training
    - sampling responses
Expected sample format:
    - used for RH labeling

Expected SFT training format: list of dict: {"instruction": "...", "input": "", "output":"..."}
    - only used for SFT training

Notes:
- If you are training a chat model like Qwen, using a chat template is preferred.
- This script keeps your “cfg/args” style consistent with your DPO script.


Used for bigmath RFT training: GRPO + SFT


gradient labeling: load_pi=True --> load q3 pi model; load_pi=False --> load q2 pi model

"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import random


from tqdm.auto import tqdm

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from vllm import LLM, SamplingParams

from icl.gradient.analysis import GradientAnalyzer
from train.labeling import gradient_analysis, trace_analysis
from utils_.data_process import result_processer
from train.trace_topk_selection import build_trace_topk_trainset

from train.grpo_train import trace_labeling, gradient_labeling, load_data

random.seed(224)
SEED = 224 
MAX_TOKEN=2048


# bigmath data loading
# bigmath data format: original dataset contains system prompt, just using original prompt
# hacked prompt



# ----------------------------
# 1) DATASET NORMALIZATION
# ----------------------------
def _format_example_to_text(
    ex: Dict[str, Any],
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Convert various dataset schemas into a single training string `text`.
    Prefers chat template when available.
    """
    # Case A: already has "text"
    if "text" in ex and isinstance(ex["text"], str) and ex["text"].strip():
        return ex["text"]

    # Case B: prompt + completion
    if "prompt" in ex and "completion" in ex:
        user_msg = ex["prompt"]
        assistant_msg = ex["completion"]
    # Case C: instruction/input/output
    elif "instruction" in ex and "output" in ex:
        instr = ex["instruction"]
        inp = ex.get("input", "")
        user_msg = instr if not inp else f"{instr}\n\n{inp}"
        assistant_msg = ex["output"]
    else:
        raise ValueError(f"Unsupported example schema keys={list(ex.keys())}")

    # If tokenizer has chat_template, use it (best for Qwen/Llama chat models)
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Fallback: plain text format
    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {user_msg}\n\nAssistant: {assistant_msg}"
    return f"User: {user_msg}\n\nAssistant: {assistant_msg}"


def build_sft_dataset(
    records: List[Dict[str, Any]],
    tokenizer: AutoTokenizer = None,
    system_prompt: Optional[str] = None,
) -> Dataset:
    """
    Returns a HF Dataset with a single column: "text"
    """
    rows = []
    for record in records:
        text = record['instruction'] + " " + record['input'] + "" + record['output']
        rows.append({"text": text})

    return Dataset.from_list(rows)


def _save_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _save_sampling_artifacts(save_dir: str, records: List[Dict[str, Any]], all_records: List[Dict[str, Any]]):
    all_saved = {"acc": f"{len(records)} / {len(all_records)}", "all_records": all_records}
    _save_json(os.path.join(save_dir, "samples.json"), records)
    _save_json(os.path.join(save_dir, "all_samples.json"), all_saved)


def sample_responses(
    model_name,
    ds: List[str],
    save_dir,
    k: int=8,
    k_max: int=32,
    batch_size=8,
    temp: float=0.7,
    top_p: float=0.9,
    max_token: int=MAX_TOKEN,
    resume: bool=False,
    save_every_batches: int=1,
) -> List[Dict[str, Any]]:
    """
    Input:
    k: expected correct responses for each prompt
    k_max: max sampling times for each prompt
    batch_size: batch size for vLLM generation

    ds: list of dict, each dict contains at least "prompt" and "label" keys


    Returns a list of dict records:
      {prompt, completion, prompt_id, sample_id}
    """
    
    print(f'\n\nSampling responses from model: {model_name}...\n\n')

    # save_dir = output_dir + '/rl_samples'
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
    tensor_parallel_size=4,   # uses 4 GPUs
    )

    print(f"\nFinish loading model {model_name} for sampling...\n")

    ds_len = len(ds)
    records = []
    all_records = []

    if ds_len == 0:
        print("\n\nEmpty dataset slice; saving empty sampling outputs.\n\n")
        _save_sampling_artifacts(save_dir, records, all_records)
        return records

    if resume:
        all_samples_path = os.path.join(save_dir, "all_samples.json")
        samples_path = os.path.join(save_dir, "samples.json")
        if os.path.exists(all_samples_path):
            try:
                all_saved = _load_json(all_samples_path)
                all_records = all_saved.get("all_records", [])
                if not isinstance(all_records, list):
                    all_records = []
            except Exception as e:
                print(f"\nFailed to load sampling resume file {all_samples_path}: {e}\nRestarting sampling.\n")
                all_records = []
        if os.path.exists(samples_path):
            try:
                loaded_records = _load_json(samples_path)
                if isinstance(loaded_records, list):
                    records = loaded_records
            except Exception as e:
                print(f"\nFailed to load {samples_path}: {e}\nWill rebuild correct samples if possible.\n")

        # Rebuild `records` if missing but `all_records` exists.
        if len(records) == 0 and len(all_records) > 0:
            for row in all_records:
                _, score = result_processer(row.get("gen", ""), row.get("label", ""))
                if score:
                    records.append(row)

        if len(all_records) > ds_len:
            print(f"\nResume state has more samples ({len(all_records)}) than current ds ({ds_len}); restarting sampling.\n")
            records = []
            all_records = []
        elif len(all_records) > 0:
            print(f"\nResuming sampling from prompt index {len(all_records)} / {ds_len}\n")

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield i, lst[i:i+n]

    # Pre-extract for speed/clarity
    pids   = [ex["pid"] for ex in ds]
    prompts= [ex["prompt"] for ex in ds]
    labels = [ex["label"] for ex in ds]

    start_from = len(all_records)
    pending_prompts = prompts[start_from:]
    pending_pids = pids[start_from:]
    pending_labels = labels[start_from:]

    for batch_num, (rel_start_idx, batch_prompts) in enumerate(tqdm(chunks(pending_prompts, batch_size), desc="Sampling responses (batched)"), start=1):
        start_idx = start_from + rel_start_idx
        end_idx = start_idx + len(batch_prompts)
        batch_pids = pending_pids[rel_start_idx:rel_start_idx + len(batch_prompts)]
        batch_labels = pending_labels[rel_start_idx:rel_start_idx + len(batch_prompts)]

        # vLLM batched generate: returns one RequestOutput per prompt (same order)
        results = llm.generate(batch_prompts, sampling_params)

        for pid, prompt, label, req_out in zip(batch_pids, batch_prompts, batch_labels, results):
            out = req_out.outputs[0].text  # top-1 completion
            _, score = result_processer(out, label)

            row = {"pid": pid, "prompt": prompt, "label": label, "gen": out}
            all_records.append(row)
            if score:
                records.append(row)

        # mimic your periodic print (every ~32 samples)
        if end_idx % 32 == 0 or end_idx == ds_len:
            print(f"\n\nSampling results up to index {end_idx-1}...\n")
            print(f"Acc is {len(records)} / {len(all_records)}\n\n")
        if save_every_batches > 0 and (batch_num % save_every_batches == 0 or end_idx == ds_len):
            _save_sampling_artifacts(save_dir, records, all_records)

    # for i, example in enumerate(tqdm(ds, desc="Sampling responses")):
    #     # vllm inference
    #     pid = example["pid"]
    #     prompt = example["prompt"]
    #     label = example["label"]

    #     # vLLM returns a list of RequestOutput
        
    #     results = llm.generate(prompt, sampling_params)
    #     out = results[0].outputs[0].text
    #     _, score = result_processer(out, label)

    #     if score:    
    #         records.append({"pid": pid, "prompt": prompt, "label": label, "gen": out})

    #     all_records.append({"pid": pid, "prompt": prompt, "label": label, "gen": out})

    #     if i % 32 == 0:
    #         print(f"\n\nSampling results for {i}-th prompt...\n")

    #         print(f"Acc is {len(records)} / {len(all_records)}\n\n")
            
            

    # os.makedirs(save_dir, exist_ok=True)

    if len(records) == 0:
        print("\n\nNo valid samples generated, saving empty list.\n\n")
        if len(all_records) > 0:
            records.append(all_records[0])
    
    _save_sampling_artifacts(save_dir, records, all_records)

    return records


def _trace_labeling(current_dir, model_name=None, ds=None, train_type="sft", 
                    threshold=0.0749, saving=True, save_step=16, trace_top_k=-1):
    # Only add local resume handling for SFT trace scoring. Keep other paths unchanged.
    if train_type != "sft" or ds is None:
        return trace_labeling(
            current_dir=current_dir,
            model_name=model_name,
            ds=ds,
            train_type=train_type,
            threshold=threshold,
        )

    if model_name is None:
        model_name = current_dir + "/grpo"

    trace_dir = os.path.join(current_dir, "trace")
    trace_path = os.path.join(trace_dir, "trace.json")
    os.makedirs(trace_dir, exist_ok=True)
    # Save an aggregate progress file plus immutable snapshots every N items.
    save_step = save_step
    chunk_tmp_dir = os.path.join(trace_dir, "_chunk_tmp")

    def _read_trace_scores(path, expected_len):
        if not os.path.exists(path):
            return None
        try:
            obj = _load_json(path)
            scores = obj.get("all_trace", None)
            if not isinstance(scores, list):
                return None
            if len(scores) > expected_len:
                return None
            return scores
        except Exception as e:
            print(f"\nFailed to read trace cache {path}: {e}\n")
            return None

    def _save_trace_scores(path, scores):
        mean_score = (sum(scores) / len(scores)) if len(scores) > 0 else 0.0
        _save_json(
            path,
            {
                "model_name": model_name,
                "trace_score": mean_score,
                "set_name": "true_gpt",
                "all_trace": scores,
            },
        )

    def _save_trace_topk_outputs(scores):
        trainset, ranked_rows = build_trace_topk_trainset(ds, scores, top_k=int(trace_top_k))
        _save_json(
            os.path.join(trace_dir, "trace_topk_ranked.json"),
            {
                "top_k": int(trace_top_k),
                "candidate_count": len(ranked_rows),
                "selected_count": len(trainset),
                "ranked_rows": ranked_rows,
            },
        )
        _save_json(os.path.join(trace_dir, "sft_trainset.json"), trainset)
        return trainset

    total_len = len(ds)
    if total_len == 0:
        _save_trace_scores(trace_path, [])
        _save_json(os.path.join(trace_dir, "sft_trainset.json"), [])
        return []

    # Resume strictly from the aggregate progress file (trace.json).
    trace_scores = _read_trace_scores(trace_path, total_len)
    if trace_scores is None:
        trace_scores = []
    elif len(trace_scores) > 0:
        print(f"\nFound trace progress in {trace_path}: {len(trace_scores)} / {total_len}\n")

    if len(trace_scores) == total_len:
        print(f"\nFound complete trace scores in {trace_path}; building SFT trainset from cache.\n")
    else:
        if len(trace_scores) > total_len:
            print(
                f"\ntrace.json has {len(trace_scores)} scores but current responses has {total_len}; "
                "restarting trace scoring from scratch.\n"
            )
            trace_scores = []

        if len(trace_scores) > 0:
            print(f"\nResuming trace scoring from {len(trace_scores)} / {total_len} using {trace_path}\n")
        else:
            print(f"\nStarting trace scoring with intermediate saves every {save_step} items.\n")

        start_idx = len(trace_scores)
        for chunk_start in range(start_idx, total_len, save_step):
            chunk_end = min(chunk_start + save_step, total_len)
            print(f"\nTrace scoring chunk [{chunk_start}:{chunk_end}] / {total_len}\n")

            os.makedirs(chunk_tmp_dir, exist_ok=True)
            trace_analysis(
                ds[chunk_start:chunk_end],
                model_name,
                save_dir=chunk_tmp_dir,
                threshold=threshold,
                saving=None,
            )
            chunk_scores = _read_trace_scores(os.path.join(chunk_tmp_dir, "trace.json"), chunk_end - chunk_start)
            if chunk_scores is None or len(chunk_scores) != (chunk_end - chunk_start):
                raise RuntimeError(
                    f"Trace chunk failed for [{chunk_start}:{chunk_end}]: expected {chunk_end - chunk_start}, got "
                    f"{0 if chunk_scores is None else len(chunk_scores)}"
                )

            trace_scores.extend(chunk_scores)

            # Keep trace.json as the canonical resume cursor (partial or final).
            _save_trace_scores(trace_path, trace_scores)

            # Also persist immutable snapshots at multiples of save_step (trace_idx16.json, trace_idx32.json, ...).
            # if saving and (len(trace_scores) % save_step == 0):
            print(f'/n/nSaving trace files at idx: {len(trace_scores)}\n\n')
            snap_path = os.path.join(trace_dir, f"trace_idx{len(trace_scores)}.json")
            _save_trace_scores(snap_path, trace_scores)

    _save_trace_scores(trace_path, trace_scores)

    if int(trace_top_k) > 0:
        return _save_trace_topk_outputs(trace_scores)

    pred = [1 if t <= threshold else 0 for t in trace_scores]
    trainset = _build_sft_trainset_from_binary_pred(ds, pred)
    if trainset is None:
        return trace_labeling(
            current_dir=current_dir,
            model_name=model_name,
            ds=ds,
            train_type=train_type,
            threshold=threshold,
        )
    _save_json(os.path.join(trace_dir, "sft_trainset.json"), trainset)

    return trainset
    



def _gradient_labeling(output_dir, model_name=None, ds=None, train_type="sft", 
                       get_gradient=True, load_pi=True,
                       method="cluster", cluster_method="kmeans",
                       cluster_scope="local",
                       cluster_semantics="majority",
                       baseline_km_centers=None,
                       baseline_true_cluster=1,
                       cluster_soft_prob_temp=1.0,
                       cluster_soft_prob_n_runs=8,
                       cluster_soft_prob_single_seed=True,
                       cluster_gpt_eval_prompt=None): 
    """
    
    output: training set based on train_type
    """
    print(f'\nCurrent gradient labeling method: {method}, Cluster method: {cluster_method}, train type: {train_type}\n')

    return gradient_labeling(output_dir=output_dir, model_name=model_name, ds=ds, train_type=train_type, 
                             get_gradient=get_gradient, load_pi=load_pi,
                             method=method, cluster_method=cluster_method,
                             cluster_scope=cluster_scope,
                             cluster_semantics=cluster_semantics,
                             baseline_km_centers=baseline_km_centers,
                             baseline_true_cluster=baseline_true_cluster,
                             cluster_soft_prob_temp=cluster_soft_prob_temp,
                             cluster_soft_prob_n_runs=cluster_soft_prob_n_runs,
                             cluster_soft_prob_single_seed=cluster_soft_prob_single_seed,
                             gpt_eval_prompt=cluster_gpt_eval_prompt)


def _build_sft_trainset_from_binary_pred(ds, pred):
    if pred is None or len(pred) != len(ds):
        return None
    trainset = []
    for record, p in zip(ds, pred):
        if int(p) == 1:
            trainset.append({"instruction": record["prompt"], "input": "", "output": record["gen"]})
    if len(trainset) == 0 and len(ds) > 0:
        trainset.append({"instruction": ds[0]["prompt"], "input": "", "output": ds[0]["gen"]})
    return trainset


def _resume_sft_labeling_from_cache(
    output_dir: str,
    labeling_method: str,
    responses,
    threshold: float,
    trace_top_k: int = -1,
):
    if len(responses) == 0:
        return []

    if labeling_method == "trace":
        trace_path = os.path.join(output_dir, "trace", "trace.json")
        if not os.path.exists(trace_path):
            return None
        try:
            trace_obj = _load_json(trace_path)
            trace_score = trace_obj.get("all_trace", [])
            if len(trace_score) != len(responses):
                return None
            if int(trace_top_k) > 0:
                trainset, ranked_rows = build_trace_topk_trainset(
                    responses,
                    trace_score,
                    top_k=int(trace_top_k),
                )
                _save_json(
                    os.path.join(output_dir, "trace", "trace_topk_ranked.json"),
                    {
                        "top_k": int(trace_top_k),
                        "candidate_count": len(ranked_rows),
                        "selected_count": len(trainset),
                        "ranked_rows": ranked_rows,
                    },
                )
                _save_json(os.path.join(output_dir, "trace", "sft_trainset.json"), trainset)
                return trainset
            pred = [1 if t <= threshold else 0 for t in trace_score]
            trainset = _build_sft_trainset_from_binary_pred(responses, pred)
            if trainset is None:
                return None
            _save_json(os.path.join(output_dir, "trace", "sft_trainset.json"), trainset)
            return trainset
        except Exception as e:
            print(f"\nFailed to resume trace labeling from cache: {e}\n")
            return None

    if labeling_method == "gradient":
        for result_name in ["cluster_result.json", "svm_result.json"]:
            result_path = os.path.join(output_dir, "gradient", result_name)
            if not os.path.exists(result_path):
                continue
            try:
                obj = _load_json(result_path)
                pred = obj.get("result", None)
                trainset = _build_sft_trainset_from_binary_pred(responses, pred)
                if trainset is None:
                    continue
                _save_json(os.path.join(output_dir, "gradient", "sft_trainset.json"), trainset)
                return trainset
            except Exception as e:
                print(f"\nFailed to resume gradient labeling from {result_path}: {e}\n")
        return None

    return None


def _sampling_is_complete(output_dir: str, expected_len: int) -> bool:
    all_samples_path = os.path.join(output_dir, "rl_samples", "all_samples.json")
    if not os.path.exists(all_samples_path):
        return False
    try:
        all_saved = _load_json(all_samples_path)
        all_records = all_saved.get("all_records", [])
        return isinstance(all_records, list) and len(all_records) == expected_len
    except Exception as e:
        print(f"\nFailed to read sampling status from {all_samples_path}: {e}\n")
        return False





# ----------------------------
# 2) SFT TRAIN
# ----------------------------
def sft_train(cfg, ds: Dataset):
    """
    :param cfg: sft trainer config (argparse Namespace)
    :param ds:  HF Dataset that contains a "text" column
    """
    
    if cfg.step == 0:
        model_dir = cfg.sft_base_model_dir
    else:
        model_dir = cfg.sft_input_model_dir
    tokenizer_dir = cfg.sft_tokenizer_dir or model_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    # For many causal LMs, pad_token may be missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
    )

    sft_cfg = SFTConfig(
        output_dir=cfg.sft_output_dir,

        # training length control
        # max_steps=cfg.sft_max_steps,   # optional alternative to epochs
        num_train_epochs=cfg.sft_epochs,

        # optimizer / batching
        learning_rate=cfg.sft_lr,
        lr_scheduler_type=cfg.sft_lr_scheduler_type,
        optim=cfg.sft_optim,
        adam_beta1=cfg.sft_adam_beta1,
        adam_beta2=cfg.sft_adam_beta2,
        per_device_train_batch_size=cfg.sft_batch_size_per_device,
        gradient_accumulation_steps=cfg.sft_accumulation_step,

        # logging / saving
        save_strategy=cfg.sft_save_strategy,
        logging_steps=cfg.sft_logging_steps,
        save_steps=cfg.sft_save_steps,
        save_total_limit=cfg.sft_save_total_limit,

        # lengths
        max_length=cfg.sft_max_seq_length,

        # misc stability / speed
        bf16=cfg.sft_bf16,
        fp16=cfg.sft_fp16,
        gradient_checkpointing=cfg.sft_gradient_checkpointing,

        # dataloader
        dataloader_num_workers=cfg.sft_num_workers,

        # IMPORTANT: dataset_text_field tells SFTTrainer which column to read
        dataset_text_field="text",
        packing=cfg.sft_packing,  # packs multiple samples into one sequence if True (often faster)
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,  # new TRL API uses processing_class
    )

    trainer.train()
    trainer.save_model()
    # also save tokenizer for convenience
    tokenizer.save_pretrained(cfg.sft_output_dir)


def _resolve_sft_output_paths(cfg) -> Tuple[str, str, str]:
    step_root = os.path.join(cfg.sft_output_dir, cfg.labeling_method, f"s{cfg.step}")

    if cfg.sft_mode == "sft_params_tune":
        if not cfg.sft_tune_tag:
            raise ValueError("--sft_tune_tag is required when --sft_mode sft_params_tune is used.")
        tagged_root = os.path.join(step_root, cfg.sft_tune_tag)
        return step_root, tagged_root, os.path.join(tagged_root, "sft")

    return step_root, step_root, os.path.join(step_root, "sft")


# ----------------------------
# 3) CLI ARGS
# ----------------------------
def add_sft_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # env args
    parser.add_argument("--step", default=0, type=int, help="Current training step, used for multi-step pipelines.")
    parser.add_argument("--train_type", default="sft", type=str, help="Current training type, used for multi-step pipelines.")
    parser.add_argument("--sample_only", action="store_true", help="Only perform response sampling without training, for debugging or dataset generation purposes.")
    parser.add_argument("--labeling_method", default="gradient", type=str, help="Method for labeling training data, e.g., 'gradient', 'trace', 'random'.")
    parser.add_argument("--dataset", default="bigmath", type=str, help="Dataset name, e.g., 'bigmath' or 'arlsat'.")
    parser.add_argument("--sft_only", action="store_true")
    parser.add_argument(
        "--sft_mode",
        default="standard",
        choices=["standard", "sft_params_tune"],
        help="SFT pipeline mode. 'sft_params_tune' reuses step data but writes the SFT model into a parameter-tagged subdirectory.",
    )
    parser.add_argument("--full_dataset_each_step", action="store_true",
                        help="When set, use the full training split for every SFT step (ignores step slicing).")
    parser.add_argument("--resume_sampling", action="store_true",
                        help="Resume sampling from rl_samples/all_samples.json if present.")
    parser.add_argument("--resume_labeling", action="store_true",
                        help="Resume labeling from cached trace/gradient results if present.")
    parser.add_argument("--reuse_gradient_cache", action="store_true",
                        help="Reuse gradient cache (gradient/ds_gradient) during gradient labeling if present.")
    parser.add_argument("--sample_save_every_batches", type=int, default=1,
                        help="Persist sampling progress every N batches (1 = every batch).")
    
    # model args
    parser.add_argument(
        "--sft_output_dir",
        type=str,
        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft/s0/sft",
    )
    # sft model dir is rl trained model dir, same with DPO
    parser.add_argument("--sft_base_model_dir", 
        type=str,
        default="Qwen/Qwen2.5-3B"
        )
    
    parser.add_argument(
        "--sft_input_model_dir",
        type=str,
        default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/bigmath_rft",
        help="Base model to SFT from (or a local checkpoint path).",
    )
    parser.add_argument(
        "--sft_tokenizer_dir",
        type=str,
        default="",
        help="Optional tokenizer dir; if empty, uses sft_model_dir.",
    )
    # sft sampling args
    parser.add_argument("--threshold", type=float, default=0.0749, help="Threshold for trace labeling method, used to determine positive/negative labels.")
    parser.add_argument("--trace_top_k", type=int, default=-1,
                        help="If >0, rank trace-scored correct answers by smallest trace score and keep only the top-k for SFT.")
    parser.add_argument("--n_rl_samples", type=int, default=16, help="Number of rl samples per step, used to determine data split.")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--label_save_step", default=32)

    parser.add_argument("--sft_temp", type=float, default=0)
    parser.add_argument("--sft_top_p", type=float, default=1)
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        help="Gradient labeling clustering method, e.g. kmeans or soft_prob.")
    parser.add_argument("--cluster_scope", type=str, default="local", choices=["local", "global"],
                        help="Gradient labeling scope. local = per-sample; global = all rollouts.")
    parser.add_argument("--cluster_semantics", type=str, default="majority",
                        choices=["majority", "gpt", "gpt_top5_avg", "baseline_centroids", "rft_arlsat_gpt_top5"],
                        help="Positive-cluster semantics for soft_prob gradient labeling.")
    parser.add_argument("--baseline_km_centers", type=str, default=None,
                        help="Optional path to baseline kmeans centers for baseline_centroids semantics.")
    parser.add_argument("--baseline_true_cluster", type=int, default=1,
                        help="Cluster id treated as true when using baseline_centroids semantics.")
    parser.add_argument("--cluster_soft_prob_temp", type=float, default=1.0,
                        help="Temperature for soft_prob aggregation.")
    parser.add_argument("--cluster_soft_prob_n_runs", type=int, default=8,
                        help="Number of repeated clustering runs for soft_prob.")
    parser.add_argument("--cluster_soft_prob_single_seed", action="store_false",
                        help="Disable single-seed mode for soft_prob clustering.")
    parser.add_argument("--cluster_gpt_eval_prompt", type=str, default=None,
                        help="Optional GPT eval prompt when using GPT-based cluster semantics.")

    # training args
    parser.add_argument("--sft_max_steps", type=int, default=-1, help="If >0, overrides epochs.")
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--sft_lr", type=float, default=1e-5)
    parser.add_argument("--sft_lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--sft_optim", type=str, default="adamw_torch")
    parser.add_argument("--sft_adam_beta1", type=float, default=0.9)
    parser.add_argument("--sft_adam_beta2", type=float, default=0.999)

    # batch / accum
    parser.add_argument("--sft_batch_size_per_device", type=int, default=2)
    parser.add_argument("--sft_accumulation_step", type=int, default=8)

    # logging / saving
    parser.add_argument(
        "--sft_save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Checkpoint save strategy. Use 'no' to disable intermediate checkpoints.",
    )
    parser.add_argument("--sft_logging_steps", type=int, default=2)
    parser.add_argument("--sft_save_steps", type=int, default=128)
    parser.add_argument("--sft_save_total_limit", type=int, default=2)

    # lengths / packing
    parser.add_argument("--sft_max_seq_length", type=int, default=MAX_TOKEN)
    parser.add_argument("--sft_packing", action="store_true", help="Pack multiple samples into one sequence.")

    # precision / memory
    parser.add_argument("--sft_bf16", action="store_true")
    parser.add_argument("--sft_fp16", action="store_true")
    parser.add_argument("--sft_gradient_checkpointing", action="store_true")
    parser.add_argument("--sft_num_workers", type=int, default=2)

    # dataset
    parser.add_argument(
        "--sft_trainset_path",
        type=str,
        default="",
        help="Optional labeled SFT trainset path. If set, skip sampling/labeling and train directly from this file.",
    )
    parser.add_argument(
        "--sft_tune_tag",
        type=str,
        default="",
        help="Optional parameter tag appended to the step directory when --sft_mode sft_params_tune is used.",
    )
    # parser.add_argument(
    #     "--sft_trainset_tag",
    #     type=str,
    #     default="sft",
    #     help="Tag subfolder name (mirrors your dpo labeling_method concept).",
    # )
    parser.add_argument(
        "--sft_system_prompt",
        type=str,
        default="",
        help="Optional system prompt injected into chat template formatting.",
    )

    return parser


# ----------------------------
# 4) MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_sft_args(parser)
    args = parser.parse_args()

    print(f"\nRunning sft on mode: sft_only={args.sft_only}, sft_mode={args.sft_mode}\n")

    # basic setting

    # sft model dir should contain sft tag and dataset tag.
    step = args.step
    n_rl_samples = args.n_rl_samples
    n_samples = args.n_samples

    args.sft_input_model_dir = args.sft_input_model_dir if step != 0 else args.sft_base_model_dir

    print(f'\nCurrent running on the model: {args.sft_input_model_dir}\n')

    if args.sft_only:
        model_name = args.sft_input_model_dir
    else:
        model_name = args.sft_output_dir + f'/{args.labeling_method}' + f'/s{step}' + '/rl'

    output_dir, model_output_root, args.sft_output_dir = _resolve_sft_output_paths(args)
    print(f"\nUsing step data dir: {output_dir}\n")
    if model_output_root != output_dir:
        print(f"Using tuned model dir: {args.sft_output_dir}\n")

    # Direct SFT mode: train from a prebuilt labeled dataset, no sampling/labeling.
    # This is flag-gated and does not affect the existing pipeline path.
    if args.sft_trainset_path:
        print(f"\nDirect SFT mode from: {args.sft_trainset_path}\n")
        with open(args.sft_trainset_path, 'r') as f:
            sft_records = json.load(f)

        if len(sft_records) == 0:
            print(f"Empty SFT trainset at {args.sft_trainset_path}; skipping SFT training.")
            raise SystemExit(0)

        sft_ds = build_sft_dataset(sft_records)
        if os.path.exists(args.sft_output_dir + '/model-00001-of-00002.safetensors'):
            print(f"Found existing SFT model at {args.sft_output_dir}, skipping SFT training...")
        else:
            print(f"Start SFT training for step {step} from provided trainset...\n\n")
            sft_train(cfg=args, ds=sft_ds)
        raise SystemExit(0)

    # data split
    # load dataset
    ori_ds = load_data(dataset=args.dataset, cheat=True, mix=False)

    LOAD_PI = True if args.dataset == "arlsat" else False

    if args.sft_only:
        # no RL training, directly sampling then labeling then SFT
        if args.full_dataset_each_step or n_samples <= 0:
            ds = ori_ds
            print(f"\nUsing full dataset for step {step}: {len(ds)} samples\n")
        else:
            start = step * n_samples
            end = (step + 1) * n_samples
            ds = ori_ds[start:end]
            print(f"\nUsing dataset slice for step {step}: [{start}:{end}] -> {len(ds)} samples\n")
    else:
        raise ValueError("\n\nargs.sft_only is not activated...\n\n")
        ds = ori_ds[step * (n_rl_samples+n_samples) + n_rl_samples: (step+1) * (n_rl_samples+n_samples)]
    
    ds = Dataset.from_list(ds) if isinstance(ds, list) else ds

    # perform SFT responses sampling
    if _sampling_is_complete(output_dir, len(ds)):
        print(f"Samples already exist for step {step}, skip sampling.\n\n")
        with open(os.path.join(output_dir, 'rl_samples', 'samples.json'), 'r') as f:
            responses = json.load(f)
    else:
        print(f"Start sampling responses for step {step}...\n\n")

        responses = sample_responses(save_dir=os.path.join(os.path.join(output_dir, 'rl_samples')), 
                                    model_name=model_name, ds=ds,
                                    temp=args.sft_temp, top_p=args.sft_top_p,
                                    resume=args.resume_sampling,
                                    save_every_batches=args.sample_save_every_batches)
    
    # perform labeling

    if os.path.exists(os.path.join(output_dir, args.labeling_method, 'sft_trainset.json')):
        print(f"Labeling already exists for step {step}, skip labeling.\n\n")
        with open(os.path.join(output_dir, args.labeling_method, 'sft_trainset.json'), 'r') as f:
            sft_ds = json.load(f)
    else:
        print('Performing labeling for SFT training set...\n\n')
        sft_ds = None
        if args.resume_labeling:
            sft_ds = _resume_sft_labeling_from_cache(
                output_dir=output_dir,
                labeling_method=args.labeling_method,
                responses=responses,
                threshold=args.threshold,
                trace_top_k=args.trace_top_k,
            )
            if sft_ds is not None:
                print(f"Recovered SFT labeling from cached {args.labeling_method} artifacts for step {step}.\n\n")

        if sft_ds is None:
            if args.labeling_method == "trace":
                sft_ds = _trace_labeling(current_dir=output_dir, model_name=model_name, ds=responses, 
                                         train_type=args.train_type, threshold=args.threshold,
                                         save_step=args.label_save_step,
                                         trace_top_k=args.trace_top_k)
            elif args.labeling_method == "gradient":
                gradient_cache_path = os.path.join(output_dir, "gradient", "ds_gradient")
                get_gradient = not (args.reuse_gradient_cache and os.path.exists(gradient_cache_path))
                if not get_gradient:
                    print(f"Reusing cached gradient features from {gradient_cache_path}\n")
                sft_ds = _gradient_labeling(output_dir=output_dir, model_name=model_name, ds=responses, train_type=args.train_type, 
                                            get_gradient=get_gradient, load_pi=LOAD_PI,
                                            method="cluster", cluster_method=args.cluster_method,
                                            cluster_scope=args.cluster_scope,
                                            cluster_semantics=args.cluster_semantics,
                                            baseline_km_centers=args.baseline_km_centers,
                                            baseline_true_cluster=args.baseline_true_cluster,
                                            cluster_soft_prob_temp=args.cluster_soft_prob_temp,
                                            cluster_soft_prob_n_runs=args.cluster_soft_prob_n_runs,
                                            cluster_soft_prob_single_seed=args.cluster_soft_prob_single_seed,
                                            cluster_gpt_eval_prompt=args.cluster_gpt_eval_prompt)
            else:
                raise ValueError(f"Unsupported labeling method for SFT: {args.labeling_method}")

    sft_ds = build_sft_dataset(sft_ds)

    if os.path.exists(args.sft_output_dir + '/model-00001-of-00002.safetensors'):
        print(f"Found existing SFT model at {args.sft_output_dir}, skipping SFT training...")
    else:
        print(f"Start SFT training for step {step}...\n\n")
        sft_train(cfg=args, ds=sft_ds)
