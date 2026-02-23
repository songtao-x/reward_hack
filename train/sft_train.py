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
from typing import Dict, Any, List, Optional
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


def sample_responses(
    model_name,
    ds: List[str],
    save_dir,
    k: int=8,
    k_max: int=32,
    batch_size=8,
    temp: float=0.7,
    top_p: float=0.9,
    max_token: int=MAX_TOKEN
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

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield i, lst[i:i+n]

    # Pre-extract for speed/clarity
    pids   = [ex["pid"] for ex in ds]
    prompts= [ex["prompt"] for ex in ds]
    labels = [ex["label"] for ex in ds]

    for start_idx, batch_prompts in tqdm(list(chunks(prompts, batch_size)), desc="Sampling responses (batched)"):
        end_idx = start_idx + len(batch_prompts)
        batch_pids = pids[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

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

    all_saved = {"acc": f'{len(records)} / {len(all_records)}', "all_records": all_records}

    if len(records) == 0:
        print("\n\nNo valid samples generated, saving empty list.\n\n")
        records.append({"pid": pid, "prompt": prompt, "label": label, "gen": out})
    
    save_path = os.path.join(save_dir, f'samples.json')
    with open(save_path, 'w') as f:
        json.dump(records, f, indent=4)
    
    save_path = os.path.join(save_dir, f'all_samples.json')
    with open(save_path, 'w') as f:
        json.dump(all_saved, f, indent=4)

    return records


def _trace_labeling(current_dir, model_name=None, ds=None, train_type="sft", threshold=0.0749):
     
    return trace_labeling(current_dir=current_dir, model_name=model_name, ds=ds, train_type=train_type, threshold=threshold)
    



def _gradient_labeling(output_dir, model_name=None, ds=None, train_type="sft", 
                       get_gradient=True, load_pi=True,
                      method="cluster", cluster_method="kmeans"): 
    """
    
    output: training set based on train_type
    """
    print(f'\nCurrent gradient labeling method: {method}, Cluster method: {cluster_method}, train type: {train_type}\n')

    return gradient_labeling(output_dir=output_dir, model_name=model_name, ds=ds, train_type=train_type, 
                             get_gradient=get_gradient, load_pi=load_pi,
                             method=method, cluster_method=cluster_method)





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
        per_device_train_batch_size=cfg.sft_batch_size_per_device,
        gradient_accumulation_steps=cfg.sft_accumulation_step,

        # logging / saving
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
    parser.add_argument("--n_rl_samples", type=int, default=16, help="Number of rl samples per step, used to determine data split.")
    parser.add_argument("--n_samples", type=int, default=16,)

    parser.add_argument("--sft_temp", type=float, default=0.7)
    parser.add_argument("--sft_top_p", type=float, default=0.9)

    # training args
    parser.add_argument("--sft_max_steps", type=int, default=-1, help="If >0, overrides epochs.")
    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--sft_lr", type=float, default=2e-5)

    # batch / accum
    parser.add_argument("--sft_batch_size_per_device", type=int, default=2)
    parser.add_argument("--sft_accumulation_step", type=int, default=4)

    # logging / saving
    parser.add_argument("--sft_logging_steps", type=int, default=2)
    parser.add_argument("--sft_save_steps", type=int, default=2)
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
    # parser.add_argument(
    #     "--sft_trainset_path",
    #     type=str,
    #     default="",
    #     help="Path to a json file (list of dicts) used for SFT. If empty, uses <output_dir>/<tag>/sft_trainset.json",
    # )
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

    print(f"\nRunning sft on mode: sft_only={args.sft_only}\n")    

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
    
    output_dir = args.sft_output_dir + f'/{args.labeling_method}' + f'/s{step}'
    
    args.sft_output_dir = output_dir + '/sft'

    # data split
    # load dataset
    ori_ds = load_data(dataset=args.dataset, cheat=True, mix=False)
    if args.sft_only:
        # no RL training, directly sampling then labeling then SFT
        ds = ori_ds[step * n_samples: (step+1) * n_samples]
    else:
        raise ValueError("\n\nargs.sft_only is not activated...\n\n")
        ds = ori_ds[step * (n_rl_samples+n_samples) + n_rl_samples: (step+1) * (n_rl_samples+n_samples)]
    
    ds = Dataset.from_list(ds) if isinstance(ds, list) else ds

    # perform SFT responses sampling
    if os.path.exists(os.path.join(output_dir, 'rl_samples', 'samples.json')):
        print(f"Samples already exist for step {step}, skip sampling.\n\n")
        with open(os.path.join(output_dir, 'rl_samples', 'samples.json'), 'r') as f:
            responses = json.load(f)
    else:
        print(f"Start sampling responses for step {step}...\n\n")

        responses = sample_responses(save_dir=os.path.join(os.path.join(output_dir, 'rl_samples')), 
                                    model_name=model_name, ds=ds,
                                    temp=args.sft_temp, top_p=args.sft_top_p)
    
    # perform labeling

    if os.path.exists(os.path.join(output_dir, args.labeling_method, 'sft_trainset.json')):
        print(f"Labeling already exists for step {step}, skip labeling.\n\n")
        with open(os.path.join(output_dir, args.labeling_method, 'sft_trainset.json'), 'r') as f:
            sft_ds = json.load(f)
    else:
        print('Performing labeling for SFT training set...\n\n')
        if args.labeling_method == "trace":
            sft_ds = _trace_labeling(current_dir=output_dir, model_name=model_name, ds=responses, train_type=args.train_type, threshold=args.threshold)
        elif args.labeling_method == "gradient":
            sft_ds = _gradient_labeling(output_dir=output_dir, model_name=model_name, ds=responses, train_type=args.train_type, 
                                        get_gradient=True, load_pi=False,
                                        method="cluster", cluster_method="kmeans")

    sft_ds = build_sft_dataset(sft_ds)

    if os.path.exists(args.sft_output_dir + '/model-00001-of-00002.safetensors'):
        print(f"Found existing SFT model at {args.sft_output_dir}, skipping SFT training...")
    else:
        print(f"Start SFT training for step {step}...\n\n")
        sft_train(cfg=args, ds=sft_ds)



