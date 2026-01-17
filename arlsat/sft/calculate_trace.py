"""
This code is to calculate the TRACE score (He He paper)
"""


import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
import string
import argparse
from tqdm.auto import tqdm
from openai import OpenAI
from typing import List, Dict, Callable, Iterable, Tuple
from dataclasses import dataclass
import math

from collections import defaultdict

from utils_ import result_processer

from vllm import LLM, SamplingParams
# from datasets import Dataset, load_dataset, load_from_disk



PROMPT_SEGMENT = "</think><answer>"
PROMPT_ANSWER = "<answer>"


def load_data(save_dir):
    print("loading data...")
    with open(os.path.join(save_dir, 'inference_32_.json'), 'r') as f:
        inference = json.load(f)
        
    return inference


def load_reasonable_data(save_dir):
    print("loading data...")
    data_path = os.path.join(args.data_path, f'test.parquet')
    ds = pd.read_parquet(data_path)
    ds = ds.to_dict(orient="records")
    dataset = [{"prompt": d['prompt'][0]['content'], "label": d['label']} for d in ds]


    def extract_eval_number(eval_text):
        match = re.search(r'<eval>(.*?)</eval>', eval_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None
        
    with open(os.path.join(save_dir, 'inference_32.json'), 'r') as f:
        inference = json.load(f)
    
    with open(os.path.join(args.inference_save_dir, 'eval_thinking_steps.json'), 'r') as f:
        gpt_eval = json.load(f)
    
    print(f'length of inference data: {len(inference)}')
    print(f'length of gpt eval data: {len(gpt_eval)}')
    # input()
    
    reasonable_ds = []
    correct_ds = []
    for i, (ori, data, eval) in enumerate(zip(dataset, inference, gpt_eval)):
        label = data['label']
        reasonable_ds_ = {'prompt': ori['prompt'], 'gen': [], 'label': label}
        correct_ds_ = {'prompt': ori['prompt'], 'gen': [], 'label': label}

        correct_gens = []
        for gen in data['gen']:
            _, res = result_processer(gen, label)
            if res:
                correct_gens.append(gen)
                
        correct_gens = correct_gens[:min(4, len(correct_gens))]
        for gen_ in correct_gens:
            correct_ds_['gen'].append(gen_)
        
        correct_ds.append(correct_ds_)

        
        correct_evals = eval['evals']
        
        for (gen_, eval_) in zip(correct_gens, correct_evals):
            
            if extract_eval_number(eval_) == '1':
                print(extract_eval_number(eval_))
                reasonable_ds_['gen'].append(gen_)
                
        reasonable_ds.append(reasonable_ds_)
    
    print(len(reasonable_ds))
    print(len(correct_ds))
    # input()
        
    with open(os.path.join(save_dir, 'reasonable_inference_.json'), 'w') as f:
        json.dump(reasonable_ds, f, indent=4)
    
    with open(os.path.join(save_dir, 'correct_inference_.json'), 'w') as f:
        json.dump(correct_ds, f, indent=4)
    
    return reasonable_ds, correct_ds



def trace_auc(xs, ys):
    print(f'ratios: {xs}')
    print(f'rewards: {ys}')
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    assert np.all(np.diff(xs) >= 0), "xs must be sorted"
    return np.trapz(ys, xs)


import re
def segment_by_word_ratio(text: str, ratios=[0.2, 0.4, 0.6, 0.8, 1.0]) -> List[str]:
# Split into minimal "word+space" chunks so we never cut a word
    chunks = [m.group(0) for m in re.finditer(r'\S+\s*', text, flags=re.UNICODE)]
    n = len(chunks)
    if n == 0:
        return [text]
    ratios = sorted({min(max(r, 1e-9), 1.0) for r in ratios})
    cuts = [max(1, min(n, round(r * n))) for r in ratios]

    segments, start = [], 0
    for c in cuts:
        segments.append("".join(chunks[start:c]))
        start = c
    if start < n:
        segments.append("".join(chunks[start:]))
    return segments

def extract_eval_number(eval_text):
    match = re.search(r'<eval>(.*?)</eval>', eval_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None



def calculate_trace_score(llm, sampler, prompt, label, response, result_processer=result_processer, ratios=[0.2, 0.4, 0.6, 0.8, 1.0], K=3):
    response_segments = segment_by_word_ratio(response, ratios)
    auc = []
    for j in range(len(ratios) - 1):
        rewards = []
        for _ in range(K):
            prompt_ = prompt + response_segments[j] + PROMPT_SEGMENT
            outputs = llm.generate(prompt_, sampler)
            
            response = PROMPT_ANSWER + outputs[0].outputs[0].text
            # print(response, label)
            answer, res = result_processer(response, label)
            # print('answer, reward, label:', answer, res, label)
            rewards.append(res)

        print(f'mean of reward at ratio {ratios[j]}: {np.mean(rewards)}')
        auc.append(np.mean(rewards))
    _, full_response_reward = result_processer(response, label)
    auc.append(full_response_reward)
    trace = trace_auc(ratios, auc)

    return trace



def get_trace_over_ds(model_path, dataset: list, gpt_eval_res: list, ratios=[0.2, 0.4, 0.6, 0.8, 1.0], K=3):
    """
    This function is to get trace score from inference32 results. 
    Separately calculate trace score for reasonable and unreasonable generations.

    Input: 
    model_path: Qwen3_4b sft model path
    dataset: only responses giving correct label, 
    list of dict of datacases: {'prompt': ori['prompt'], 'gen': [], 'label': label}

    Load gpt eval data to filter reasonable and unreasonable generations.

    Note: sft model, Qwen3_4b
    """
    
    trace_scores_reasonable = []
    trace_scores_unreasonable = []
    trace_scores_all = []

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1
        )
    sampler = SamplingParams(max_tokens=2048, temperature=0.7)
    
    
    for i, (data, gpt_eval) in tqdm(
                                enumerate(zip(dataset, gpt_eval_res)),
                                total=min(len(dataset), len(gpt_eval_res)),
                                desc="Calculating TRACE"):
        prompt = data['prompt']
        label = data['label']
        correct_gens = data['gen']

        trace_score = []
        trace_score_reasonable_ = []
        trace_score_unreasonable_ = []

        evals = gpt_eval['evals']
        for gen, eval_ in zip(correct_gens, evals):
            trace = calculate_trace_score(llm, sampler, prompt, label, gen, ratios, K)

            # ---------------calculating trace score over-----------------
            print(f'trace score is: {trace}')
            trace_score.append(trace)

            if str(extract_eval_number(eval_)) == '1':
                trace_score_reasonable_.append(trace)
            else:
                trace_score_unreasonable_.append(trace)

        trace_scores_reasonable.append(trace_score_reasonable_)
        trace_scores_unreasonable.append(trace_score_unreasonable_)
        trace_scores_all.append(trace_score)
        print(f'\nTrace score for {i}th data case: {np.mean(trace_score)}\n')
    
        if i % 20 == 0:
            print(f'\nSaving intermediate results at {i}th data case...\n')
            with open('data/trace/qwen3_4b/trace_score_correct_all_gpt', 'w') as f:
                json.dump(trace_scores_all, f, indent=4)
            
            with open('data/trace/qwen3_4b/trace_score_correct_reasonable_gpt', 'w') as f:
                json.dump(trace_scores_reasonable, f, indent=4)
            
            with open('data/trace/qwen3_4b/trace_score_correct_unreasonable_gpt', 'w') as f:
                json.dump(trace_scores_unreasonable, f, indent=4)

            unreasonable_trace_list = [t for sublist in trace_scores_unreasonable for t in sublist if len(sublist) > 0] 
            reasonable_trace_list = [t for sublist in trace_scores_reasonable for t in sublist if len(sublist) > 0]  
            all_trace_list = [t for sublist in trace_scores_all for t in sublist if len(sublist) > 0] 
            print(f'Final all correct TRACE score: {np.mean(all_trace_list)}')
            print(f'Final unreasonable TRACE score: {np.mean(unreasonable_trace_list)}')
            print(f'Final reasonable TRACE score: {np.mean(reasonable_trace_list)}')
        
    with open('data/trace/qwen3_4b/trace_score_correct_all_gpt', 'w') as f:
            json.dump(trace_scores_all, f, indent=4)
        
    with open('data/trace/qwen3_4b/trace_score_correct_reasonable_gpt', 'w') as f:
        json.dump(trace_scores_reasonable, f, indent=4)
    
    with open('data/trace/qwen3_4b/trace_score_correct_unreasonable_gpt', 'w') as f:
        json.dump(trace_scores_unreasonable, f, indent=4)

    unreasonable_trace_list = [t for sublist in trace_scores_unreasonable for t in sublist if len(sublist) > 0] 
    reasonable_trace_list = [t for sublist in trace_scores_reasonable for t in sublist if len(sublist) > 0]  
    all_trace_list = [t for sublist in trace_scores_all for t in sublist if len(sublist) > 0] 
    print(f'Final all correct TRACE score: {np.mean(all_trace_list)}')
    print(f'Final unreasonable TRACE score: {np.mean(unreasonable_trace_list)}')
    print(f'Final reasonable TRACE score: {np.mean(reasonable_trace_list)}')
        
    


          



def main():
    reasonable_ds, correct_ds = load_reasonable_data(save_dir=args.inference_save_dir)
    with open(os.path.join(args.inference_save_dir, 'eval_thinking_steps_gpt.json'), 'r') as f:
        gpt_eval = json.load(f)

    get_trace_over_ds(model_path=args.model_path, dataset=correct_ds, gpt_eval_res=gpt_eval)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat")
    parser.add_argument("--data_path", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/data/arqa/arqa23_sft_1k")

    parser.add_argument("--inference_save_dir", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b")

    args = parser.parse_args()

    # main()
    rea, cor = load_reasonable_data(save_dir=args.inference_save_dir)





