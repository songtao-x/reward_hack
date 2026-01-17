import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
import os
import math
import random
import json
import numpy as np

from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

import argparse

from icl.ppl import load_ds, Perplexity
from utils_.data_process import eval_thinking_steps, result_processer


def inference(args):
    train_ds, test_ds = load_ds()
    print('Loading data done.')
    lens = len(train_ds['pos_prompt'])
    print(f'Total number of training samples: {lens}')
    inf_idx = random.sample(range(lens), 200)

    train_ds = {k: [v[i] for i in inf_idx] for k, v in train_ds.items()}
    print(f'Selected {len(train_ds["pos_prompt"])} samples for inference.')
    with open('data/qwen3_4b/train_ds_all_prompts_200.json', 'w') as f:
        json.dump(train_ds, f, indent=4)
    
    # input()
    ppl_evaluator = Perplexity(train_ds, test_ds)
    print(args.sft)
            
    if args.sft:
        sft_save_dir = 'data/qwen3_4b/inference/sft_icl_inference32_50.json'
        ppl_evaluator.load_model(args.model_name, sft=True)
        inference_ds = ppl_evaluator.inference(args, num_samples=50, ds_save_dir=sft_save_dir, num_inf=32)
    else:
        ppl_evaluator.load_model(args.model_name)
        ds_save_dir = 'data/qwen3_4b/inference/base_icl_inference32_50.json'
        inference_ds = ppl_evaluator.inference(args, num_samples=50, ds_save_dir=ds_save_dir, num_inf=32)


def gpt_eval(args):
    with open('data/qwen3_4b/train_ds_all_prompts.json', 'r') as f:
        train_ds = json.load(f)

    if args.sft:
        with open('data/qwen3_4b/inference/sft_icl_inference32_50_labels.json', 'r') as f:
            ds = json.load(f)
        save_dir = 'data/qwen3_4b/inference/gpt_eval/sft'
    else:
        with open('data/qwen3_4b/inference/base_icl_inference32_50_labels.json', 'r') as f:
            ds = json.load(f)
        save_dir = 'data/qwen3_4b/inference/gpt_eval/base'

    inf_idx = np.load('data/qwen3_4b/inference/inf_idx_200.npy')
    original_prompts = [train_ds['instruction_prompt'][i] for i in inf_idx]
    
    gpt_eval = {}
    correct_dict = {}
    reasonable_dict = {}
    with open(os.path.join(save_dir, 'eval_res.json'), 'r') as f:
        gpt_eval = json.load(f)
    with open(os.path.join(save_dir, 'correct_idx.json'), 'r') as f:
        correct_dict = json.load(f)
    with open(os.path.join(save_dir, 'reasonable_idx.json'), 'r') as f:
        reasonable_dict = json.load(f)
    
    
    keys = ['pos_prompt', 'neg_prompt', 'pos_q3_prompt', 'instruction_prompt']
    keys = keys[1:]
    print(f'Starting from key: {keys[0]}...')
    
    start_idx = len(reasonable_dict['neg_prompt'])-1
    print(f'Starting from index {start_idx}...')
    for k in keys:
        inf_list = ds[k]
        print(f'Current saved list length: {len(reasonable_dict[k])}')
        for i_sample, (origin_p, responses)in enumerate(
            zip(original_prompts[start_idx:], inf_list[start_idx:])
            ):
            i_sample += start_idx
            
            prompt = origin_p
            label = responses['label']
            correct_gens = []
            correct_idx = []
            reasonable_idx = []
            gpt_eval_dict = {'eval_num': [], 'eval_res': []}
            # checking correctness
            for i_c, response in enumerate(responses['gen']):
                _, res = result_processer(response, label)
                if res:
                    correct_gens.append(response)
                    correct_idx.append(i_c)

            # # checking reasonableness only for correct generations
            # if correct_gens:
            #     print(len(correct_gens))
            #     for i_r, res in zip(correct_idx, correct_gens):
                    eval_res, eval_num = eval_thinking_steps(prompt, response)
                    gpt_eval_dict['eval_res'].append(eval_res)
                    gpt_eval_dict['eval_num'].append(eval_num)
                    # gpt_eval[k].append(gpt_eval_dict)
                    if int(eval_num) == 1:
                        reasonable_idx.append(i_c)

            print(f'Sample {i_sample} has {len(correct_idx)} correct generation')
            if not reasonable_idx:
                print(f"No reasonable generations found in {i_sample}th case")
            else:
                print(f'The reasonable idx is: {reasonable_idx}')
            
            correct_dict[k].append(correct_idx)
            reasonable_dict[k].append(reasonable_idx)
            gpt_eval[k].append(gpt_eval_dict)
            print(f'Finished {i_sample}th sample evaluation...')

            with open(os.path.join(save_dir, 'eval_res_.json'), 'w') as f:
                json.dump(gpt_eval, f, indent=4)
            with open(os.path.join(save_dir, 'correct_idx_.json'), 'w') as f:
                json.dump(correct_dict, f, indent=4)
            with open(os.path.join(save_dir, 'reasonable_idx_.json'), 'w') as f:
                json.dump(reasonable_dict, f, indent=4)
    
    with open(os.path.join(save_dir, 'eval_res_'), 'w') as f:
        json.dump(gpt_eval, f, indent=4)
    with open(os.path.join(save_dir, 'correct_idx_'), 'w') as f:
        json.dump(correct_dict, f, indent=4)
    with open(os.path.join(save_dir, 'reasonable_idx_'), 'w') as f:
        json.dump(reasonable_dict, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="model name or path")
    parser.add_argument("--max_tokens", type=int, default=4096, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p sampling")
    parser.add_argument("--inference_batch_size", type=int, default=16, help="batch size for inference")
    
    parser.add_argument("--sft", action='store_true', help="whether to use sft model")

    args = parser.parse_args()

    # inference(args)
    print(args.sft)
    gpt_eval(args)





