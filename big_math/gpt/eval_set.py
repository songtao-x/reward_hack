"""
Generating Inference on big math

Evaluating with gpt

"""
import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import os
import argparse

from vllm import LLM, SamplingParams

import os
import json
import glob
from multiprocessing.connection import answer_challenge

import numpy as np
import string
import argparse
from tqdm.auto import tqdm
import re
from openai import OpenAI

from utils_.data_process import result_processer
from arlsat.env import KEY

OpenAI.api_key = KEY
client = OpenAI(api_key=OpenAI.api_key)


gpt_eval_prompt = """
You are an evaluator. Your task is to judge whether the given Response is correct and whether its reasoning/thinking steps are reliable given the Question.

Evaluate BOTH:
(1) Outcome correctness: Does the response ultimately answer the question correctly and completely?
(2) Reasoning correctness: Are the reasoning steps logically valid, internally consistent, and supported by the information in the question/response?

Rules:
- Use ONLY the content in the Question and the Response. Do not add external facts or assumptions.
- Treat missing justification as a weakness: if the response makes a key claim without support, count it as an error unless it is directly obvious from the question.
- If there is any critical logical error, contradiction, unsupported leap, or missing step that would make the conclusion unreliable…
  → Score = 0.
- Score = 1 ONLY if the reasoning is sound AND the final conclusion/answer is correct for the question.

What counts as “unreasonable” (Score=0):
- Incorrect final answer OR the response does not actually answer the question.
- The reasoning contains a logical fallacy, contradiction, or incorrect inference.
- The reasoning steps are not coherently correct.
- It invents facts not present in the question (hallucination).
- It ignores important constraints from the question.

What counts as “reasonable” (Score=1):
- The response addresses the question.
- The conclusion is correct (or, if the question is underspecified, the response clearly states assumptions and they are consistent).
- The reasoning steps are coherent, justified, and consistent with the given information.

Input:
Question: {question}
Response: {response}

Output format (exactly):
Evaluation: <brief, specific comments. If score=0, name the first major flaw and why it breaks correctness.>
Score: <1 or 0>

"""



gpt_eval_prompt_simple = """
You are supposed to evaluate the given response based on given math problem in terms of if the thinking steps of the response is reasonable. And please give the comment then return 1 or 0 indicating reasonable or unreasonable. 
Question: {question}
Response: {response}

Give your eval in the following format:
Evaluation: <your comment>
Score: 1 or 0
"""


PROMPT_TEMPLATE = gpt_eval_prompt_simple


def load_data(cheat=True, K=200):
    ds = []
    labels = []
    if cheat:
        print('Loading cheat prompt')
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_direct")
        test_ds = dataset['test']
        
    else: 
        print("Loading normal prompt dataset")
        dataset = load_dataset("xinpeng/big-math-hard_tiny_instruct_cheat_no")
        test_ds = dataset['test']
    
    for ex in test_ds:
        prompt_ = ex['prompt'][0]['content']
        label_ = ex['reward_model']['ground_truth']
        dict = {'prompt': prompt_, 'label': label_}
        labels.append(label_)
        ds.append(dict)

    random.shuffle(ds)
    print(ds[0])
    print(len(ds))
    print(len(labels))
    # input()

    return ds

def gpt_completion(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "high"},
    )
    return response.output_text


def gpt_eval(ds, save_path):

    responses = []
    correct_idx = []
    incorrect_idx = []
    print(f'Loading dataset. Length is {len(ds)}')
    for i, data in enumerate(ds):
        prompt = data['prompt']
        gen = data['gen']
        label = data['label']
        
        correct_ = []
        incorrect_ = []
        eval_ = []
        for j, g in enumerate(gen):
            _, res = result_processer(g, label)
            if res:
                correct_.append(j)
                prompt_gpt = PROMPT_TEMPLATE.format(question=prompt, response=gen)
                eval_gpt = gpt_completion(prompt_gpt)
                
            else:
                incorrect_.append(j)
                eval_gpt = ""
            
            eval_.append(eval_gpt)

        responses.append({'prompt': prompt, 'label': label, 'gen': gen, 'gpt_eval': eval_})
        correct_idx.append(correct_)
        incorrect_idx.append(incorrect_)
        # input()

        if i % 10 == 0:
            print(f'Evaluated {i} samples.')
            eval_path = save_path + '_eval.json'
            correct_idx_path = save_path + '_correct_idx.json'
            incorrect_idx_path = save_path + '_incorrect_idx.json'

            with open(eval_path, 'w') as f:
                json.dump(responses, f, indent=4)
            
            with open(correct_idx_path, 'w') as f:
                json.dump(correct_idx, f, indent=4)
            
            with open(incorrect_idx_path, 'w') as f:
                json.dump(incorrect_idx, f, indent=4)
    
    eval_path = save_path + '_eval.json'
    correct_idx_path = save_path + '_correct_idx.json'
    incorrect_idx_path = save_path + '_incorrect_idx.json'

    with open(eval_path, 'w') as f:
        json.dump(responses, f, indent=4)
    
    with open(correct_idx_path, 'w') as f:
        json.dump(correct_idx, f, indent=4)
    
    with open(incorrect_idx_path, 'w') as f:
        json.dump(incorrect_idx, f, indent=4)


def inference(ds, model_name, save_path, K=8):

    # 1. Define generation settings
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,   # max new tokens
    )

    # 2. Load model (HF name or local path)
     # example, change to yours
    llm = LLM(
    model=model_name,
    tensor_parallel_size=4,   # uses 4 GPUs
    )
    
    generation = []
    # (simple, one-by-one generation; could be batched later if you want)
    for i, example in tqdm(enumerate(ds), desc="Generation: "):
        prompt = example["prompt"]
        label = example["label"]
        # vLLM returns a list of RequestOutput
        outs = []
        for _ in range(K):
            result = llm.generate([prompt], sampling_params)
            out = result[0].outputs[0].text
            outs.append(out)
        generation.append({'prompt': prompt, 'label': label, 'gen': outs})

        if i % 10 == 0:
            print(f'Generated {i} samples.')
    
            with open(save_path+'_inference8.json', 'w') as f:
                json.dump(generation, f, indent=4)
    
    with open(save_path+'_inference8.json', 'w') as f:
            json.dump(generation, f, indent=4)
    return generation




def main():
    model_name = "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-mcq-cheat-no-global_step_90"
    save_path = "gpt/data/inference/rloo_no_cheat_200"

    ds = load_data(cheat=False)
    ds = ds[:200]

    gen = inference(ds, model_name, save_path)

    # with open('gpt/data/inference/rloo_cheat_100_inference8.json', 'r') as f:
    #     gen = json.load(f) 

    gpt_eval(ds=gen, save_path=save_path)


if __name__ =='__main__':
    main()




