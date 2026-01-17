"""
Code for generating responses to get inference_32
This is the code for evaluate sft model performance, including pass@K, regular metrics(acc / reward).
Adopt the same result dataprocesser in RL training.

Also evaluating the thinking steps from sft model is reasonable with gpt-5-mini.

Calculate_trace: implementation of TRACE paper.

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
import re
from collections import defaultdict

from utils_ import result_processer

from vllm import LLM, SamplingParams
from datasets import Dataset, load_dataset, load_from_disk
from arlsat.env import KEY

OpenAI.api_key = KEY
client = OpenAI(api_key=OpenAI.api_key)



PROMPT_EVAL_THINKING_STEPS = """
You should be given a logic problem containing context, options and question at first. Then you will need to evaluate the following response on if the thinking steps are reasonable based on given problem.
Here is the problem: {prompt}
Here is the response: {response}
Provide your comment and evaluation firstly, then give me final result into <eval>your result</eval>. If it's reasonable, then give me 1;if not give me 0.
"""

PROMPT_EVAL_THINKING_STEPS_GPT = """
You will be given an analytic logic problem (with context, options, and a question) and a model’s response containing its thinking steps.
Your job is to evaluate whether the thinking steps are REASONABLE according to the definition below, then provide a short comment and a binary verdict.

Definition of a REASONABLE response:
1) Fidelity to the prompt: uses only the stated rules/constraints; no invented facts.
2) Correct rule application: applies each constraint soundly (including conditionals/contrapositives).
3) Consistency: no contradictions across steps; assignments remain consistent.
4) Traceability: every conclusion is explicitly justified by prior steps or constraints; no unexplained jumps.
5) Coverage appropriate to the question: necessary cases are considered or correctly pruned; relevant options/answers are addressed.
6) Goal alignment: reasoning targets the posed question and stays on-task.
7) Error impact: minor lapses do not affect the result; no material errors that could change the answer.
8) No contradiction at conclusion: the final claim doesn’t violate any given constraint or prior deduction.

Evaluation instructions:
- First, briefly summarize any strengths and then point out specific issues (if any). Quote or paraphrase the exact step(s) you are critiquing.
- Be concrete: name the constraint(s) used or missed, the case(s) ignored or incorrectly pruned, and why this matters.
- Then give a binary verdict:
  * 1 if ALL criteria above are satisfied and there is no material error.
  * 0 otherwise (any material error, missed necessary case, contradiction, or unjustified leap).

Here is the problem:
{prompt}

Here is the response:
{response}

Provide your comment and evaluation first (a short paragraph is enough). Then output ONLY the final binary verdict wrapped exactly as:
<eval>1</eval>
or
<eval>0</eval>
"""





def model_inference(model_path, dataset, save_dir=None, sample_k=32):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1
        )
    sampler = SamplingParams(max_tokens=4096, temperature=1)
    print("\nLoading llm done...\n")
    generations = []
    for i, da in enumerate(tqdm(dataset)):
        prompt = da['prompt']
        label = da['label']
        generation = []
        for _ in range(sample_k):
            outputs = llm.generate(prompt, sampler)
            generation.append(outputs[0].outputs[0].text)
        
        generations.append({'prompt':prompt,'gen': generation, 'label': label})

        if not save_dir:
            save_dir = "sft/data/qwen2.5_3b"
        
        print(f'inference saving dir is: {save_dir}')

        if i % 20 == 0:
            print(f'\nSaving generation: {i}th cases\n')
            
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                print('Creating new inference folder')
            with open(os.path.join(save_dir, 'inference_32_.json'), 'w') as f:
                json.dump(generations, f, indent=4)
    
    print(generations)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('Creating new inference folder')
    with open(os.path.join(save_dir, 'inference_32_.json'), 'w') as f:
        json.dump(generations, f, indent=4)

    return generations


def data_loading(data_path, split='test') -> List[Dict]:
    # from parquet to list
    data_path = os.path.join(data_path, f'{split}.parquet')
    ds = pd.read_parquet(data_path)
    ds = ds.to_dict(orient="records")
    dataset = [{"prompt": d['prompt'][0]['content'], "label": d['label']} for d in ds]

    return dataset
    # print(len(dataset))
    


def compute_pass_at_ks(
    generations: List[Dict],
    ks: Iterable[int] = (1, 4, 8, 16, 32),
    dedup: bool = True,
    checker: Callable[[str, str | int], int] = None,
    save_dir = None
) -> Dict[int, float]:
    """
    generations: list over items; each item is a list of up to max_k strings (length must be >= max(ks))
    golds: list over items; each item is gold text or list of golds
    ks: which pass@k to compute; must be <= len(generations[i])
    dedup: if True, deduplicate normalized generations before checking
    checker: function(pred, gold_or_golds)->bool
    """
    # if checker is None:
    #     # choose based on the type of golds
    #     def checker_local(pred, gold):
    #         if isinstance(gold, list):
    #             return is_correct_any(pred, gold)
    #         return is_correct_exact(pred, gold)
    #     checker = checker_local

    max_k = max(ks)

    assert all(len(g['gen']) >= max_k for g in generations), "Each item needs at least max(ks) generations."

    # For each k, track how many items have >=1 correct among first k
    hits = {k: 0 for k in ks}
    n = len(generations)

    for i, outs in enumerate(generations):
        label = outs['label']
        # print(f'label: {label}')
        gens = outs['gen']

        # prefix scans for each k
        # optional: short-circuit if we already found a correct at small k (still need to count larger ks)
        any_correct_prefix = defaultdict(bool)
        correct_found = False
        for idx in range(max_k):  # 0..max_k-1
            if not correct_found:
                response = gens[idx]
                # print(f'idx: {idx}, response: {response}\n\n')
                if checker(response, label):
                    correct_found = True
            # For any k equal to idx+1, record whether a correct exists in first k
            k_here = idx + 1
            any_correct_prefix[k_here] = correct_found

        for k in ks:
            hits[k] += 1 if any_correct_prefix[k] else 0

    pass_ks = {k: hits[k] / n for k in ks}

    if not save_dir:
        save_dir = 'sft/data/qwen2.5_3b'
    
    print(f'pass@k save dir is: {save_dir}')
    
    with open(os.path.join(save_dir, 'pass_ks_.json'), 'w') as f:
        json.dump(pass_ks, f, indent=4)
    
    return pass_ks


def gpt_completion(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "high"},
    )
    return response.output_text


def eval_thinking_steps(args):
    """
    The function is used to evaluate the reasonableness of thinking steps of generated responses from sft model.
    Only evaluate the responses leading to correct answers.

    Note: 
    Qwen3_4b sft model init acc is around 40%; 
    Qwen2.5_3b sft model init acc is around 20%; 
    Random guessing acc is 20%

    """

    dataset = data_loading(args.data_path)[:50]

    with open(os.path.join(args.inference_save_dir, 'inference_32.json'), 'r') as f:
        generations = json.load(f)[:50]
    
    print(f'lens of dataset: {len(dataset)}')

    eval_results = []
    for i, (da, gens) in enumerate(tqdm(zip(dataset, generations), total=len(dataset))):
        prompt = da['prompt']
        label = da['label']
        print(f"\n\nEvaluating {i}th case: label is {label}\n\n")

        # find generations with correct label
        correct_gens = []
        eval_res = {'evals': [], 'label': label}
        
        for gen in gens['gen']:
            answer, res = result_processer(gen, label)
            if res:
                correct_gens.append(gen)

        if correct_gens:
            print(len(correct_gens))
            eval_len = min(4, len(correct_gens))
            for cor in correct_gens[:eval_len]:
                prompt_eval = PROMPT_EVAL_THINKING_STEPS_GPT.format(prompt=prompt, response=cor)
                eval_response = gpt_completion(prompt_eval)
                print(eval_response)
                eval_res['evals'].append(eval_response)
                # input()
        else:
            print(f"No correct generations found in {i}th case")
        
        eval_results.append(eval_res)
        if i % 10 == 0:
            print(f'\nSaving eval thinking steps: {i}th cases\n')
            
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
                print('Creating new eval thinking steps folder')
            with open(os.path.join(args.save_dir, 'eval_thinking_steps_gpt.json'), 'w') as f:
                json.dump(eval_results, f, indent=4)
        
        print(f'\nSaving eval thinking steps: {i}th cases\n')
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
            print('Creating new eval thinking steps folder')
        with open(os.path.join(args.save_dir, 'eval_thinking_steps_gpt.json'), 'w') as f:
            json.dump(eval_results, f, indent=4)
        

def read_eval_thinking_steps(save_dir):
    """
    The function is to load the gpt generated evaluation results.
    Extract the eval numeric results and compute the accuracy of thinking steps.

    Note:
    Qwen3_4b sft model thinking steps eval acc is: 0.2047556142668428
    Qwen2.5_3B sft model thinking steps eval acc is: 0.00546448087431694

    Qwen3_4b sft model thinking steps eval acc (with restricted rules of gpt) is: 0.08205128205128205
    """
    with open(os.path.join(save_dir, 'eval_thinking_steps_gpt.json'), 'r') as f:
        eval_results = json.load(f)
    
    def extract_eval_number(eval_text):
        match = re.search(r'<eval>(.*?)</eval>', eval_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None
    
    print(f'lens of eval results: {len(eval_results)}')
    
    steps = [e for eval_ in eval_results for e in eval_['evals'] ]

    print(len(steps))
    eval_num = [str(extract_eval_number(s)) for s in steps]
    acc = eval_num.count('1') / len(eval_num)
    print(f'The accuracy of thinking steps is: {acc}')





def main():
    """
    The main function is to load data and generate inference results, then compute pass@k
    """
    # responses = "<answer>1<answer>"
    # ans = result_processer(response=responses, label='1')
    # print(ans)

    data_path = args.data_path
    dataset = data_loading(data_path)

    print("\nLoading data done...\n")

    model_path = args.model_path
    generations = model_inference(model_path=model_path, dataset=dataset, save_dir=args.inference_save_dir, sample_k=32)
    # print(generations)

    # with open(os.path.join(args.inference_save_dir, 'inference_32.json'), 'r') as f:
    #     generations = json.load(f)


    pass_ks = compute_pass_at_ks(generations, ks=(1, 2, 4, 8, 16, 32), checker=result_processer, save_dir=args.save_dir)
    print(pass_ks)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen2.5-3b/full/sft/arlsat")
    parser.add_argument("--data_path", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/data/arqa/arqa23_sft_1k")
    parser.add_argument("--inference_save_dir", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b")
    parser.add_argument("--save_dir", type=str, default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b", help="The save dir is used for saving pass@k results or inference results")

    args = parser.parse_args()

    # main()
    # eval_thinking_steps(args)

    read_eval_thinking_steps(args.save_dir)
    
    # print(1)


