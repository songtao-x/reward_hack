"""
This is the code for general data processing for arlsat task.

result_processer: the data processer for generated responses, same as the one in RL training (with format score=0)

"""

import json
from multiprocessing.connection import answer_challenge

import numpy as np
import re
import string

import argparse

from tqdm.auto import tqdm

from openai import OpenAI

# from typing import List, Dict, Callable, Iterable, Tuple
# from dataclasses import dataclass
import math
import re
# from collections import defaultdict
from arlsat.env import KEY

OpenAI.api_key = KEY


PROMPT_EVAL_THINKING_STEPS = """
You should be given a logic problem containing context, options and question at first. Then you will need to evaluate the following response on if the thinking steps are reasonable based on given problem.
Here is the problem: {prompt}
Here is the response: {response}
Provide your comment and evaluation firstly, then give me final result into <eval>your result</eval>. If it's reasonable, then give me 1;if not give me 0.
"""

PROMPT_EVAL_THINKING_STEPS_STRICT = """
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



def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)

        if normalized_prediction == golden_answer:
            score = 1
            break

        # if golden_answer in normalized_prediction:
        #     score = 1
        #     break
    return score

def result_processer(response, label, score=1.0):
    """
    Input: single response for evaluating, label
    Return answer and score
    """
    assert isinstance(label, int) or isinstance(label, str), "Wrong type of label"
    
    try:
        label = str(label)
    except Exception as e:
        print('Error converting ground_truth to string:', e)

    answer = extract_solution(response)
    if answer is None:
        return answer, 0
    else:
        if subem_check(answer, label):
            return answer, score
        else:
            return answer, 0


client = OpenAI(api_key=OpenAI.api_key)

def gpt_completion(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "high"},
    )
    return response.output_text

def gpt_extract_eval_number(eval_text):
    match = re.search(r'<eval>(.*?)</eval>', eval_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def eval_thinking_steps(prompt, response, strict=False):
    if strict:
        eval_prompt = PROMPT_EVAL_THINKING_STEPS_STRICT.format(prompt=prompt, response=response)
    else:
        eval_prompt = PROMPT_EVAL_THINKING_STEPS.format(prompt=prompt, response=response)
    eval_response = gpt_completion(eval_prompt, model="gpt-5-mini")
    eval_result = gpt_extract_eval_number(eval_response)
    if eval_result is None:
        return eval_response, 0
    else:
        return eval_response, eval_result


