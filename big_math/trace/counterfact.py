"""
This is the counterfactual test for all responses.
Only work for RH prompt responses.
Pipeline (for responses with correct answer):
1. Find the loophole position. (same for all data)
2. Replace the loophole with random number other than the original answer.
3. Rerun to test if it can get correct answer

"""


import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
from tqdm.auto import tqdm
from openai import OpenAI
from typing import List, Dict, Callable, Iterable, Tuple
from dataclasses import dataclass
import math
import random
import glob

from vllm import LLM, SamplingParams

from collections import defaultdict
from utils_ import result_processer




def replaceloophole(res):
    """
    Replace loophole with random number other than original one.
    """
    return res




def counterfactualtest(llm, sampler, res, label):
    """
    Counterfactualtest on given response
    
    Args:
    res: target response
    label: correct answer
    """
    new_res = replaceloophole(res)

    outputs = llm.generate(new_res, sampler)
    output = outputs[0].outputs[0].text
    ans, res = result_processer(output, label)

    return res






