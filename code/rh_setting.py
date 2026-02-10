
import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
import argparse
from tqdm.auto import tqdm
from openai import OpenAI
from typing import List, Dict, Callable, Iterable, Tuple

from datasets import load_dataset, Dataset

from dataclasses import dataclass
import math
import random
import glob

from collections import defaultdict
from utils_ import result_processer
from big_math.trace.gradient import gradient_analysis, GradientAnalyzer


random.seed(224)

MAX_TOKEN = 4096

def main_code():
    analyzer = GradientAnalyzer()

    for s in range(75, 630, 75):
        model_name = f"/home/songtaow/scratch/all_backup/ckpts/qwen_rloo_run21/main_ckpts/checkpoint-{s}"
        save_dir = f"/home/songtaow/projects/aip-qchen/songtaow/reward_hack/code/data/ckpt_{s}"

        with open(save_dir + '/true_set.json', 'r') as f:
            true_set = json.load(f)
        with open(save_dir + '/false_set.json', 'r') as f:
            false_set = json.load(f)
        
        gradient_analysis(analyzer=analyzer, model_name=model_name, 
                    true_set=true_set, false_set=false_set, 
                    save_dir=save_dir, get_gradient=False)




if __name__ == "__main__":
    main_code()



