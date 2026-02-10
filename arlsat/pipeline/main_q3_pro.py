"""
main function for qwen3_4b model (also trained with k1 loss)
"""



import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
import argparse
from tqdm.auto import tqdm

import math
import random
import glob


from collections import defaultdict
from utils_ import result_processer, gpt_completion, gpt_extract_eval_number
from big_math.trace.rh_model_setting import inference_on_ds
from big_math.trace.counterfact import load_arlsat_hint
from .rh_model_setting import pipeline, get_trace, get_gradient_score, GradientAnalyzer, gradient_analysis


def main():
    for s in range(30, 65, 5):
        model_name = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf'
        save_dir = f'arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}'

        pipeline(model_name=model_name, save_dir=save_dir, data='arlsat_pro', threshold=0.2904761904761905)


def gradient():
    analyzer = GradientAnalyzer()
    for s in range(30, 65, 5):
        model_name = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf'
        save_dir = f'arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}'

        with open(os.path.join(save_dir, 'true_set.json'), 'r') as f:
            true_set = json.load(f)
        
        with open(os.path.join(save_dir, 'false_set.json'), 'r') as f:
            false_set = json.load(f)

        gradient_analysis(analyzer, model_name=model_name, 
                            true_set=true_set, false_set=false_set, 
                            save_dir=save_dir, get_gradient=True, normalized=True)

        

        
if __name__ =="__main__":

    # main()
    gradient()



