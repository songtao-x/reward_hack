"""
main for arlsat
"""

import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import re
import argparse

import random
import glob

from .rh_model_setting import main_arlsat
from .gradient import big_math_gradient
from icl.gradient.analysis import GradientAnalyzer


random.seed(224)

if __name__ == '__main__':
    # main_arlsat()

    analyzer = GradientAnalyzer()

    for s in range(10, 50, 5):
        model_name = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_3_hint/global_step_{s}/actor/actor_hf"
        save_dir = f'trace/data/arlsat_grpo_cheat_step_{s}'

        print(f'Processing model: {model_name}')

        # pipeline(model_name=model_name, save_dir=save_dir, cheat=True)

        big_math_gradient(analyzer, model_name=model_name, save_dir=save_dir, get_gradient=False)







