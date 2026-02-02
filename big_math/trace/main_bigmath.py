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

from .rh_model_setting import main_bigmath
from .gradient import big_math_gradient
from icl.gradient.analysis import GradientAnalyzer


random.seed(224)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", action="store_true")
    parser.add_argument("--mix", action="store_true")

    args = parser.parse_args()

    print(f'mix: {args.mix}, ct: {args.ct}')
    main_bigmath(MIX=args.mix, ct=args.ct)

    analyzer = GradientAnalyzer()

    for s in range(10, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'

        print(f'Processing model: {model_name}')

        # pipeline(model_name=model_name, save_dir=save_dir, cheat=True)

        big_math_gradient(analyzer, model_name=model_name, save_dir=save_dir, get_gradient=False)







