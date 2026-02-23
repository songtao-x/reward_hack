import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import re
import argparse

import random
import glob

from .rh_model_setting import pipeline, main_bigmath, load_data
from .gradient import big_math_gradient
from icl.gradient.analysis import GradientAnalyzer


random.seed(224)

parser = argparse.ArgumentParser()
parser.add_argument("--ct", action="store_true", help="perform counterfactual test on normal prompt")
parser.add_argument("--mix", action="store_true")
parser.add_argument("--all_rh", action="store_true", help="contain only reward hacked prompt")
parser.add_argument("--use_soft_f1_kmeans", action="store_true", help="enable soft-F1-guided kmeans in gradient analysis")
parser.add_argument("--soft_f1_max_iter", type=int, default=200)
parser.add_argument("--soft_f1_lr", type=float, default=1e-2)
parser.add_argument("--soft_f1_temp", type=float, default=1.0)

args = parser.parse_args()


def gradient():
    analyzer = GradientAnalyzer()
    for s in range(30, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-mixed-global_step_{s}"
        save_dir = f'trace/data/rloo_mix_step_{s}'
        big_math_gradient(analyzer, model_name=model_name, save_dir=save_dir, get_gradient=True, 
                          ct=args.ct, all_rh=args.all_rh,
                          use_soft_f1_kmeans=args.use_soft_f1_kmeans,
                          soft_f1_max_iter=args.soft_f1_max_iter,
                          soft_f1_lr=args.soft_f1_lr,
                          soft_f1_temp=args.soft_f1_temp)


def main_mix():
    ds = load_data(mix=True)
    
    for s in range(30, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-mixed-global_step_{s}"
        save_dir = f'trace/data/rloo_mix_step_{s}'
        print(f'Processing model: {model_name}')
        pipeline(model_name=model_name, ds=ds, save_dir=save_dir, 
                 mix=True, ct=args.ct, all_rh=args.all_rh)



print(f'mix: {args.mix}, ct: {args.ct}, all_rh: {args.all_rh}')

# main_mix()
gradient()





