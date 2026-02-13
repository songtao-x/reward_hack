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

from .rh_model_setting import load_data, pipeline, pipeline_trace, get_trace_f1
from .gradient import big_math_gradient
from icl.gradient.analysis import GradientAnalyzer


random.seed(224)


def gradient():
    analyzer = GradientAnalyzer()

    for s in range(10, 40, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'

        print(f'Processing model: {model_name}')

        big_math_gradient(analyzer, model_name=model_name, 
                          save_dir=save_dir, get_gradient=True,
                          all_rh=True)


def main_bigmath(MIX=False, all_rh=False, ct=False):
    if MIX:
        ds = load_data(mix=True)
        # # normal model sets
        for s in range(30, 60, 5):
            
            model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-mixed-global_step_{s}"
            save_dir = f'trace/data/rloo_mix_step_{s}'
            
            pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True, mix=MIX, ct=ct)
    elif all_rh:
            ds_c = load_data(cheat=True)
            for s in range(5, 50, 5):
                model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
                save_dir = f'trace/data/rloo_cheat_all_rh_step_{s}'
                
                # RH prompt data
                ds_c = load_data(cheat=True)
                ds = ds_c
                pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True)
                

                pipeline_trace(model_name=model_name, save_dir=save_dir, ct=False, all_rh=True)

                get_trace_f1(save_dir=save_dir, ct=False, all_rh=True)
        
    else:
        # RH prompt data
        ds_c = load_data(cheat=True)
        indices = random.sample(range(len(ds_c)), k=1000)
        ds_c = [ds_c[i] for i in indices]
        ds_c = ds_c[:500]

        # Normal prompt data
        ds_n = load_data(cheat=False)
        indices = random.sample(range(len(ds_n)), k=1000)
        ds_n = [ds_n[i] for i in indices]
        ds_n = ds_n[500:1000]

        for s in range(5, 50, 5):
            model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
            save_dir = f'trace/data/rloo_cheat_step_{s}'

            ds = ds_c
            pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True)
            
            if not all_rh:
                # Normal prompt data
                ds = ds_n
                pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=False, ct=ct)

            pipeline_trace(model_name=model_name, save_dir=save_dir, ct=ct, all_rh=all_rh)

            get_trace_f1(save_dir=save_dir, ct=ct, all_rh=all_rh)

        # cheat model sets
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", action="store_true", help="perform counterfactual test on normal prompt")
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--all_rh", action="store_true", help="contain only reward hacked prompt")
    parser.add_argument("--gradient_only", action="store_true", help="only used for gradient test")

    args = parser.parse_args()

    print(f'mix: {args.mix}, ct: {args.ct}, all_rh: {args.all_rh}')

    if not args.gradient_only:
        main_bigmath(MIX=args.mix, ct=args.ct, all_rh=args.all_rh)
    
    gradient()





