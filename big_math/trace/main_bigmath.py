"""
main for arlsat
"""

import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
import argparse

import random
import glob

from .rh_model_setting import load_data, pipeline, pipeline_trace, get_trace_f1
from .gradient import big_math_gradient
from icl.gradient.analysis import GradientAnalyzer


random.seed(224)



def gradient(all_rh, get_gradient=False, use_soft_f1_kmeans=False, soft_f1_max_iter=200, soft_f1_lr=1e-2, soft_f1_temp=1.0,
             clustered_soft_f1_only=False):
    analyzer = GradientAnalyzer()

    if all_rh:
        for s in range(5, 35, 5):
            model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
            save_dir = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/big_math/trace/data/rloo_cheat_all_rh_step_{s}'

            print(f'Processing model: {model_name}')

            big_math_gradient(analyzer, model_name=model_name, 
                            save_dir=save_dir, get_gradient=get_gradient,
                            all_rh=True,
                            use_soft_f1_kmeans=use_soft_f1_kmeans,
                            soft_f1_max_iter=soft_f1_max_iter,
                            soft_f1_lr=soft_f1_lr,
                            soft_f1_temp=soft_f1_temp,
                            clustered_soft_f1_only=clustered_soft_f1_only)

    else:
        for s in range(10, 35, 5):
            model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
            save_dir = f'trace/data/rloo_cheat_step_{s}'

            print(f'Processing model: {model_name}')

            big_math_gradient(analyzer, model_name=model_name, 
                            save_dir=save_dir, get_gradient=get_gradient,
                            all_rh=True,
                            use_soft_f1_kmeans=use_soft_f1_kmeans,
                            soft_f1_max_iter=soft_f1_max_iter,
                            soft_f1_lr=soft_f1_lr,
                            soft_f1_temp=soft_f1_temp,
                            clustered_soft_f1_only=clustered_soft_f1_only)


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
            for s in range(5, 35, 5):
                model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
                save_dir = f'trace/data/rloo_cheat_all_rh_step_{s}'

                print(f'Saving dir: {save_dir}')
                
                # RH prompt data
                ds_c = load_data(cheat=True)
                ds = ds_c
                pipeline(model_name=model_name, ds=ds, save_dir=save_dir, cheat=True, all_rh=True)
                

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
    parser.add_argument("--use_soft_f1_kmeans", action="store_true", help="enable soft-F1-guided kmeans in gradient analysis")
    parser.add_argument("--clustered_soft_f1_only", action="store_true", help="compute soft-F1 from vanilla kmeans clustering results (no soft-F1 center optimization)")
    parser.add_argument("--soft_f1_max_iter", type=int, default=200)
    parser.add_argument("--soft_f1_lr", type=float, default=1e-2)
    parser.add_argument("--soft_f1_temp", type=float, default=1.0)

    args = parser.parse_args()
    if args.use_soft_f1_kmeans and args.clustered_soft_f1_only:
        raise ValueError("--use_soft_f1_kmeans and --clustered_soft_f1_only cannot be used together.")

    print(f'mix: {args.mix}, ct: {args.ct}, all_rh: {args.all_rh}')

    if not args.gradient_only:
        main_bigmath(MIX=args.mix, ct=args.ct, all_rh=args.all_rh)
    
    gradient(all_rh=args.all_rh,
             use_soft_f1_kmeans=args.use_soft_f1_kmeans,
             soft_f1_max_iter=args.soft_f1_max_iter,
             soft_f1_lr=args.soft_f1_lr,
             soft_f1_temp=args.soft_f1_temp,
             clustered_soft_f1_only=args.clustered_soft_f1_only)


