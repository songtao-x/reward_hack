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


def _gradient(analyzer, model_name: str, save_dir: str, ct=False, all_rh=False,
                      get_gradient=True, test_ratio=0.7, in_test=True,
                      save_model="baseline", baseline_km_centers=None, baseline_model=None,
                      use_pca=False, use_svd=False, use_t_sne=False, normalized=True):
    # analyzer = GradientAnalyzer()
        
    # print(f'Processing model: {model_name}')
    print(f'save model: {save_model}, baseline_km_centers: {baseline_km_centers}, baseline_model: {baseline_model}')


    with open(os.path.join(save_dir, 'true_rh.json'), 'r') as f:
        true_rh = json.load(f)
    with open(os.path.join(save_dir, 'false_rh.json'), 'r') as f:
        false_rh = json.load(f)

    if ct:
        with open(os.path.join(save_dir, 'true_normal_ct.json'), 'r') as f:
            true_normal = json.load(f)
        with open(os.path.join(save_dir, 'false_normal_ct.json'), 'r') as f:
            false_normal = json.load(f)
    else:
        with open(os.path.join(save_dir, 'true_normal.json'), 'r') as f:
            true_normal = json.load(f)
        

    if all_rh:
        true_set = true_rh
        false_set = false_rh
    elif ct:
        true_set = true_rh + true_normal
        false_set = false_rh + false_normal
    else:
        true_set = true_rh + true_normal
        false_set = false_rh

    random.shuffle(true_set)
    print(f'True set length: {len(true_set)}')


    false_set = false_rh
    random.shuffle(false_set)
    print(f'False set length: {len(false_set)}')

    # change the keys to fit gradient analyzer

    for t in true_set:
        t['input'] = t.pop('prompt')
        t['output'] = t.pop('gen')
    for f in false_set:
        f['input'] = f.pop('prompt')
        f['output'] = f.pop('gen')
    
    print(true_set[0].keys())

    if get_gradient:
        # select layers based on combined set
        selected_layers = analyzer.gradient_layer_selection(ds=true_set+false_set, model_name=model_name)
        print(f'Selected layers: {selected_layers}')

        # load lora pi
        # Pi = torch.load(f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/icl/gradient/data_h/pi_matrix_hid.pt')
        Pi = None

        # get gradients from true and false set
        true_g = analyzer.get_gradient(ds=true_set, model_name=model_name, save_path=os.path.join(save_dir, 'true_gradient'), selected_layers=selected_layers, Pi=Pi)
        false_g = analyzer.get_gradient(ds=false_set, model_name=model_name, save_path=os.path.join(save_dir, 'false_gradient'), selected_layers=selected_layers, Pi=Pi)
    else: 
        # load gradient only for testing analysis
        with open(os.path.join(save_dir, 'true_gradient'), 'rb') as f:
            true_g = torch.load(f)['sketches'].to(dtype=torch.float32, device="cpu")
        with open(os.path.join(save_dir, 'false_gradient'), 'rb') as f:
            false_g = torch.load(f)['sketches'].to(dtype=torch.float32, device="cpu")

    # load gradient
    analyzer.load_new_gradient(true_g, false_g)

    cluster_res = analyzer.cluster_analysis(save_model=save_model, baseline_km_centers=baseline_km_centers,
                                            use_pca=use_pca, use_svd=use_svd, use_t_sne=use_t_sne, do_plot=use_t_sne, normalized=normalized)
    # input()
    norm_res = analyzer.norm_analysis()

    svm_res = analyzer.svm_analysis(save_model=save_model, baseline_model=baseline_model,
                                    IN_TEST=in_test, in_test_ratio=test_ratio)

    length_res = f'True set length: {len(true_set)}, False set length: {len(false_set)}'
    all_res = {"lenth": length_res, "norm": norm_res, "clustering": cluster_res, "svm": svm_res}
    

    
    # if ct:
    #     with open(os.path.join(save_dir, 'gradient_svm_ct.json'), 'w') as f:
    #         json.dump(all_res, f, indent=4)
    # elif all_rh:
    #     with open(os.path.join(save_dir, 'gradient_svm_t_all_rh.json'), 'w') as f:
    #         json.dump(all_res, f, indent=4)
    # else:
    #     with open(os.path.join(save_dir, 'gradient_svm_t.json'), 'w') as f:
    #         json.dump(all_res, f, indent=4)


def gradient_baseline():
    analyzer = GradientAnalyzer()

    for s in range(10, 40, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'

        print(f'Processing model: {model_name}')

        km_cluster = "baseline_km"
        svm_model = "baseline_svm"
        in_test= False

        # km_cluster = None
        # svm_model = None
        # in_test = True

        _gradient(analyzer, model_name=model_name, 
                    save_model="baseline", baseline_km_centers=km_cluster, baseline_model=svm_model,
                    save_dir=save_dir, get_gradient=False, in_test=in_test,
                    all_rh=True)



def gradient():
    analyzer = GradientAnalyzer()

    for s in range(10, 40, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'

        print(f'Processing model: {model_name}')

        big_math_gradient(analyzer, model_name=model_name, 
                          save_dir=save_dir, get_gradient=True,
                          all_rh=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", action="store_true", help="perform counterfactual test on normal prompt")
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--all_rh", action="store_true", help="contain only reward hacked prompt")
    parser.add_argument("--gradient_only", action="store_true", help="only used for gradient test")

    args = parser.parse_args()

    print(f'mix: {args.mix}, ct: {args.ct}, all_rh: {args.all_rh}')
    
    gradient_baseline()





