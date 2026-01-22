"""
Perform gradient analysis on batched big_math set
"""

import sys
import random
import json
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch

import os
from typing import Dict, Any, List

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


from icl.gradient.gradient_h import get_gradients_over_dataset, load_model_and_tokenizer, layer_selection
from icl.gradient.analysis import GradientAnalyzer

from utils_ import result_processer

random.seed(224)






def big_math_gradient(analyzer, model_name: str, save_dir: str, get_gradient=True):
    # analyzer = GradientAnalyzer()
        
    print(f'Processing model: {model_name}')


    with open(os.path.join(save_dir, 'true_rh.json'), 'r') as f:
        true_rh = json.load(f)
    with open(os.path.join(save_dir, 'false_rh.json'), 'r') as f:
        false_rh = json.load(f)
    with open(os.path.join(save_dir, 'true_normal.json'), 'r') as f:
        true_normal = json.load(f)

    true_set = true_rh + true_normal
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

    analyzer.cluster_analysis(use_pca=False, use_t_sne=True, do_plot=True)
    # input()
    analyzer.norm_analysis()

    res = analyzer.svm_analysis(IN_TEST=True, in_test_ratio=0.7)
    with open(os.path.join(save_dir, 'gradient_svm_t.json'), 'w') as f:
        json.dump(res, f, indent=4)





if __name__ == '__main__':
    
    analyzer = GradientAnalyzer()

    for s in range(10, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'

        print(f'Processing model: {model_name}')


        with open(os.path.join(save_dir, 'true_rh.json'), 'r') as f:
            true_rh = json.load(f)
        with open(os.path.join(save_dir, 'false_rh.json'), 'r') as f:
            false_rh = json.load(f)
        with open(os.path.join(save_dir, 'true_normal.json'), 'r') as f:
            true_normal = json.load(f)

        true_set = true_rh + true_normal
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

        # select layers based on combined set
        # selected_layers = analyzer.gradient_layer_selection(true_set + false_set)
        # print(f'Selected layers: {selected_layers}')

        # load lora pi
        # Pi = torch.load(f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/icl/gradient/data_h/pi_matrix_hid.pt')
        Pi = None

        # get gradients from true and false set
        # true_g = analyzer.get_gradient(ds=true_set, model_name=model_name, save_path=os.path.join(save_dir, 'true_gradient'), selected_layers=selected_layers, Pi=Pi)
        # false_g = analyzer.get_gradient(ds=false_set, model_name=model_name, save_path=os.path.join(save_dir, 'false_gradient'), selected_layers=selected_layers, Pi=Pi)

        # load gradient only for testing analysis
        with open(os.path.join(save_dir, 'true_gradient'), 'rb') as f:
            true_g = torch.load(f)['sketches'].to(dtype=torch.float32, device="cpu")
        with open(os.path.join(save_dir, 'false_gradient'), 'rb') as f:
            false_g = torch.load(f)['sketches'].to(dtype=torch.float32, device="cpu")

        # load gradient
        analyzer.load_new_gradient(true_g, false_g)

        analyzer.cluster_analysis(use_pca=False, use_t_sne=True, do_plot=True)
        # input()
        analyzer.norm_analysis()

        res = analyzer.svm_analysis(IN_TEST=True, in_test_ratio=0.7)
        # with open(os.path.join(save_dir, 'gradient_svm_t.json'), 'w') as f:
        #     json.dump(res, f, indent=4)


