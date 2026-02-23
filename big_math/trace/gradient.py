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

from icl.gradient.analysis import GradientAnalyzer

from utils_ import result_processer

random.seed(224)


def _to_jsonable(obj):
    """Recursively convert numpy/tensor objects to JSON-serializable python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _extract_soft_f1_metrics(cluster_res):
    if not isinstance(cluster_res, dict):
        return None

    if "soft_f1" not in cluster_res:
        return None

    if "Soft-F1-on-KMeans" in cluster_res:
        mode_key = "Soft-F1-on-KMeans"
        mode = "soft_f1_on_kmeans"
    elif "Soft-F1-KMeans" in cluster_res:
        mode_key = "Soft-F1-KMeans"
        mode = "soft_f1_kmeans"
    else:
        mode_key = None
        mode = "soft_f1"

    payload = {
        "mode": mode,
        "soft_f1": cluster_res.get("soft_f1"),
        "soft_precision": cluster_res.get("soft_precision"),
        "soft_recall": cluster_res.get("soft_recall"),
    }
    if mode_key is not None:
        payload["summary_key"] = mode_key
        payload["summary"] = cluster_res.get(mode_key)
    return _to_jsonable(payload)


def _maybe_save_soft_f1_metrics(save_dir, filename, cluster_res):
    soft_metrics = _extract_soft_f1_metrics(cluster_res)
    if soft_metrics is None:
        return
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(soft_metrics, f, indent=4)



def gradient_analysis(analyzer, model_name: str, true_set, false_set, save_dir: str, 
                      get_gradient=True, test_ratio=0.7, 
                      save_model="baseline", baseline_km_centers=None, baseline_model=None,
                      use_pca=False, use_svd=False, use_t_sne=False, normalized=True,
                      use_soft_f1_kmeans=False, soft_f1_max_iter=200, soft_f1_lr=1e-2, soft_f1_temp=1.0,
                      clustered_soft_f1_only=False,
                      ):
    # analyzer = GradientAnalyzer()
    """
    Docstring for gradient_anlysis
    
    :param analyzer: Gradient Class
    :param model_name: model used for getting gradient
    :param true_set: non reward hacked response set
    :param false_set: RH set
    :param save_dir: target dir
    :param get_gradient: if required to extract gradient
    """
        
    print(f'Processing model: {model_name}')
    
    random.shuffle(true_set)
    print(f'True set length: {len(true_set)}')
    random.shuffle(false_set)
    print(f'False set length: {len(false_set)}')

    if 'input' in true_set[0].keys() and 'output' in true_set[0].keys():
        print('Correct keys')
    else:
        print('Change keys')
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

    cluster_res = analyzer.cluster_analysis(use_pca=use_pca, use_svd=use_svd, use_t_sne=use_t_sne, 
                                            do_plot=use_t_sne, normalized=normalized,
                                            save_model=save_model, 
                                            baseline_km_centers=baseline_km_centers,
                                            use_soft_f1_kmeans=use_soft_f1_kmeans,
                                            soft_f1_max_iter=soft_f1_max_iter,
                                            soft_f1_lr=soft_f1_lr,
                                            soft_f1_temp=soft_f1_temp,
                                            clustered_soft_f1_only=clustered_soft_f1_only,
                                            )
    # input()
    norm_res = analyzer.norm_analysis()

    svm_res = analyzer.svm_analysis(IN_TEST=True, in_test_ratio=test_ratio)

    length_res = f'True set length: {len(true_set)}, False set length: {len(false_set)}'
    with open(os.path.join(save_dir, 'gradient_svm_t.json'), 'w') as f:
        all_res = {"lenth": length_res, "norm": norm_res, "clustering": cluster_res, "svm": svm_res}
        all_res = _to_jsonable(all_res)
        json.dump(all_res, f, indent=4)
    _maybe_save_soft_f1_metrics(save_dir, "gradient_soft_f1_t.json", cluster_res)





def big_math_gradient(analyzer, model_name: str, save_dir: str, ct=False, all_rh=False,
                      get_gradient=True, test_ratio=0.7, 
                      use_pca=False, use_svd=False, use_t_sne=False, normalized=True,
                      use_soft_f1_kmeans=False, soft_f1_max_iter=200, soft_f1_lr=1e-2, soft_f1_temp=1.0,
                      clustered_soft_f1_only=False):
    # analyzer = GradientAnalyzer()
        
    print(f'Processing model: {model_name}')

    if all_rh:

        with open(os.path.join(save_dir, 'true_rh_all_rh.json'), 'r') as f:
            true_rh = json.load(f)
        with open(os.path.join(save_dir, 'false_rh_all_rh.json'), 'r') as f:
            false_rh = json.load(f)

    elif ct:
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

    cluster_res = analyzer.cluster_analysis(use_pca=use_pca, use_svd=use_svd, use_t_sne=use_t_sne, do_plot=use_t_sne, normalized=normalized,
                                            use_soft_f1_kmeans=use_soft_f1_kmeans,
                                            soft_f1_max_iter=soft_f1_max_iter,
                                            soft_f1_lr=soft_f1_lr,
                                            soft_f1_temp=soft_f1_temp,
                                            clustered_soft_f1_only=clustered_soft_f1_only)
    # input()
    norm_res = analyzer.norm_analysis()

    svm_res = analyzer.svm_analysis(IN_TEST=True, in_test_ratio=test_ratio)

    length_res = f'True set length: {len(true_set)}, False set length: {len(false_set)}'
    all_res = {"lenth": length_res, "norm": norm_res, "clustering": cluster_res, "svm": svm_res}
    all_res = _to_jsonable(all_res)
    
    if ct:
        with open(os.path.join(save_dir, 'gradient_svm_ct.json'), 'w') as f:
            json.dump(all_res, f, indent=4)
        _maybe_save_soft_f1_metrics(save_dir, "gradient_soft_f1_ct.json", cluster_res)
    elif all_rh:
        with open(os.path.join(save_dir, 'gradient_svm_t_all_rh.json'), 'w') as f:
            json.dump(all_res, f, indent=4)
        _maybe_save_soft_f1_metrics(save_dir, "gradient_soft_f1_t_all_rh.json", cluster_res)
    else:
        with open(os.path.join(save_dir, 'gradient_svm_t.json'), 'w') as f:
            json.dump(all_res, f, indent=4)
        _maybe_save_soft_f1_metrics(save_dir, "gradient_soft_f1_t.json", cluster_res)





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
