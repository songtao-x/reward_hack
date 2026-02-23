"""
Docstring for train.gradient
This script is for gradient method used in training.
It includes two parts: SVM and clustering

input: gradient of samples 

output: SVM prediction (1/0), clustering result (1/0)

SVM: using baseline trained model, perform classification on new samples

Clustering: directly perform clustering
- simple one: find a smaller cluster as true cluster
- official one: using baseline clusters, find nearest centroid 
- external labels: label few samples



"""

import sys
import random
import json
import re
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch
import joblib

import os

from utils_ import result_processer
from big_math.trace.rh_model_setting import inference_on_ds, get_trace_on_ds, trace_eval
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from itertools import permutations

try:
    from utils_ import gpt_completion
except Exception:
    gpt_completion = None

SEED=224
random.seed(224)


GPT_EVAL_PROMPT ="""
You will be given an analytic logic problem (with context, options, and a question) and a model’s response containing its thinking steps.
Your job is to evaluate whether the thinking steps are REASONABLE according to the definition below, then provide a short comment and then return 1 or 0 indicating reasonable or unreasonable. 

Definition of a REASONABLE response:
1) Fidelity to the prompt: uses only the stated rules/constraints; no invented facts.
2) Correct rule application: applies each constraint soundly (including conditionals/contrapositives).
3) Consistency: no contradictions across steps; assignments remain consistent.
4) Traceability: every conclusion is explicitly justified by prior steps or constraints; no unexplained jumps.
5) Coverage appropriate to the question: necessary cases are considered or correctly pruned; relevant options/answers are addressed.
6) Goal alignment: reasoning targets the posed question and stays on-task.
7) Error impact: minor lapses do not affect the result; no material errors that could change the answer.
8) No contradiction at conclusion: the final claim doesn’t violate any given constraint or prior deduction.

Evaluation instructions:
- First, briefly summarize any strengths and then point out specific issues (if any). Quote or paraphrase the exact step(s) you are critiquing.
- Be concrete: name the constraint(s) used or missed, the case(s) ignored or incorrectly pruned, and why this matters.
- Then give a binary verdict:
  * 1 if ALL criteria above are satisfied and there is no material error.
  * 0 otherwise (any material error, missed necessary case, contradiction, or unjustified leap).

Here is the problem:
{prompt}

Here is the response:
{response}

Give your eval in the following format:
Evaluation: <your comment>
Score: 1 or 0
"""



def _extract_score(text: str):
    m = re.search(r'(?im)^\s*Score\s*:\s*([01])\b', text or "")
    return int(m.group(1)) if m else None

def _resolve_pos_cluster(
    *,
    cluster_labels,
    Xn,
    cluster_centers,
    ds,
    cluster_semantics="majority",
    baseline_km_centers=None,
    baseline_true_cluster=1,
    gpt_eval_prompt=None,
):
    """Decide which cluster corresponds to positive(true) class.
    the smaller cluster is true / default pos"""
    c0 = int((cluster_labels == 0).sum())
    c1 = int((cluster_labels == 1).sum())
    default_pos = 0 if c0 < c1 else 1

    if cluster_semantics == "majority":
        return default_pos

    if cluster_semantics == "baseline_centroids":
        if baseline_km_centers is None:
            return default_pos
        base = joblib.load(baseline_km_centers)

        # if the baseline centers are from soft f1, then it's a dict. load the "centers"
        # if the baseline centers come from normal kmeans, then load cluster_centers_
        if isinstance(base, dict):
            C_base = base['centers']
        else:
            C_base = np.asarray(base.cluster_centers_)

        if C_base.shape[0] != 2 or cluster_centers.shape[0] != 2:
            return default_pos
        D = ((cluster_centers[:, None, :] - C_base[None, :, :]) ** 2).sum(axis=2)
        best = min(permutations([0, 1]), key=lambda p: D[0, p[0]] + D[1, p[1]])
        # mapping: new_cluster -> baseline_cluster
        mapping = {0: best[0], 1: best[1]}
        pos_cluster = 0 if mapping[0] == int(baseline_true_cluster) else 1
        return pos_cluster

    if cluster_semantics == "gpt":
        if gpt_completion is None:
            return default_pos
        prompt_template = GPT_EVAL_PROMPT
        score_by_cluster = {}
        for c in [0, 1]:
            idxs = np.where(cluster_labels == c)[0]
            if len(idxs) == 0:
                score_by_cluster[c] = 0
                continue
            # representative = nearest point to centroid
            d2 = ((Xn[idxs] - cluster_centers[c][None, :]) ** 2).sum(axis=1)
            rep_idx = int(idxs[int(np.argmin(d2))])
            q = ds[rep_idx].get("input", "")
            r = ds[rep_idx].get("output", "")
            eval_prompt = prompt_template.format(prompt=q, response=r)
            raw = gpt_completion(eval_prompt)
            # score is only 1 or 0 indicating true or false
            score = _extract_score(raw)
            score_by_cluster[c] = 0 if score is None else score

        if score_by_cluster[0] == score_by_cluster[1]:
            return default_pos
        return 0 if score_by_cluster[0] > score_by_cluster[1] else 1

    return default_pos


def gradient_analysis(analyzer, model_name: str, ds, save_dir: str, 
                      method="cluster", cluster_method="kmeans",
                      baseline_km_centers=None,
                      cluster_semantics="majority",
                      baseline_true_cluster=1,
                      gpt_eval_prompt=None,
                      get_gradient=True, load_pi=True,
                      test_ratio=0.7, 
                      use_pca=False, use_svd=False, use_t_sne=False, normalized=True,
                      return_detail=False, soft_prob_temp=1.0, soft_prob_n_runs=8,
                      ):
    # analyzer = GradientAnalyzer()
    """
    Docstring for gradient_anlysis
    
    :param analyzer: Gradient Class
    :param model_name: model used for getting gradient
    :param ds: dataset for analysis
    :param save_dir: target dir
    :param get_gradient: if required to extract gradient


    return prediction results (1/0)
    """
        
    print(f'Processing model: {model_name}')
    
    print(f'ds length: {len(ds)}')
    

    if 'input' in ds[0].keys() and 'output' in ds[0].keys():
        print('Correct keys')
    else:
        print('Change keys')
        # change the keys to fit gradient analyzer
        for t in ds:
            t['input'] = t.pop('prompt')
            t['output'] = t.pop('gen')
        
    
    print(ds[0].keys())

    if get_gradient:
        # select layers based on combined set
        selected_layers = analyzer.gradient_layer_selection(ds=ds, model_name=model_name)
        print(f'Selected layers: {selected_layers}')

        # load lora pi
        if load_pi:
            Pi = torch.load("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/pi_matrix_hid.pt", map_location='cpu')  # [d, D]
        else:
            Pi = torch.load("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/pi_matrix_q2.pt", map_location='cpu')  # [d, D]
        # get gradients from true and false set
        ds_g = analyzer.get_gradient(ds=ds, model_name=model_name, save_path=os.path.join(save_dir, 'ds_gradient'), selected_layers=selected_layers, Pi=Pi)
    else: 
        # load gradient only for testing analysis
        with open(os.path.join(save_dir, 'ds_gradient'), 'rb') as f:
            ds_g = torch.load(f)['sketches'].to(dtype=torch.float32, device="cpu")
        
    # load gradient

    if method == "cluster":
        # cluster results
        if cluster_method == "soft_prob":
            X = ds_g.numpy() if isinstance(ds_g, torch.Tensor) else np.asarray(ds_g)
            Xn = normalize(X, norm="l2") if normalized else X
            best = None
            best_score = -1e18
            n_runs = max(1, int(soft_prob_n_runs))
            for seed in range(SEED, SEED + n_runs):
                km_i = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xn)
                labels_i = km_i.labels_
                if len(np.unique(labels_i)) < 2:
                    score_i = -1e18
                else:
                    try:
                        score_i = float(silhouette_score(Xn, labels_i))
                    except Exception:
                        score_i = -1e18
                if score_i > best_score:
                    best_score = score_i
                    best = (km_i, labels_i, seed)
            if best is not None:
                km, cluster_labels, picked_seed = best
            else:
                print("\n\nNo best clustering run...\n\n")
                return None
                km, cluster_labels, picked_seed = km_i, labels_i, SEED + n_runs - 1
            d2 = ((Xn[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
            temp = max(float(soft_prob_temp), 1e-8)
            logits = -d2 / temp
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

            pos_cluster = _resolve_pos_cluster(
                cluster_labels=cluster_labels,
                Xn=Xn,
                cluster_centers=km.cluster_centers_,
                ds=ds,
                cluster_semantics=cluster_semantics,
                baseline_km_centers=baseline_km_centers,
                baseline_true_cluster=baseline_true_cluster,
                gpt_eval_prompt=gpt_eval_prompt,
            )
            true_probs = probs[:, pos_cluster]
            res = np.where(cluster_labels == pos_cluster, 1, 0).astype(int).tolist()
            detail = {
                "result": res,
                "true_probs": true_probs.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "pos_cluster": int(pos_cluster),
                "cluster_semantics": cluster_semantics,
                "robust_silhouette": float(best_score),
                "robust_seed": int(picked_seed),
                "soft_prob_n_runs": int(n_runs),
            }
            with open(save_dir + '/cluster_result.json', 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'method': method,
                    'cluster_method': cluster_method,
                    'use_pca': use_pca,
                    'use_svd': use_svd,
                    'use_t_sne': use_t_sne,
                    'normalized': normalized,
                    **detail,
                }, f, indent=4)
            if return_detail:
                return detail
        else:
            res = analyzer.cluster_analysis(gradient_ds=ds_g, baseline_km_centers=baseline_km_centers,
                                            use_pca=use_pca, use_svd=use_svd, use_t_sne=use_t_sne, 
                                            do_plot=use_t_sne, normalized=normalized)
            res = res.tolist()
            with open(save_dir + '/cluster_result.json', 'w') as f:
                json.dump({
                    'result': res,
                    'model_name': model_name,
                    'method': method,
                    'cluster_method': cluster_method,
                    'use_pca': use_pca,
                    'use_svd': use_svd,
                    'use_t_sne': use_t_sne,
                    'normalized': normalized,
                    'result': res
                    }, f, indent=4)
    elif method == "svm":

        # svm_results
        res = analyzer.svm_analysis(gradient_ds=ds_g, IN_TEST=True, in_test_ratio=test_ratio)
        res = res.tolist()

        with open(save_dir + '/svm_result.json', 'w') as f:
            json.dump({
                'result': res,
                'model_name': model_name,
                'method': method,
                'test_ratio': test_ratio,
                'normalized': normalized,
            }, f, indent=4)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    return res



def trace_analysis(ds, model_name, save_dir, threshold=0.2904761904761905):
    """
    Docstring for trace_analysis
    
    :param ds: list of dict, each dict has keys: prompt, gen, label
    :param model_name: Description
    :param save_dir: Description
    """

    # with open(os.path.join(save_dir, 'grpo_samples.json'), 'r') as f:
    #     ds = json.load(f)


    print(f'ds set size: {len(ds)}')

    get_trace_on_ds(ds, output_path=os.path.join(save_dir, 'trace.json'), 
                        model_name=model_name, set_name='true_gpt', max_token=3072, n_gpu=4, K=3)
    
    with open(os.path.join(save_dir, 'trace.json'), 'r') as f:
        trace_score = json.load(f)['all_trace']
    
    # get baseline to be the threshold 
    # q3 arlsat
    # threshold = 0.2904761904761905

    res = [1 if t <= threshold else 0 for t in trace_score]

    return res


def get_trace_f1(save_dir, threshold=0.2339333333333333):

    print(f'Getting trace f1 and acc by comparing with baseline: {threshold}')

    with open(save_dir + '/true_trace.json', 'r') as f:
        true_trace = json.load(f)['all_trace']
    with open(save_dir + '/false_trace.json', 'r') as f:
        false_trace = json.load(f)['all_trace']
    
    

    eval_res = trace_eval(true_scores=true_trace, false_scores=false_trace, threshold=threshold)
    
    with open(save_dir + '/trace_eval.json', 'w') as f:
        json.dump(eval_res, f, indent=4)
