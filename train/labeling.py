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
from big_math.trace.rh_model_setting import (
    inference_on_ds,
    get_trace_on_ds,
    trace_eval,
    create_trace_runtime,
    close_trace_runtime,
)
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

def _write_json_artifact(path, payload):
    if not path:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_gradient_cache_payload(path):
    with open(path, "rb") as f:
        raw_payload = torch.load(f)

    if isinstance(raw_payload, dict):
        sketches = raw_payload.get("sketches", raw_payload)
        response_token_counts = raw_payload.get("response_token_counts")
    else:
        sketches = raw_payload
        response_token_counts = None

    if isinstance(sketches, torch.Tensor):
        sketches = sketches.to(dtype=torch.float32, device="cpu")
    else:
        sketches = torch.as_tensor(np.asarray(sketches), dtype=torch.float32, device="cpu")

    if response_token_counts is None:
        counts = None
    elif isinstance(response_token_counts, torch.Tensor):
        counts = response_token_counts.to(dtype=torch.float32, device="cpu")
    else:
        counts = torch.as_tensor(
            np.asarray(response_token_counts), dtype=torch.float32, device="cpu"
        )

    return {
        "sketches": sketches,
        "response_token_counts": counts,
    }


def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.size < 2 or y.size < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def _max_abs_feature_correlation(X, response_token_counts):
    if response_token_counts is None:
        return None
    values = []
    for col in range(X.shape[1]):
        corr = _safe_corrcoef(X[:, col], response_token_counts)
        if corr is not None:
            values.append(abs(corr))
    if not values:
        return None
    return float(max(values))


def _apply_cluster_preprocess(X, cluster_preprocess="none", response_token_counts=None):
    X = np.asarray(X, dtype=np.float32)
    diagnostics = {
        "cluster_preprocess": cluster_preprocess,
        "response_token_count_available": response_token_counts is not None,
        "length_signal_removed": False,
        "max_abs_feature_corr_before": None,
        "max_abs_feature_corr_after": None,
    }
    if cluster_preprocess != "residualize_response_length" or response_token_counts is None:
        return X, diagnostics

    counts = np.asarray(response_token_counts, dtype=np.float32).reshape(-1)
    if counts.shape[0] != X.shape[0]:
        return X, diagnostics

    log_counts = np.log1p(counts)
    denom = float(log_counts.std())
    if denom <= 1e-8:
        return X, diagnostics

    z = (log_counts - float(log_counts.mean())) / denom
    Xc = X - X.mean(axis=0, keepdims=True)
    z_norm = float(np.dot(z, z))
    if z_norm <= 1e-8:
        return X, diagnostics

    diagnostics["max_abs_feature_corr_before"] = _max_abs_feature_correlation(Xc, z)
    beta = (z[:, None] * Xc).sum(axis=0) / z_norm
    residualized = Xc - np.outer(z, beta)
    diagnostics["max_abs_feature_corr_after"] = _max_abs_feature_correlation(
        residualized, z
    )

    before = diagnostics["max_abs_feature_corr_before"]
    after = diagnostics["max_abs_feature_corr_after"]
    if before is not None and after is not None and after < before:
        diagnostics["length_signal_removed"] = True
    return residualized, diagnostics


def _compute_cluster_length_diagnostics(cluster_labels, response_token_counts):
    if response_token_counts is None:
        return {"available": False}

    labels = np.asarray(cluster_labels, dtype=np.float32).reshape(-1)
    counts = np.asarray(response_token_counts, dtype=np.float32).reshape(-1)
    if labels.shape[0] != counts.shape[0]:
        return {"available": False}

    diagnostics = {
        "available": True,
        "cluster_length_corr": _safe_corrcoef(labels, counts),
        "cluster_0_mean_response_tokens": None,
        "cluster_1_mean_response_tokens": None,
    }
    for cluster_id in [0, 1]:
        cluster_counts = counts[labels == float(cluster_id)]
        if cluster_counts.size > 0:
            diagnostics[f"cluster_{cluster_id}_mean_response_tokens"] = float(
                cluster_counts.mean()
            )
    return diagnostics


def _resolve_topk_gpt_cluster(
    *,
    cluster_labels,
    ds,
    cluster_semantics,
    cluster_scope,
    soft_cluster_probs,
    gpt_eval_log_path,
    prompt_template,
    default_pos,
    aggregate="sum",
    tie_fallback="majority_tie",
):
    eval_log_payload = {
        "cluster_scope": cluster_scope,
        "cluster_semantics": cluster_semantics,
        "evaluations": [],
        "cluster_summary": {},
        "default_pos": int(default_pos),
        "fallback": None,
        "selection_metric": "score_avg" if aggregate == "avg" else "score_sum",
    }
    if gpt_completion is None:
        eval_log_payload["fallback"] = "gpt_unavailable"
        _write_json_artifact(gpt_eval_log_path, eval_log_payload)
        return default_pos
    if soft_cluster_probs is None:
        eval_log_payload["fallback"] = "missing_soft_cluster_probs"
        _write_json_artifact(gpt_eval_log_path, eval_log_payload)
        return default_pos

    try:
        metric_by_cluster = {}
        for c in [0, 1]:
            idxs = np.where(cluster_labels == c)[0]
            if len(idxs) == 0:
                eval_log_payload["fallback"] = f"empty_cluster_{c}"
                return default_pos

            ranked_idxs = sorted(
                idxs,
                key=lambda idx: float(soft_cluster_probs[idx, c]),
                reverse=True,
            )
            top_k = ranked_idxs[: min(5, len(ranked_idxs))]
            scores = []
            for rank, idx in enumerate(top_k, start=1):
                q = ds[idx].get("input", "")
                r = ds[idx].get("output", "")
                eval_prompt = prompt_template.format(prompt=q, response=r)
                raw = gpt_completion(eval_prompt)
                parsed_score = _extract_score(raw)
                score = None if parsed_score is None else int(parsed_score)
                eval_log_payload["evaluations"].append(
                    {
                        "cluster": int(c),
                        "rank_in_cluster": int(rank),
                        "sample_idx": int(idx),
                        "cluster_prob": float(soft_cluster_probs[idx, c]),
                        "input": q,
                        "output": r,
                        "gpt_completion": raw,
                        "score": score,
                    }
                )
                if score is None:
                    eval_log_payload["fallback"] = "parse_failure"
                    return default_pos
                scores.append(score)

            score_sum = sum(scores)
            score_avg = float(score_sum) / float(len(scores)) if scores else 0.0
            metric_value = score_avg if aggregate == "avg" else float(score_sum)
            metric_by_cluster[c] = float(metric_value)
            eval_log_payload["cluster_summary"][str(c)] = {
                "scores": [int(s) for s in scores],
                "score_sum": int(score_sum),
                "score_avg": float(score_avg),
                "selection_metric_value": float(metric_value),
                "top_k_count": int(len(top_k)),
                "cluster_size": int(len(idxs)),
            }

        if np.isclose(metric_by_cluster[0], metric_by_cluster[1]):
            eval_log_payload["fallback"] = tie_fallback
            return default_pos

        return 0 if metric_by_cluster[0] > metric_by_cluster[1] else 1
    except Exception as exc:
        eval_log_payload["fallback"] = f"exception:{type(exc).__name__}"
        eval_log_payload["exception"] = str(exc)
        return default_pos
    finally:
        _write_json_artifact(gpt_eval_log_path, eval_log_payload)

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
    cluster_scope="local",
    soft_cluster_probs=None,
    gpt_eval_log_path=None,
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

    if cluster_semantics in {"rft_arlsat_gpt_top5", "gpt_top5_avg"}:
        return _resolve_topk_gpt_cluster(
            cluster_labels=cluster_labels,
            ds=ds,
            cluster_semantics=cluster_semantics,
            cluster_scope=cluster_scope,
            soft_cluster_probs=soft_cluster_probs,
            gpt_eval_log_path=gpt_eval_log_path,
            prompt_template=gpt_eval_prompt or GPT_EVAL_PROMPT,
            default_pos=default_pos,
            aggregate="avg" if cluster_semantics == "gpt_top5_avg" else "sum",
            tie_fallback="average_tie" if cluster_semantics == "gpt_top5_avg" else "majority_tie",
        )

    if cluster_semantics == "gpt":
        if gpt_completion is None:
            return default_pos
        prompt_template = gpt_eval_prompt or GPT_EVAL_PROMPT

        if cluster_scope == "global":
            if soft_cluster_probs is None:
                raise ValueError("Global GPT cluster semantics requires soft cluster probabilities.")

            print("\n\nRunning cluster labeling with gpt evaluation")
            eval_log_payload = {
                "cluster_scope": "global",
                "cluster_semantics": "gpt",
                "evaluations": [],
                "cluster_summary": {},
                "vote_by_cluster": {},
            }
            vote_by_cluster = {}
            try:
                for c in [0, 1]:
                    idxs = np.where(cluster_labels == c)[0]
                    if len(idxs) == 0:
                        raise ValueError(f"Global GPT cluster semantics failed: empty cluster {c}.")

                    ranked_idxs = sorted(idxs, key=lambda idx: float(soft_cluster_probs[idx, c]), reverse=True)
                    top_k = ranked_idxs[: min(5, len(ranked_idxs))]

                    votes = []
                    for rank, idx in enumerate(top_k, start=1):
                        q = ds[idx].get("input", "")
                        r = ds[idx].get("output", "")
                        eval_prompt = prompt_template.format(prompt=q, response=r)
                        raw = gpt_completion(eval_prompt)
                        parsed_score = _extract_score(raw)
                        score = 0 if parsed_score is None else int(parsed_score)
                        votes.append(score)
                        eval_log_payload["evaluations"].append(
                            {
                                "cluster": int(c),
                                "rank_in_cluster": int(rank),
                                "sample_idx": int(idx),
                                "cluster_prob": float(soft_cluster_probs[idx, c]),
                                "input": q,
                                "output": r,
                                "gpt_completion": raw,
                                "score": int(score),
                            }
                        )

                    pos_votes = sum(votes)
                    neg_votes = len(votes) - pos_votes
                    eval_log_payload["cluster_summary"][str(c)] = {
                        "votes": [int(v) for v in votes],
                        "pos_votes": int(pos_votes),
                        "neg_votes": int(neg_votes),
                        "cluster_size": int(len(idxs)),
                    }
                    if pos_votes == neg_votes:
                        raise ValueError(
                            f"Global GPT cluster semantics tie on cluster {c}: votes={votes}. "
                            "Cannot determine true/false label."
                        )
                    vote_by_cluster[c] = 1 if pos_votes > neg_votes else 0

                eval_log_payload["vote_by_cluster"] = {str(k): int(v) for k, v in vote_by_cluster.items()}
                if vote_by_cluster[0] == vote_by_cluster[1]:
                    raise ValueError(
                        "Global GPT cluster semantics produced identical labels for both clusters: "
                        f"{vote_by_cluster}. Stop labeling."
                    )
                return 0 if vote_by_cluster[0] == 1 else 1
            finally:
                _write_json_artifact(gpt_eval_log_path, eval_log_payload)

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
                      cluster_scope="local",
                      cluster_preprocess="none",
                      get_gradient=True, load_pi=True,
                      test_ratio=0.7, 
                      use_pca=False, use_svd=False, use_t_sne=False, normalized=True,
                      return_detail=False, soft_prob_temp=1.0, soft_prob_n_runs=8,
                      soft_prob_single_seed=False,
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

    gradient_path = os.path.join(save_dir, "ds_gradient")
    gradient_exists = os.path.exists(gradient_path)

    print(f'gradient path: {gradient_path}')

    response_token_counts = None

    if get_gradient and not gradient_exists:

        # select layers based on combined set
        selected_layers = analyzer.gradient_layer_selection(ds=ds, model_name=model_name)
        print(f'Selected layers: {selected_layers}')

        # get gradients from true and false set
        ds_g = analyzer.get_gradient(
            ds=ds,
            model_name=model_name,
            save_path=gradient_path,
            selected_layers=selected_layers,
            Pi=None,
            pi_cache_dir="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train",
            pi_cache_prefix="hid" if load_pi else "q2",
        )
        grad_payload = getattr(analyzer, "last_gradient_payload", None)
        if isinstance(grad_payload, dict):
            ds_g = grad_payload.get("sketches", ds_g)
            response_token_counts = grad_payload.get("response_token_counts")
    else:
        if not gradient_exists:
            raise FileNotFoundError(
                f"Gradient cache not found at {gradient_path}. "
                "Either set get_gradient=True to compute it or provide existing ds_gradient."
            )
        print(f"\n\nUsing cached gradient from {gradient_path}\n\n")
        grad_payload = _load_gradient_cache_payload(gradient_path)
        ds_g = grad_payload["sketches"]
        response_token_counts = grad_payload["response_token_counts"]
        
    # load gradient

    if method == "cluster":
        # cluster results
        if cluster_method == "soft_prob":
            print("\n\nRunning clustering K-means with soft prob mode...\n\n")
            X = ds_g.numpy() if isinstance(ds_g, torch.Tensor) else np.asarray(ds_g)
            X_prepared, preprocess_diagnostics = _apply_cluster_preprocess(
                X,
                cluster_preprocess=cluster_preprocess,
                response_token_counts=response_token_counts.numpy()
                if isinstance(response_token_counts, torch.Tensor)
                else response_token_counts,
            )
            Xn = normalize(X_prepared, norm="l2") if normalized else X_prepared
            best = None
            best_score = -1e18
            if bool(soft_prob_single_seed):
                seed_schedule = [SEED]
            else:
                n_runs = max(1, int(soft_prob_n_runs))
                seed_schedule = list(range(SEED, SEED + n_runs))
            n_runs_effective = len(seed_schedule)
            for seed in seed_schedule:
                km_i = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xn)
                labels_i = km_i.labels_
                if len(np.unique(labels_i)) < 2:
                    score_i = -1e18
                else:
                    try:
                        score_i = float(silhouette_score(Xn, labels_i))
                    except Exception:
                        score_i = -1e18
                if best is None or score_i > best_score:
                    best_score = score_i
                    best = (km_i, labels_i, seed)
            if best is not None:
                km, cluster_labels, picked_seed = best
            else:
                print("\n\nNo best clustering run...\n\n")
                return None
                km, cluster_labels, picked_seed = km_i, labels_i, SEED + n_runs_effective - 1
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
                cluster_scope=cluster_scope,
                soft_cluster_probs=probs,
                gpt_eval_log_path=(
                    os.path.join(save_dir, "global_gpt_top5_eval.json")
                    if cluster_scope == "global" and cluster_semantics == "gpt"
                    else os.path.join(save_dir, "rft_arlsat_gpt_top5_eval.json")
                    if cluster_semantics == "rft_arlsat_gpt_top5"
                    else os.path.join(save_dir, "gpt_top5_avg_eval.json")
                    if cluster_semantics == "gpt_top5_avg"
                    else None
                ),
            )
            true_probs = probs[:, pos_cluster]
            res = np.where(cluster_labels == pos_cluster, 1, 0).astype(int).tolist()
            length_diagnostics = _compute_cluster_length_diagnostics(
                cluster_labels,
                response_token_counts.numpy()
                if isinstance(response_token_counts, torch.Tensor)
                else response_token_counts,
            )
            detail = {
                "result": res,
                "true_probs": true_probs.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "pos_cluster": int(pos_cluster),
                "cluster_semantics": cluster_semantics,
                "cluster_scope": cluster_scope,
                "cluster_preprocess": cluster_preprocess,
                "cluster_preprocess_diagnostics": preprocess_diagnostics,
                "length_diagnostics": length_diagnostics,
                "robust_silhouette": float(best_score),
                "robust_seed": int(picked_seed),
                "soft_prob_n_runs": int(n_runs_effective),
                "soft_prob_single_seed": bool(soft_prob_single_seed),
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
            km_centers = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/baseline_km"

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



def trace_analysis(ds, model_name, save_dir, threshold=0.2904761904761905, saving=False,
                   trace_llm=None, trace_sampler=None, return_trace_score=False):
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
                        model_name=model_name, set_name='true_gpt', 
                        max_token=2048, n_gpu=4, K=3,
                        saving_inter=saving,
                        llm=trace_llm, sampler=trace_sampler)
    
    with open(os.path.join(save_dir, 'trace.json'), 'r') as f:
        trace_score = json.load(f)['all_trace']
    
    # get baseline to be the threshold 
    # q3 arlsat
    # threshold = 0.2904761904761905

    res = [1 if t <= threshold else 0 for t in trace_score]

    if return_trace_score:
        return res, trace_score
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
