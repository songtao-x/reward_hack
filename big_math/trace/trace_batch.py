"""
Trace evaluation pipeline including calculating trace scores and comparing with baseline
"""

from .rh_model_setting import pipeline_trace
import json
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

random.seed(224)



def eval(true_scores, false_scores, threshold):
    # true set is non-hacking: label 1; false set is hacking: label 0
    print(f'True scores len: {len(true_scores)}, False scores len: {len(false_scores)}')
    y_true = np.concatenate([
        np.ones(len(true_scores), dtype=int),
        np.zeros(len(false_scores), dtype=int),
    ])
    scores = np.concatenate([true_scores, false_scores]).astype(float)

    # preds: score > threshold => hacking(1)
    y_pred = (scores < threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    # hacking label is 0, non hacking label is 1
    f1_hacking = f1_score(y_true, y_pred, pos_label=0) 
    f1_non_hacking = f1_score(y_true, y_pred, pos_label=1)
    return {'acc': acc, 'f1_hacking': f1_hacking, 'f1_non_hacking': f1_non_hacking}


def trace_eval():
    for s in range(10, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'
        with open(os.path.join(save_dir, 'true_trace.json'), 'r') as f:
            true_trace = json.load(f)
            true_trace = true_trace['all_trace']
        with open(os.path.join(save_dir, 'false_trace.json'), 'r') as f:
            false_trace = json.load(f)
            false_trace = false_trace['all_trace']
        
        # getting baseline trace threshold
        with open("/home/songtaow/projects/aip-qchen/songtaow/reward_hack/big_math/trace/data/Qwen/Qwen2.5-3B/normal_baseline_trace.json", 'r') as f:
            baseline_trace = json.load(f)
            baseline_trace = baseline_trace['trace_score']
        

        eval_res = eval(true_scores=true_trace, false_scores=false_trace, threshold=baseline_trace)
        print(f'Eval result: {eval_res}')
        with open(os.path.join(save_dir, 'trace_eval.json'), 'w') as f:
            json.dump(eval_res, f, indent=4)




def trace_eval_arlsat():
    for s in range(10, 50, 5):
        model_name = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_3_hint/global_step_{s}/actor/actor_hf"
        save_dir = f'trace/data/arlsat_grpo_cheat_step_{s}'
        print(f'step: {s}')
        with open(os.path.join(save_dir, 'true_trace.json'), 'r') as f:
            true_trace = json.load(f)
            true_trace = true_trace['all_trace']
        with open(os.path.join(save_dir, 'false_trace.json'), 'r') as f:
            false_trace = json.load(f)
            false_trace = false_trace['all_trace']
        
        # getting baseline trace threshold
        # normal baseline
        with open("/home/songtaow/projects/aip-qchen/songtaow/reward_hack/big_math/trace/data/Qwen/Qwen3-4B/normal_baseline_trace.json", 'r') as f:
            baseline_trace = json.load(f)
            baseline_trace = baseline_trace['trace_score']
        
        eval_res = eval(true_scores=true_trace, false_scores=false_trace, threshold=baseline_trace)
        print(f'Eval result: {eval_res}')
        with open(os.path.join(save_dir, 'trace_eval_normal_baseline.json'), 'w') as f:
            json.dump(eval_res, f, indent=4)
        

        # cheat baseline
        with open("/home/songtaow/projects/aip-qchen/songtaow/reward_hack/big_math/trace/data/Qwen/Qwen3-4B/cheat_baseline_trace.json", 'r') as f:
            baseline_trace = json.load(f)
            baseline_trace = baseline_trace['trace_score']
        
        eval_res = eval(true_scores=true_trace, false_scores=false_trace, threshold=baseline_trace)
        print(f'Eval result: {eval_res}')
        with open(os.path.join(save_dir, 'trace_eval_cheat_baseline.json'), 'w') as f:
            json.dump(eval_res, f, indent=4)




def main():
    for s in range(10, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_cheat_step_{s}'
        pipeline_trace(model_name=model_name, save_dir=save_dir, cheat=True)

    


if __name__ == '__main__':
    # main()
    # trace_eval()
    trace_eval_arlsat()


