import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import requests
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams


# Ensure `import train.*` works when running this file directly.
# File is located at reward_hack/train/codegen/codegen_sample_label_dpo.py
THIS_DIR = os.path.dirname(__file__)               # .../train/codegen
TRAIN_DIR = os.path.dirname(THIS_DIR)              # .../train
REPO_ROOT = os.path.dirname(TRAIN_DIR)             # .../reward_hack
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
ARLSAT_DIR = os.path.join(REPO_ROOT, "arlsat")
if ARLSAT_DIR not in sys.path:
    sys.path.insert(0, ARLSAT_DIR)

from train.grpo_train import gradient_labeling, trace_labeling  # noqa: E402


def build_prompt(question: str, style: str = "plain") -> str:
    if style == "plain":
        return (
            "You are a Python programming assistant.\n"
            "Solve the following programming problem and return only Python code "
            "inside a ```python ... ``` code block.\n\n"
            f"{question.strip()}\n"
        )
    raise ValueError(f"Unsupported prompt style: {style}")


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    m = re.findall(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip()
    m = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    if m:
        return m[0].strip()
    return text.strip()


def load_codegen_labeled_jsonl(paths: List[str], prompt_style: str = "plain", limit: int = -1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                q = ex["question"]
                rows.append(
                    {
                        "pid": f"{os.path.basename(path)}:{i}",
                        "question": q,
                        "prompt": build_prompt(q, style=prompt_style),
                        # Trace path in train.grpo_train expects a label field; codegen trace does not use it directly.
                        "label": "",
                        "test_cases": ex["test_cases"],
                    }
                )
                if limit > 0 and len(rows) >= limit:
                    return rows
    return rows


def _score_code_batch(reward_url: str, batch: List[Dict[str, Any]], timeout_s: int = 120) -> List[List[bool]]:
    payload = [
        {
            "query": ex["question"],
            "response": ex["code"],
            "test_cases": ex["test_cases"],
        }
        for ex in batch
    ]
    resp = requests.post(reward_url, data=json.dumps(payload), timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()["flags"]


def sample_correct_rollouts(
    model_name: str,
    ds: List[Dict[str, Any]],
    save_dir: str,
    reward_url: str,
    k: int = 8,
    k_max: int = 32,
    batch_size: int = 8,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 384,
    tensor_parallel_size: int = 1,
    reward_timeout_s: int = 120,
) -> List[Dict[str, Any]]:
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    records: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []

    total_first_correct = 0
    total_any_correct = 0

    for i, ex in enumerate(tqdm(ds, desc="Codegen sampling")):
        prompt = ex["prompt"]
        used = 0
        correct: List[str] = []
        all_gen: List[str] = []
        first_correct = 0

        while used < k_max and len(correct) < k:
            bsz = min(batch_size, k_max - used)
            outputs = llm.generate([prompt] * bsz, sampling_params)
            batch_candidates = []
            for out in outputs:
                raw = out.outputs[0].text
                code = extract_python_code(raw)
                batch_candidates.append({"question": ex["question"], "test_cases": ex["test_cases"], "code": code, "raw": raw})

            flags_batch = _score_code_batch(reward_url, batch_candidates, timeout_s=reward_timeout_s)

            for cand, flags in zip(batch_candidates, flags_batch):
                is_correct = bool(flags) and all(bool(x) for x in flags)
                used += 1
                all_gen.append(cand["raw"])
                if used == 1:
                    first_correct = int(is_correct)
                if is_correct:
                    correct.append(cand["raw"])
                    if len(correct) >= k:
                        break

        total_first_correct += first_correct
        total_any_correct += int(len(correct) > 0)

        records.append(
            {
                "pid": ex["pid"],
                "prompt": ex["prompt"],
                "label": ex["label"],
                "gen": correct,  # matches grpo_train.py expected format for DPO labeling
            }
        )
        all_records.append(
            {
                "pid": ex["pid"],
                "prompt": ex["prompt"],
                "label": ex["label"],
                "gen": all_gen,
                "question": ex["question"],
                "n_correct": len(correct),
                "used": used,
                "first_correct": first_correct,
            }
        )

        if (i + 1) % 16 == 0 or (i + 1) == len(ds):
            print(
                f"progress={i+1}/{len(ds)} "
                f"acc@1={total_first_correct/max(1,i+1):.4f} "
                f"acc@{k_max}(any-correct)={total_any_correct/max(1,i+1):.4f}"
            )

    with open(os.path.join(save_dir, "samples.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(save_dir, "all_samples.json"), "w") as f:
        json.dump(
            {
                "acc_at_1": total_first_correct / max(1, len(ds)),
                f"acc_any_at_{k_max}": total_any_correct / max(1, len(ds)),
                "n_total": len(ds),
                "records": all_records,
            },
            f,
            indent=2,
        )

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="Labeled programming jsonl files")
    p.add_argument("--model", required=True, help="Model checkpoint to sample from")
    p.add_argument("--output_root", required=True, help="Root output dir; script writes <root>/<labeling>/s<step>/...")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--labeling_method", choices=["gradient", "trace"], default="gradient")
    p.add_argument("--train_type", choices=["dpo"], default="dpo")
    p.add_argument("--prompt_style", default="plain")

    p.add_argument("--reward_url", default="http://localhost:8118/batched_unittest")
    p.add_argument("--reward_timeout_s", type=int, default=120)
    p.add_argument("--limit", type=int, default=-1)

    p.add_argument("--k", type=int, default=8, help="Required correct samples per prompt for labeling pool")
    p.add_argument("--k_max", type=int, default=32, help="Max sampling attempts per prompt")
    p.add_argument("--sample_batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=384)
    p.add_argument("--tensor_parallel_size", type=int, default=1)

    # Pass-through knobs for labeling behavior (matching grpo_train.py)
    p.add_argument("--cluster_method", default="kmeans", choices=["kmeans", "soft_prob"])
    p.add_argument("--cluster_semantics", default="majority", choices=["majority", "gpt", "baseline_centroids"])
    p.add_argument("--baseline_km_centers", default=None)
    p.add_argument("--baseline_true_cluster", type=int, default=1)
    p.add_argument("--cluster_soft_prob_temp", type=float, default=1.0)
    p.add_argument("--cluster_soft_prob_n_runs", type=int, default=8)
    p.add_argument("--cluster_gpt_eval_prompt", default=None)
    p.add_argument("--trace_threshold", type=float, default=0.0749, help="You likely need to calibrate this for codegen.")

    p.add_argument("--skip_sampling", action="store_true")
    p.add_argument("--force_resample", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    step_dir = os.path.join(args.output_root, args.labeling_method, f"s{args.step}")
    os.makedirs(step_dir, exist_ok=True)

    samples_path = os.path.join(step_dir, "grpo_samples", "samples.json")
    if (not args.skip_sampling) and (args.force_resample or not os.path.exists(samples_path)):
        ds = load_codegen_labeled_jsonl(args.data, prompt_style=args.prompt_style, limit=args.limit)
        if not ds:
            raise RuntimeError("No codegen examples loaded.")
        _ = sample_correct_rollouts(
            model_name=args.model,
            ds=ds,
            save_dir=os.path.join(step_dir, "grpo_samples"),
            reward_url=args.reward_url,
            k=args.k,
            k_max=args.k_max,
            batch_size=args.sample_batch_size,
            temp=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            reward_timeout_s=args.reward_timeout_s,
        )
    elif not os.path.exists(samples_path):
        raise FileNotFoundError(f"{samples_path} not found. Remove --skip_sampling or provide existing samples.")

    with open(samples_path, "r") as f:
        samples = json.load(f)

    if args.labeling_method == "gradient":
        _ = gradient_labeling(
            output_dir=step_dir,
            model_name=args.model,
            ds=samples,
            train_type="dpo",
            method="cluster",
            cluster_method=args.cluster_method,
            cluster_semantics=args.cluster_semantics,
            baseline_km_centers=args.baseline_km_centers,
            baseline_true_cluster=args.baseline_true_cluster,
            cluster_soft_prob_temp=args.cluster_soft_prob_temp,
            cluster_soft_prob_n_runs=args.cluster_soft_prob_n_runs,
            gpt_eval_prompt=args.cluster_gpt_eval_prompt,
        )
        print(f"Wrote DPO pairs to {os.path.join(step_dir, 'gradient', 'dpo_trainset.json')}")
    else:
        _ = trace_labeling(
            current_dir=step_dir,
            model_name=args.model,
            ds=samples,
            train_type="dpo",
            threshold=args.trace_threshold,
        )
        print(f"Wrote DPO pairs to {os.path.join(step_dir, 'trace', 'dpo_trainset.json')}")


if __name__ == "__main__":
    main()
