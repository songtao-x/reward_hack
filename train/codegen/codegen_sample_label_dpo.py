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


def _load_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: failed to load json {path}: {e}")
        return None


def _dump_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _sampling_partial_paths(save_dir: str):
    return (
        os.path.join(save_dir, "samples.partial.json"),
        os.path.join(save_dir, "all_samples.partial.json"),
    )


def _save_sampling_partial(save_dir: str, records, all_records):
    samples_partial_path, all_partial_path = _sampling_partial_paths(save_dir)
    _dump_json(samples_partial_path, records)
    _dump_json(all_partial_path, all_records)


def _load_sampling_partial(save_dir: str, ds_len: int):
    samples_partial_path, all_partial_path = _sampling_partial_paths(save_dir)
    records = _load_json_if_exists(samples_partial_path)
    all_records = _load_json_if_exists(all_partial_path)

    if not isinstance(records, list):
        records = []
    if not isinstance(all_records, list):
        all_records = []

    if len(all_records) > ds_len:
        print(
            f"WARNING: partial sampling length {len(all_records)} > dataset length {ds_len}; "
            "ignoring partial sampling files."
        )
        return [], [], 0

    if len(records) != len(all_records):
        m = min(len(records), len(all_records))
        print(
            f"WARNING: partial sampling mismatch (records={len(records)}, all_records={len(all_records)}); "
            f"truncating to {m}."
        )
        records = records[:m]
        all_records = all_records[:m]

    return records, all_records, len(all_records)


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
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    k: int = 8,
    k_max: int = 32,
    batch_size: int = 8,
    temp: float = 0,
    top_p: float = 1,
    max_tokens: int = 3072,
    tensor_parallel_size: int = 4,
    reward_timeout_s: int = 120,
    resume: bool = True,
    save_every_prompts: int = 8,
) -> List[Dict[str, Any]]:
    os.makedirs(save_dir, exist_ok=True)
    final_samples_path = os.path.join(save_dir, "samples.json")
    final_all_samples_path = os.path.join(save_dir, "all_samples.json")

    if resume and os.path.exists(final_samples_path) and os.path.exists(final_all_samples_path):
        loaded = _load_json_if_exists(final_samples_path)
        if isinstance(loaded, list):
            print(f"Found completed sampling outputs in {save_dir}; loading existing samples.")
            return loaded
    
    from vllm.lora.request import LoRARequest

    # llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    LORA_DIR = model_name # the folder in your screenshot

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,   # set >= your LoRA rank; safe to set a bit larger
    )

    # Attach LoRA for this request
    lora_req = LoRARequest("my_lora", 1, LORA_DIR)

    records: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    start_i = 0
    if resume:
        records, all_records, start_i = _load_sampling_partial(save_dir, len(ds))
        if start_i > 0:
            print(f"Resuming codegen sampling from prompt index {start_i} / {len(ds)}")

    total_first_correct = sum(int(r.get("first_correct", 0)) for r in all_records)
    total_any_correct = sum(int(int(r.get("n_correct", 0)) > 0) for r in all_records)

    for i in tqdm(range(start_i, len(ds)), desc="Codegen sampling"):
        ex = ds[i]
        prompt = ex["prompt"]
        used = 0
        correct: List[str] = []
        all_gen: List[str] = []
        first_correct = 0

        while used < k_max and len(correct) < k:
            bsz = min(batch_size, k_max - used)
            outputs = llm.generate([prompt] * bsz, sampling_params, lora_request=lora_req)
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
        if save_every_prompts > 0 and ((i + 1) % save_every_prompts == 0 or (i + 1) == len(ds)):
            _save_sampling_partial(save_dir, records, all_records)

    _dump_json(final_samples_path, records)
    _dump_json(
        final_all_samples_path,
        {
            "acc_at_1": total_first_correct / max(1, len(ds)),
            f"acc_any_at_{k_max}": total_any_correct / max(1, len(ds)),
            "n_total": len(ds),
            "records": all_records,
        },
    )

    samples_partial_path, all_partial_path = _sampling_partial_paths(save_dir)
    for p in [samples_partial_path, all_partial_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print(f"WARNING: failed to remove partial sampling file {p}: {e}")

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
    p.add_argument("--no_resume_sampling", action="store_true")
    p.add_argument("--sample_save_every_prompts", type=int, default=8)
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
            resume=(not args.no_resume_sampling),
            save_every_prompts=args.sample_save_every_prompts,
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
