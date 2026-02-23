import argparse
import json
import os
import re
from typing import Any, Dict, List

import requests
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams


def build_prompt(question: str) -> str:
    return (
        "You are a Python programming assistant.\n"
        "Solve the following programming problem and return only Python code "
        "inside a ```python ... ``` code block.\n\n"
        f"{question.strip()}\n"
    )


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


def load_labeled_data(paths: List[str], limit: int = -1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                rows.append(
                    {
                        "pid": f"{os.path.basename(path)}:{i}",
                        "question": ex["question"],
                        "prompt": build_prompt(ex["question"]),
                        "test_cases": ex["test_cases"],
                    }
                )
                if limit > 0 and len(rows) >= limit:
                    return rows
    return rows


def score_batch(reward_url: str, batch: List[Dict[str, Any]], timeout_s: int = 120):
    payload = []
    for ex in batch:
        payload.append(
            {
                "query": ex["question"],
                "response": ex["code"],
                "test_cases": ex["test_cases"],
            }
        )
    resp = requests.post(reward_url, data=json.dumps(payload), timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()["flags"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="Labeled programming jsonl files")
    p.add_argument("--model", required=True, help="HF model or local checkpoint for vLLM")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reward_url", default="http://localhost:8118/batched_unittest")

    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8, help="vLLM generation batch size")
    p.add_argument("--eval_batch_size", type=int, default=16, help="Reward API batch size")
    p.add_argument("--max_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--reward_timeout_s", type=int, default=120)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_labeled_data(args.data, limit=args.limit)
    if not data:
        raise RuntimeError("No examples loaded.")

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    generated_rows: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(data), args.batch_size), desc="Generate"):
        batch = data[start : start + args.batch_size]
        prompts = [ex["prompt"] for ex in batch]
        outputs = llm.generate(prompts, sampling_params)

        for ex, out in zip(batch, outputs):
            raw = out.outputs[0].text
            code = extract_python_code(raw)
            generated_rows.append(
                {
                    "pid": ex["pid"],
                    "question": ex["question"],
                    "prompt": ex["prompt"],
                    "test_cases": ex["test_cases"],
                    "raw_response": raw,
                    "code": code,
                }
            )

    # reward scoring
    total = 0
    correct = 0
    for start in tqdm(range(0, len(generated_rows), args.eval_batch_size), desc="Score"):
        batch = generated_rows[start : start + args.eval_batch_size]
        flags_batch = score_batch(args.reward_url, batch, timeout_s=args.reward_timeout_s)
        for ex, flags in zip(batch, flags_batch):
            ex["flags"] = [bool(x) for x in flags]
            ex["acc1"] = float(all(ex["flags"])) if ex["flags"] else 0.0
            total += 1
            correct += int(ex["acc1"])

    acc1 = correct / max(1, total)
    summary = {
        "acc@1": acc1,
        "n_correct": correct,
        "n_total": total,
        "model": args.model,
        "reward_url": args.reward_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for ex in generated_rows:
            ex2 = dict(ex)
            ex2.pop("test_cases", None)
            f.write(json.dumps(ex2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
