import argparse
import json
import os
import re
from typing import Any, Dict, List

import requests
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def load_programming_jsonl(paths: List[str], prompt_style: str = "plain") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                question = ex["question"].strip()
                prompt = build_codegen_prompt(question, style=prompt_style)
                rows.append(
                    {
                        "pid": f"{os.path.basename(path)}:{i}",
                        "prompt": prompt,
                        "test_cases": ex["test_cases"],
                    }
                )
    print(rows[0])
    return rows


def build_codegen_prompt(question: str, style: str = "plain") -> str:
    if style == "plain":
        return (
            "You are a Python programming assistant.\n"
            "Solve the following programming problem and return only Python code "
            "inside a ```python ... ``` code block.\n\n"
            f"{question}\n"
        )
    raise ValueError(f"Unsupported prompt style: {style}")


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # TRL may return list[{"role":..., "content":...}] or nested structures.
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


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


def make_codegen_reward_fn(reward_url: str, timeout_s: int = 120, soft_reward: bool = False):
    def codegen_reward_fn(prompts, completions, **kwargs) -> List[float]:
        test_cases_list = kwargs["test_cases"]

        if isinstance(completions, str):
            completions_list = [completions]
        else:
            completions_list = list(completions)

        payload = []
        for prompt, completion, test_cases in zip(prompts, completions_list, test_cases_list):
            raw_text = _completion_to_text(completion)
            code = extract_python_code(raw_text)
            payload.append({"query": prompt, "response": code, "test_cases": test_cases})

        try:
            resp = requests.post(reward_url, data=json.dumps(payload), timeout=timeout_s)
            resp.raise_for_status()
            flags = resp.json()["flags"]
        except Exception as e:
            print(f"WARNING: reward call failed ({type(e).__name__}: {e}); returning zero rewards.")
            return [0.0 for _ in payload]

        rewards: List[float] = []
        for per_case_flags in flags:
            if not per_case_flags:
                rewards.append(0.0)
                continue
            if soft_reward:
                rewards.append(sum(float(bool(x)) for x in per_case_flags) / len(per_case_flags))
            else:
                rewards.append(float(all(bool(x) for x in per_case_flags)))
        return rewards

    return codegen_reward_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="Labeled programming jsonl files")
    p.add_argument("--model", required=True, help="HF model name or local checkpoint path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reward_url", default="http://localhost:8118/batched_unittest")
    p.add_argument("--prompt_style", default="plain")

    p.add_argument("--soft_reward", action="store_true", help="Use mean pass rate instead of pass-all reward")
    p.add_argument("--reward_timeout_s", type=int, default=120)

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--rollout_n", type=int, default=8)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--max_completion_length", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)

    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=1)

    p.add_argument("--deepspeed_cfg", default="/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/ds_zero3.json")
    p.add_argument("--vllm_tp", type=int, default=4)
    p.add_argument("--vllm_gpu_mem", type=float, default=0.7)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    rows = load_programming_jsonl(args.data, prompt_style=args.prompt_style)
    # input()
    if not rows:
        raise RuntimeError("No training rows loaded from --data")
    ds = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    with open(args.deepspeed_cfg, "r") as f:
        ds_cfg = json.load(f)
    ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_cfg["gradient_accumulation_steps"] = args.grad_accum

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.rollout_n,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=True,
        vllm_tensor_parallel_size=args.vllm_tp,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        vllm_mode="colocate",
        deepspeed=ds_cfg,
        bf16=args.bf16,
    )

    reward_fn = make_codegen_reward_fn(
        reward_url=args.reward_url,
        timeout_s=args.reward_timeout_s,
        soft_reward=args.soft_reward,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=ds,
        args=grpo_cfg,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    # main()
    data_path = ["/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/code/Reward-Hacking-main/examples/programming/data/all_labeled/data_1-4_tstMin6_tstMax50.jsonl",
                 "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/code/Reward-Hacking-main/examples/programming/data/all_labeled/data_5-10_tstMin6_tstMax50.jsonl"]
    rows = load_programming_jsonl(paths=data_path)

    
