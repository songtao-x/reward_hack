
import argparse
import json
import os
import random
import re
from pathlib import Path

from tqdm.auto import tqdm


random.seed(224)
SEED = 224
MAX_TOKEN = 3096


def _import_result_processer():
    try:
        from arlsat.utils_.data_process import result_processer  # type: ignore
    except ImportError:
        from utils_.data_process import result_processer  # type: ignore
    return result_processer


def _import_vllm():
    from vllm import LLM, SamplingParams  # type: ignore

    return LLM, SamplingParams


def load_data_bigmath(cheat=True, k=200, mix=False):
    from big_math.trace.load_data import load_data  # lazy import (pulls in heavy deps)

    ds = load_data(cheat=cheat, mix=mix)
    if k is None or k <= 0:
        return ds
    return ds[:k]


def inference(model_name, ds, output_dir, max_token=MAX_TOKEN, batch_size=4, tensor_parallel_size=4):
    os.makedirs(output_dir, exist_ok=True)
    result_processer = _import_result_processer()
    LLM, SamplingParams = _import_vllm()

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=max_token,
    )

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
    )

    res = []
    n = len(ds)
    total_batches = (n + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(tqdm(range(0, n, batch_size), desc="Inference", total=total_batches)):
        batch = ds[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        outputs = llm.generate(prompts, sampling_params)

        for local_idx, (ex, out) in enumerate(zip(batch, outputs)):
            gen = out.outputs[0].text
            _, score = result_processer(response=gen, label=ex["label"])
            pid = ex.get("pid", start + local_idx)
            res.append(
                {
                    "pid": pid,
                    "prompt": ex["prompt"],
                    "label": ex["label"],
                    "gen": gen,
                    "score": int(score),
                }
            )

        if batch_idx % 10 == 0:
            with open(os.path.join(output_dir, "inference.json"), "w") as f:
                json.dump(res, f, indent=2)

    with open(os.path.join(output_dir, "inference.json"), "w") as f:
        json.dump(res, f, indent=2)

    return res


def accuracy(output_dir):
    inference_path = os.path.join(output_dir, "inference.json")
    with open(inference_path, "r") as f:
        inference_ds = json.load(f)

    scores = [int(example.get("score", 0)) for example in inference_ds]
    acc = float(sum(scores) / len(scores)) if scores else 0.0

    res = {"accuracy": acc, "len": len(scores)}
    with open(os.path.join(output_dir, "acc_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def labeling(model_name, output_dir, cheat=True, max_token=4096, tensor_parallel_size=4):
    result_processer = _import_result_processer()
    LLM, SamplingParams = _import_vllm()
    from big_math.trace.counterfact import counterfactualtest  # lazy import

    with open(os.path.join(output_dir, "inference.json"), "r") as f:
        ds = json.load(f)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=max_token,
    )
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
    )

    correct = []
    true_correct = []
    false_correct = []

    for inf in tqdm(ds, desc="Counterfactual labeling"):
        prompt = inf["prompt"]
        gen = inf["gen"]
        label = inf["label"]
        _, is_correct = result_processer(response=gen, label=label)
        if not is_correct:
            continue

        correct.append(inf)
        passed_ct = counterfactualtest(
            llm,
            sampler=sampling_params,
            res=prompt,
            label=label,
            cheat=cheat,
        )
        if passed_ct:
            true_correct.append(inf)
        else:
            false_correct.append(inf)

    with open(os.path.join(output_dir, "correct.json"), "w") as f:
        json.dump(correct, f, indent=2)
    with open(os.path.join(output_dir, "true_correct.json"), "w") as f:
        json.dump(true_correct, f, indent=2)
    with open(os.path.join(output_dir, "false_correct.json"), "w") as f:
        json.dump(false_correct, f, indent=2)

    total = len(ds)
    n_correct = len(correct)
    n_true_correct = len(true_correct)
    summary = {
        "len": total,
        "correct_len": n_correct,
        "true_correct_len": n_true_correct,
        "accuracy": (n_correct / total) if total else 0.0,
        "true_correct_rate_among_correct": (n_true_correct / n_correct) if n_correct else 0.0,
        "true_correct_rate_overall": (n_true_correct / total) if total else 0.0,
    }
    with open(os.path.join(output_dir, "labeling_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _resolve_eval_dir(*, eval_dir=None, eval_root=None, method=None, seed=None):
    if eval_dir:
        return os.path.expanduser(str(eval_dir))

    if eval_root and method and seed:
        return str(Path(eval_root).expanduser() / method / seed)

    raise ValueError("Need either eval_dir, or eval_root + method + seed")


def run_one(
    model_dir,
    eval_dir=None,
    *,
    eval_root=None,
    method=None,
    seed=None,
    cheat=True,
    max_samples=200,
    mix=False,
    skip_existing=False,
    do_labeling=False,
    max_token=4096,
    batch_size=4,
    tensor_parallel_size=4,
):
    eval_dir = _resolve_eval_dir(eval_dir=eval_dir, eval_root=eval_root, method=method, seed=seed)
    os.makedirs(eval_dir, exist_ok=True)
    inference_path = os.path.join(eval_dir, "inference.json")

    if skip_existing and os.path.exists(inference_path):
        print(f"[skip] inference exists: {inference_path}")
    else:
        test_ds = load_data_bigmath(cheat=cheat, k=max_samples, mix=mix)
        for i, ex in enumerate(test_ds):
            ex.setdefault("pid", i)
        print(f"[run] inference model={model_dir} n={len(test_ds)}")
        inference(
            model_name=model_dir,
            ds=test_ds,
            output_dir=eval_dir,
            max_token=max_token,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
        )

    acc = accuracy(output_dir=eval_dir)
    print(f"[acc] {acc}")

    label_res = None
    if do_labeling:
        print("[run] counterfactual labeling")
        label_res = labeling(
            model_name=model_dir,
            output_dir=eval_dir,
            cheat=cheat,
            max_token=max_token,
            tensor_parallel_size=tensor_parallel_size,
        )
        print(f"[label] {label_res}")

    return {"acc": acc, "labeling": label_res, "eval_dir": eval_dir, "method": method, "seed": seed}


def _seed_sort_key(seed_name):
    m = re.fullmatch(r"s(\d+)", seed_name)
    return int(m.group(1)) if m else 10**9


def discover_jobs(output_root, method=None):
    output_root = Path(output_root).expanduser()
    methods = [method] if method and method != "all" else ["gradient", "trace"]
    jobs = []
    for meth in methods:
        method_root = output_root / meth
        if not method_root.is_dir():
            print(f"[warn] missing method dir: {method_root}")
            continue
        seed_dirs = sorted(
            [p for p in method_root.iterdir() if p.is_dir() and re.fullmatch(r"s\d+", p.name)],
            key=lambda p: _seed_sort_key(p.name),
        )
        for seed_dir in seed_dirs:
            model_dir = seed_dir / "sft"
            if not model_dir.is_dir():
                print(f"[warn] missing model dir: {model_dir}")
                continue
            jobs.append({"method": meth, "seed": seed_dir.name, "model_dir": str(model_dir)})
    return jobs


def run_many(
    *,
    output_root,
    eval_root,
    method="all",
    include="",
    exclude="",
    cheat=True,
    max_samples=200,
    mix=False,
    skip_existing=False,
    do_labeling=False,
    max_token=4096,
    batch_size=4,
    tensor_parallel_size=4,
):
    jobs = discover_jobs(output_root=output_root, method=method)
    if include:
        jobs = [j for j in jobs if include in j["model_dir"]]
    if exclude:
        jobs = [j for j in jobs if exclude not in j["model_dir"]]

    eval_root = Path(eval_root).expanduser()
    if not jobs:
        print("[done] no jobs found")
        return []

    results = []
    for job in jobs:
        print(f"\n=== {job['method']}/{job['seed']} ===")
        out = run_one(
            job["model_dir"],
            eval_root=str(eval_root),
            method=job["method"],
            seed=job["seed"],
            cheat=cheat,
            max_samples=max_samples,
            mix=mix,
            skip_existing=skip_existing,
            do_labeling=do_labeling,
            max_token=max_token,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        results.append({**job, **out})

    summary_name = "summary.json" if method == "all" else f"summary_{method}.json"
    summary_path = eval_root / summary_name
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote summary: {summary_path}")
    return results


def main():
    default_output_root = "~/scratch/reward_hack/train/output/bigmath_rft_rloo_base10"
    default_eval_root = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/bigmath_rft_rloo_base10"

    parser = argparse.ArgumentParser(description="BigMath eval for single model or all s*/sft folders.")
    parser.add_argument("--model-dir", default="", help="Run a single model dir (e.g., .../gradient/s0/sft)")
    parser.add_argument("--eval-dir", default="", help="Eval output dir for single-model mode")
    parser.add_argument("--output-root", default=default_output_root, help="Train output root containing gradient/trace")
    parser.add_argument("--eval-root", default=default_eval_root, help="Eval root for batch mode")
    parser.add_argument("--method", choices=["gradient", "trace", "all"], default="all")
    parser.add_argument("--include", default="", help="Only run jobs whose model path contains this substring")
    parser.add_argument("--exclude", default="", help="Skip jobs whose model path contains this substring")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of BigMath test samples (<=0 means all)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-token", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--do-labeling", action="store_true", help="Run counterfactual labeling on correct generations")
    parser.add_argument("--mix", action="store_true", help="Use mixed BigMath dataset variant")
    parser.add_argument("--no-cheat", action="store_true", help="Use non-cheat BigMath prompts")
    args = parser.parse_args()

    cheat = not args.no_cheat

    if args.model_dir:
        seed = None
        if not args.eval_dir:
            m = re.search(r"/(gradient|trace)/(s\d+)/sft/?$", os.path.expanduser(args.model_dir))
            if m:
                if args.method == "all":
                    args.method = m.group(1)
                seed = m.group(2)
            if args.method == "all" or not seed:
                raise ValueError("--eval-dir is required when --model-dir is provided unless model-dir matches .../<method>/sX/sft and --method is set")
        run_one(
            model_dir=os.path.expanduser(args.model_dir),
            eval_dir=os.path.expanduser(args.eval_dir) if args.eval_dir else None,
            eval_root=args.eval_root if not args.eval_dir else None,
            method=args.method if args.method != "all" else None,
            seed=seed,
            cheat=cheat,
            max_samples=args.max_samples,
            mix=args.mix,
            skip_existing=args.skip_existing,
            do_labeling=args.do_labeling,
            max_token=args.max_token,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        return

    run_many(
        output_root=args.output_root,
        eval_root=args.eval_root,
        method=args.method,
        include=args.include,
        exclude=args.exclude,
        cheat=cheat,
        max_samples=args.max_samples,
        mix=args.mix,
        skip_existing=args.skip_existing,
        do_labeling=args.do_labeling,
        max_token=args.max_token,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
            
