# File: train/eval/arlsat_eval_all_parallel.py
#
# Run ARLSAT eval (same logic as arlsat_eval.py) on all discovered output folders in parallel.
#
# Example:
#   python train/eval/arlsat_eval_all_parallel.py \
#     --max-workers 2 \
#     --gpu-groups "0,1,2,3;4,5,6,7"
#
# Notes:
# - This assumes each model folder is like: train/output/<exp>/<method>/sX/dpo
# - It mirrors eval outputs to: train/eval/data/<exp>/<method>/sX
# - `arlsat_eval.inference()` uses tensor_parallel_size=4, so each parallel job should get 4 GPUs.
#   Use --gpu-groups to isolate jobs.

import os
import re
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


PROJECT_ROOT = Path("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack")
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "train/output"
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "train/eval/data"


def discover_jobs(train_output_root: Path):
    jobs = []
    # discover all .../s*/dpo folders
    for dpo_dir in sorted(train_output_root.glob("*/*/s*/dpo")):
        if not dpo_dir.is_dir():
            continue

        seed_dir = dpo_dir.parent            # .../sX
        method_dir = seed_dir.parent         # .../<method>
        exp_dir = method_dir.parent          # .../<exp>

        seed = seed_dir.name
        method = method_dir.name
        exp = exp_dir.name

        if not re.fullmatch(r"s\d+", seed):
            continue

        eval_dir = EVAL_OUTPUT_ROOT / exp / method / seed
        jobs.append(
            {
                "exp": exp,
                "method": method,
                "seed": seed,
                "model_dir": str(dpo_dir),
                "eval_dir": str(eval_dir),
            }
        )
    return jobs


def _run_one(job, gpu_group=None, skip_if_done=True):
    # Set CUDA visibility BEFORE importing arlsat_eval/vllm in this process.
    if gpu_group:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group

    from train.eval.arlsat_eval import load_test_data, inference, accuracy

    eval_dir = Path(job["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    inference_path = eval_dir / "inference.json"
    acc_path = eval_dir / "acc_results.json"

    if skip_if_done and inference_path.exists() and acc_path.exists():
        with open(acc_path, "r") as f:
            acc = json.load(f)
        return {
            "status": "skipped",
            "job": job,
            "gpu_group": gpu_group,
            "acc": acc,
        }

    test_ds = load_test_data()

    if not inference_path.exists():
        inference(model_name=job["model_dir"], ds=test_ds, output_dir=str(eval_dir))
    acc = accuracy(output_dir=str(eval_dir), gpt_eval=False)

    return {
        "status": "done",
        "job": job,
        "gpu_group": gpu_group,
        "acc": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--gpu-groups",
        type=str,
        default="",
        help='Semicolon-separated GPU groups, e.g. "0,1,2,3;4,5,6,7". '
             "If empty, workers inherit current CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--include", type=str, default="", help="Only run jobs whose path contains this substring")
    parser.add_argument("--exclude", type=str, default="", help="Skip jobs whose path contains this substring")
    parser.add_argument("--force", action="store_true", help="Re-run even if inference.json and acc_results.json exist")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = discover_jobs(TRAIN_OUTPUT_ROOT)

    if args.include:
        jobs = [j for j in jobs if args.include in j["model_dir"]]
    if args.exclude:
        jobs = [j for j in jobs if args.exclude not in j["model_dir"]]

    if not jobs:
        print("No jobs found.")
        return

    gpu_groups = [g.strip() for g in args.gpu_groups.split(";") if g.strip()]
    if gpu_groups and args.max_workers > len(gpu_groups):
        print(f"[warn] max-workers={args.max_workers} > gpu-groups={len(gpu_groups)}; capping to {len(gpu_groups)}")
        args.max_workers = len(gpu_groups)

    print(f"Discovered {len(jobs)} jobs")
    for j in jobs:
        print(f"- {j['model_dir']}  ->  {j['eval_dir']}")

    if args.dry_run:
        return

    # Safer for CUDA/vLLM than fork
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=mp_ctx) as ex:
        futures = []
        for i, job in enumerate(jobs):
            gpu_group = gpu_groups[i % len(gpu_groups)] if gpu_groups else None
            futures.append(
                ex.submit(_run_one, job, gpu_group, not args.force)
            )

        done_cnt = 0
        skip_cnt = 0
        fail_cnt = 0

        for fut in as_completed(futures):
            try:
                out = fut.result()
                status = out["status"]
                job = out["job"]
                acc = out.get("acc", {})
                if status == "done":
                    done_cnt += 1
                else:
                    skip_cnt += 1
                print(
                    f"[{status}] {job['exp']}/{job['method']}/{job['seed']} "
                    f"gpu={out.get('gpu_group')} acc={acc.get('accuracy')}"
                )
            except Exception as e:
                fail_cnt += 1
                print(f"[failed] {repr(e)}")

    print(f"Summary: done={done_cnt}, skipped={skip_cnt}, failed={fail_cnt}")


if __name__ == "__main__":
    main()
import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack")
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "train/output"
SH_ROOT = PROJECT_ROOT / "train/sh_eval/arlsat_inference"
LOG_ROOT = SH_ROOT / "log"


def discover_jobs(train_output_root: Path):
    jobs = []
    # Expected layout: train/output/<exp>/<method>/sX/dpo
    for dpo_dir in sorted(train_output_root.glob("*/*/s*/dpo")):
        if not dpo_dir.is_dir():
            continue

        seed_dir = dpo_dir.parent
        method_dir = seed_dir.parent
        exp_dir = method_dir.parent

        seed = seed_dir.name
        method = method_dir.name
        exp = exp_dir.name

        if not re.fullmatch(r"s\d+", seed):
            continue

        jobs.append(
            {
                "exp": exp,
                "method": method,
                "seed": seed,
                "model_dir": str(dpo_dir),
            }
        )
    return jobs


def sh_name(job):
    return f"eval_{job['exp']}_{job['method']}_{job['seed']}.sh"


def job_name(job):
    base = f"eval_{job['method']}_{job['seed']}"
    # SLURM job names can be short; keep the identifying suffix.
    return base[:100]


def render_script(job, gpus=4, cpus=10, mem="220G", hours="08:00:00"):
    exp = job["exp"]
    method = job["method"]
    seed = job["seed"]
    script_stem = f"{exp}_{method}_{seed}"
    log_dir = LOG_ROOT / script_stem
    model_dir = job["model_dir"]
    eval_dir = PROJECT_ROOT / f"train/eval/data/{exp}/{method}/{seed}"

    return f"""#!/bin/bash
set -euo pipefail

#SBATCH --output={log_dir}/%j.out
#SBATCH --error={log_dir}/%j.err
#SBATCH --job-name={job_name(job)}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={hours}

mkdir -p "{log_dir}"
mkdir -p "{eval_dir}"

cd "{PROJECT_ROOT}"

python -m train.eval.arlsat_eval_single \\
  --model-dir "{model_dir}" \\
  --eval-dir "{eval_dir}" \\
  --skip-existing
"""


def main():
    parser = argparse.ArgumentParser(description="Generate one SLURM eval script per output folder (inference-first).")
    parser.add_argument("--include", default="", help="Only generate for model paths containing this substring")
    parser.add_argument("--exclude", default="", help="Skip model paths containing this substring")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument("--mem", default="220G")
    parser.add_argument("--time", default="08:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = discover_jobs(TRAIN_OUTPUT_ROOT)
    if args.include:
        jobs = [j for j in jobs if args.include in j["model_dir"]]
    if args.exclude:
        jobs = [j for j in jobs if args.exclude not in j["model_dir"]]

    SH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]

    for job in jobs:
        script_path = SH_ROOT / sh_name(job)
        script_text = render_script(
            job,
            gpus=args.gpus,
            cpus=args.cpus,
            mem=args.mem,
            hours=args.time,
        )
        print(f"{job['model_dir']} -> {script_path}")
        submit_lines.append(f"sbatch {script_path}")

        if not args.dry_run:
            script_path.write_text(script_text)
            script_path.chmod(0o755)

    submit_all = SH_ROOT / "submit_all.sh"
    if not args.dry_run:
        submit_all.write_text("\n".join(submit_lines) + "\n")
        submit_all.chmod(0o755)

    print(f"\nGenerated {len(jobs)} scripts in: {SH_ROOT}")
    print(f"Submit-all script: {submit_all}")


if __name__ == "__main__":
    main()
