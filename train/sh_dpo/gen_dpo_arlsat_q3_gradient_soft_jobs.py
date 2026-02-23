#!/usr/bin/env python3
"""
Generate multiple SLURM job scripts for GRPO->labeling->DPO runs.

Each generated script is standalone (can be tested individually),
and a submit_all.sh is generated to sbatch all jobs together.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _sanitize_tag(s: str) -> str:
    if "Qwen" in s:
        return "base"
    elif "LLaMA" in s:
        return "sft"
    else:
        s = s.strip().replace("/", "_").replace(":", "_").replace("-", "_")
        return "".join(c for c in s if c.isalnum() or c in {"_", "."}).strip("_.")


def render_job(
    *,
    job_name: str,
    log_tag: str,
    output_root: str,
    grpo_init_model: str,
    labeling_method: str,
    cluster_method: str,
    cluster_semantics: str,
    baseline_km_centers: str,
    cluster_soft_prob_temp: float,
    cluster_soft_prob_n_runs: int,
) -> str:
    return f"""#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/sh_dpo/log/{log_tag}/%j.out
#SBATCH --error=/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/sh_dpo/log/{log_tag}/%j.err
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

dpo_output_dir="{output_root}"
grpo_output_dir="{output_root}"
grpo_init_model="{grpo_init_model}"

labeling_method="{labeling_method}"
cluster_method="{cluster_method}"
cluster_semantics="{cluster_semantics}"
cluster_soft_prob_temp={cluster_soft_prob_temp}
cluster_soft_prob_n_runs={cluster_soft_prob_n_runs}
baseline_km_centers={baseline_km_centers}

n_samples=32
n_grpo_training_samples=32
grpo_save_steps=4
grpo_batch_size_per_device=2
dpo_epochs=1

for STEP in $(seq 0 1 40); do
    echo "Training step: $STEP"
    grpo_input_model="$grpo_output_dir/$labeling_method/s$((STEP-1))/dpo"

    accelerate launch --num_processes 4 -m grpo_train \\
        --labeling_method $labeling_method \\
        --cluster_method $cluster_method \\
        --cluster_semantics $cluster_semantics \\
        --cluster_soft_prob_temp $cluster_soft_prob_temp \\
        --cluster_soft_prob_n_runs $cluster_soft_prob_n_runs \\
        --n_samples $n_samples \\
        --n_grpo_training_samples $n_grpo_training_samples \\
        --grpo_init_model $grpo_init_model \\
        --grpo_input_model $grpo_input_model \\
        --grpo_output_dir $grpo_output_dir \\
        --grpo_batch_size_per_device $grpo_batch_size_per_device \\
        --grpo_save_steps $grpo_save_steps \\
        --step $STEP

    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

    python -m grpo_train \\
        --labeling_method $labeling_method \\
        --cluster_method $cluster_method \\
        --cluster_semantics $cluster_semantics \\
        --baseline_km_centers $baseline_km_centers \\
        --n_samples $n_samples \\
        --n_grpo_training_samples $n_grpo_training_samples \\
        --grpo_init_model $grpo_init_model \\
        --grpo_input_model $grpo_input_model \\
        --cluster_soft_prob_temp $cluster_soft_prob_temp \\
        --cluster_soft_prob_n_runs $cluster_soft_prob_n_runs \\
        --grpo_output_dir $grpo_output_dir \\
        --grpo_sample \\
        --step $STEP

    python -m dpo_train \\
        --dpo_output_dir $dpo_output_dir \\
        --labeling_method $labeling_method \\
        --dpo_save_step 4 \\
        --dpo_epochs $dpo_epochs \\
        --step $STEP
done
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="train/sh/generated_dpo_arlsat_q3_soft")
    parser.add_argument("--labeling_method", default="gradient")
    parser.add_argument("--cluster_method", default="soft_prob")
    parser.add_argument("--baseline_km_centers", default=None)
    parser.add_argument("--cluster_soft_prob_temp", type=float, default=1.0)
    parser.add_argument("--cluster_soft_prob_n_runs", type=int, default=8)
    parser.add_argument(
        "--grpo_init_models",
        nargs="+",
        default=[
            "Qwen/Qwen3-4B",
            "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1/gradient/s0/grpo",
        ],
    )
    parser.add_argument(
        "--cluster_semantics_list",
        nargs="+",
        default=["majority", "baseline_centroids", "gpt"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]

    for model in args.grpo_init_models:
        # model_tag = _sanitize_tag(model.split("/")[-1] if "/" in model else model)
        model_tag = _sanitize_tag(model)
        for semantics in args.cluster_semantics_list:
            run_tag = f"arlsat_d1_soft_{model_tag}_{semantics}"
            job_name = f"dpo_{model_tag[:10]}_{semantics[:8]}"
            log_tag = f"dpo_gradient_arlsat_q3_{run_tag}"
            output_root = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/{run_tag}"
            baseline_km_centers = args.baseline_km_centers if semantics == "baseline_centroids" else None

            script_name = f"dpo_arlsat_q3_gradient_{model_tag}_{semantics}.sh"
            script_path = out_dir / script_name
            script_path.write_text(
                render_job(
                    job_name=job_name,
                    log_tag=log_tag,
                    output_root=output_root,
                    grpo_init_model=model,
                    baseline_km_centers=baseline_km_centers,
                    labeling_method=args.labeling_method,
                    cluster_method=args.cluster_method,
                    cluster_semantics=semantics,
                    cluster_soft_prob_temp=args.cluster_soft_prob_temp,
                    cluster_soft_prob_n_runs=args.cluster_soft_prob_n_runs,
                ),
                encoding="utf-8",
            )
            os.chmod(script_path, 0o750)
            submit_lines.append(f"sbatch {script_path}")

    submit_all = out_dir / "submit_all.sh"
    submit_all.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    os.chmod(submit_all, 0o750)
    print(f"Generated scripts in: {out_dir}")
    print(f"Submit all with: {submit_all}")


if __name__ == "__main__":
    main()

