import argparse
import re
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack")
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "train/output"
SH_ROOT = PROJECT_ROOT / "train/eval/sh_eval/arlsat_inference"
LOG_ROOT = SH_ROOT / "log"


def discover_jobs(train_output_root: Path):
    jobs = []
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


def seed_num(seed_name: str) -> int:
    m = re.fullmatch(r"s(\d+)", seed_name)
    return int(m.group(1)) if m else 10**9


def group_jobs(jobs):
    groups = defaultdict(list)
    for job in jobs:
        groups[(job["exp"], job["method"])].append(job)

    grouped = []
    for (exp, method), items in sorted(groups.items()):
        items = sorted(items, key=lambda x: seed_num(x["seed"]))
        grouped.append({"exp": exp, "method": method, "jobs": items})
    return grouped


def sh_name(group):
    return f"eval_{group['exp']}_{group['method']}.sh"


def job_name(group):
    return f"eval_{group['method']}"[:100]


def render_script(group, gpus=4, cpus=10, mem="220G", hours="08:00:00"):
    exp = group["exp"]
    method = group["method"]
    log_dir = LOG_ROOT / f"{exp}_{method}"

    body_lines = []
    for job in group["jobs"]:
        seed = job["seed"]
        model_dir = job["model_dir"]
        eval_dir = PROJECT_ROOT / f"train/eval/data/{exp}/{method}/{seed}"
        body_lines.extend(
            [
                f'mkdir -p "{eval_dir}"',
                "python -m train.eval.arlsat_eval_single \\",
                f'  --model-dir "{model_dir}" \\',
                f'  --eval-dir "{eval_dir}" \\',
                "  --skip-existing",
                "",
            ]
        )

    body = "\n".join(body_lines).rstrip()

    return f"""#!/bin/bash

#SBATCH --output={log_dir}/%j.out
#SBATCH --error={log_dir}/%j.err
#SBATCH --job-name={job_name(group)}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time={hours}
#SBATCH --mail-user=songtao2@ualberta.ca


mkdir -p "{log_dir}"

cd "{PROJECT_ROOT}"

{body}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate one SLURM eval shell script per output folder.")
    parser.add_argument("--include", default="", help="Only generate for model paths containing this substring")
    parser.add_argument("--exclude", default="", help="Skip model paths containing this substring")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument("--mem", default="300G")
    parser.add_argument("--time", default="07:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = discover_jobs(TRAIN_OUTPUT_ROOT)
    if args.include:
        jobs = [j for j in jobs if args.include in j["model_dir"]]
    if args.exclude:
        jobs = [j for j in jobs if args.exclude not in j["model_dir"]]

    SH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    groups = group_jobs(jobs)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for group in groups:
        script_path = SH_ROOT / sh_name(group)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_text = render_script(group, gpus=args.gpus, cpus=args.cpus, mem=args.mem, hours=args.time)
        print(
            f"{group['exp']}/{group['method']} ({len(group['jobs'])} seeds)"
            f" -> {script_path}"
        )
        submit_lines.append(f"sbatch {script_path}")

        if not args.dry_run:
            script_path.write_text(script_text)
            script_path.chmod(0o755)

    submit_all = SH_ROOT / "submit_all.sh"
    if not args.dry_run:
        submit_all.write_text("\n".join(submit_lines) + "\n")
        submit_all.chmod(0o755)

    print(f"\nGenerated {len(groups)} scripts in {SH_ROOT}")
    print(f"Submit via: {submit_all}")


if __name__ == "__main__":
    main()
