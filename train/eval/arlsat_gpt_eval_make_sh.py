# File: train/eval/arlsat_eval_make_gpt_sh.py
import argparse
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack")
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "train/output"
SH_ROOT = PROJECT_ROOT / "train/eval/sh_eval/arlsat_gpt_eval"


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
                "eval_dir": str(PROJECT_ROOT / f"train/eval/data/{exp}/{method}/{seed}"),
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

    out = []
    for (exp, method), items in sorted(groups.items()):
        out.append(
            {
                "exp": exp,
                "method": method,
                "jobs": sorted(items, key=lambda x: seed_num(x["seed"])),
            }
        )
    return out


def sh_name(group):
    return f"gpt_eval_{group['exp']}_{group['method']}.sh"


def render_script(group):
    lines = [
        "#!/bin/bash",
        "",
        f'cd "{PROJECT_ROOT}"',
        "",
    ]

    for job in group["jobs"]:
        eval_dir = job["eval_dir"]
        model_dir = job["model_dir"]
        inference_json = f"{eval_dir}/inference.json"

        lines += [
            f'echo "=== {group["exp"]}/{group["method"]}/{job["seed"]} ==="',
            f'if [ ! -f "{inference_json}" ]; then',
            f'  echo "Skip (missing inference): {inference_json}"',
            "  continue",
            "fi",
            # Uses current arlsat_eval_single.py path, skips inference and runs GPT eval + accuracy.
            f'python -m train.eval.arlsat_eval_single \\',
            f'  --model-dir "{model_dir}" \\',
            f'  --eval-dir "{eval_dir}" \\',
            f'  --skip-existing \\',
            f'  --do-gpt-eval',
            "",
        ]

    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate CPU-only GPT-eval shell scripts (one per folder with all s*).")
    parser.add_argument("--include", default="", help="Only generate for paths containing this substring")
    parser.add_argument("--exclude", default="", help="Skip paths containing this substring")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = discover_jobs(TRAIN_OUTPUT_ROOT)
    if args.include:
        jobs = [j for j in jobs if args.include in j["model_dir"]]
    if args.exclude:
        jobs = [j for j in jobs if args.exclude not in j["model_dir"]]

    groups = group_jobs(jobs)
    SH_ROOT.mkdir(parents=True, exist_ok=True)

    run_all_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for group in groups:
        script_path = SH_ROOT / sh_name(group)
        script_text = render_script(group)
        print(f'{group["exp"]}/{group["method"]} ({len(group["jobs"])} seeds) -> {script_path}')
        run_all_lines.append(f"bash {script_path}")

        if not args.dry_run:
            script_path.write_text(script_text)
            script_path.chmod(0o755)

    run_all = SH_ROOT / "run_all.sh"
    if not args.dry_run:
        run_all.write_text("\n".join(run_all_lines) + "\n")
        run_all.chmod(0o755)

    print(f"\nGenerated {len(groups)} GPT-eval scripts in {SH_ROOT}")
    print(f"Run all: {run_all}")


if __name__ == "__main__":
    main()
