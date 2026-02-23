import argparse
import os

from train.eval.bigmath_eval import run_one


def main():
    parser = argparse.ArgumentParser(description="Run BigMath eval for a single model folder.")
    parser.add_argument("--model-dir", required=True, help="Path to model folder (e.g., .../gradient/s0/sft)")
    parser.add_argument("--eval-dir", required=True, help="Path to eval output folder")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inference if inference.json exists")
    parser.add_argument("--do-labeling", action="store_true", help="Run counterfactual labeling on correct generations")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of BigMath test samples (<=0 means all)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-token", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--no-cheat", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.expanduser(args.eval_dir), exist_ok=True)
    res = run_one(
        model_dir=os.path.expanduser(args.model_dir),
        eval_dir=os.path.expanduser(args.eval_dir),
        cheat=not args.no_cheat,
        max_samples=args.max_samples,
        mix=args.mix,
        skip_existing=args.skip_existing,
        do_labeling=args.do_labeling,
        max_token=args.max_token,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print(f"[done] {res}")


if __name__ == "__main__":
    main()
