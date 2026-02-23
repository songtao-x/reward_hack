import argparse
import os

from train.eval.arlsat_eval import load_test_data, inference, accuracy, gpt_eval


def main():
    parser = argparse.ArgumentParser(description="Run ARLSAT eval for a single model folder.")
    parser.add_argument("--model-dir", required=True, help="Path to model folder (e.g., .../s0/dpo)")
    parser.add_argument("--eval-dir", required=True, help="Path to eval output folder")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inference if inference.json exists")
    parser.add_argument("--do-gpt-eval", action="store_true", help="Also run GPT reasonableness eval")
    args = parser.parse_args()

    os.makedirs(args.eval_dir, exist_ok=True)
    inference_path = os.path.join(args.eval_dir, "inference.json")

    test_ds = load_test_data()

    if args.skip_existing and os.path.exists(inference_path):
        print(f"[skip] inference exists: {inference_path}")
    else:
        print(f"[run] inference model={args.model_dir}")
        inference(model_name=args.model_dir, ds=test_ds, output_dir=args.eval_dir)

    if args.do_gpt_eval:
        print("[run] gpt_eval")
        gpt_eval(output_dir=args.eval_dir)
        res = accuracy(output_dir=args.eval_dir, gpt_eval=True)
    else:
        res = accuracy(output_dir=args.eval_dir, gpt_eval=False)

    print(f"[done] {res}")


if __name__ == "__main__":
    main()
