import argparse
import json
import os

from train.eval.arlsat_eval import accuracy, build_sampling_kwargs as _build_sampling_kwargs, inference, load_test_data
from train.rh.rl_step_pipeline import select_train_subset


def build_sampling_kwargs(*, temperature: float, top_p: float, max_token: int) -> dict:
    return _build_sampling_kwargs(temperature=temperature, top_p=top_p, max_token=max_token)


def build_arlsat_eval_dir(*, eval_dir: str, eval_split: str, decode_tag: str) -> str:
    return os.path.join(os.path.expanduser(eval_dir), eval_split, decode_tag)


def _compute_train_slice_bounds(*, outer_step: int, train_slice_size: int, n_samples: int) -> tuple[int, int]:
    start = outer_step * (train_slice_size + n_samples)
    end = start + train_slice_size
    return start, end


def _load_train_subset(*, outer_step: int, train_slice_size: int, n_samples: int, sample_size: int):
    ds = load_train_data("arlsat")
    start, end = _compute_train_slice_bounds(
        outer_step=outer_step,
        train_slice_size=train_slice_size,
        n_samples=n_samples,
    )
    train_slice = ds[start:end]
    effective_sample_size = sample_size if sample_size > 0 else len(load_test_data())
    return select_train_subset(train_slice, sample_size=effective_sample_size, task="arlsat", outer_step=outer_step)


def _write_metrics_summary(eval_dir: str, *, args: argparse.Namespace, sample_size: int, acc: dict) -> None:
    payload = {
        "eval_split": args.eval_split,
        "decode_tag": args.decode_tag,
        "prompt_variant": "default",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sample_size": sample_size,
        "accuracy": acc["accuracy"],
        "reward": acc["accuracy"],
    }
    with open(os.path.join(eval_dir, "metrics_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run ARLSAT eval for a single model folder.")
    parser.add_argument("--model-dir", required=True, help="Path to model folder (e.g., .../s0/dpo)")
    parser.add_argument("--eval-dir", required=True, help="Path to eval output folder")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inference if inference.json exists")
    parser.add_argument("--do-gpt-eval", action="store_true", help="Also run GPT reasonableness eval")
    parser.add_argument("--decode-tag", default="train_decode")
    parser.add_argument("--eval-split", choices=["test", "train_subset"], default="test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--outer-step", type=int, default=0)
    parser.add_argument("--train-slice-size", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--max-token", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()

    eval_dir = os.path.expanduser(args.eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    inference_path = os.path.join(eval_dir, "inference.json")

    if args.eval_split == "test":
        ds = load_test_data()
    else:
        ds = _load_train_subset(
            outer_step=args.outer_step,
            train_slice_size=args.train_slice_size,
            n_samples=args.n_samples,
            sample_size=args.sample_size,
        )

    if args.skip_existing and os.path.exists(inference_path):
        print(f"[skip] inference exists: {inference_path}")
    else:
        print(f"[run] inference model={args.model_dir}")
        inference(
            model_name=os.path.expanduser(args.model_dir),
            ds=ds,
            output_dir=eval_dir,
            max_token=args.max_token,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    if args.do_gpt_eval:
        from train.eval.arlsat_eval import gpt_eval

        print("[run] gpt_eval")
        gpt_eval(output_dir=eval_dir)
        res = accuracy(output_dir=eval_dir, gpt_eval=True)
    else:
        res = accuracy(output_dir=eval_dir, gpt_eval=False)

    _write_metrics_summary(eval_dir, args=args, sample_size=len(ds), acc=res)
    print(f"[done] {res}")


if __name__ == "__main__":
    main()
