import argparse
import json
import os
from typing import Dict, List

from train.eval.bigmath_eval import accuracy, build_sampling_kwargs as _build_sampling_kwargs, inference, load_data_bigmath
from train.rh.rl_step_pipeline import select_train_subset


def build_sampling_kwargs(*, temperature: float, top_p: float, max_token: int) -> dict:
    return _build_sampling_kwargs(temperature=temperature, top_p=top_p, max_token=max_token)


def build_bigmath_eval_dirs(*, eval_dir: str, eval_split: str, decode_tag: str) -> Dict[str, str]:
    base_dir = os.path.join(os.path.expanduser(eval_dir), eval_split, decode_tag)
    return {
        "cheat": os.path.join(base_dir, "cheat"),
        "no_cheat": os.path.join(base_dir, "no_cheat"),
    }


def _compute_train_slice_bounds(*, outer_step: int, train_slice_size: int, n_samples: int) -> tuple[int, int]:
    start = outer_step * (train_slice_size + n_samples)
    end = start + train_slice_size
    return start, end


def _load_train_subset(*, outer_step: int, train_slice_size: int, n_samples: int, sample_size: int, mix: bool) -> List[dict]:
    ds = load_train_data("bigmath", cheat=True, mix=mix)
    start, end = _compute_train_slice_bounds(
        outer_step=outer_step,
        train_slice_size=train_slice_size,
        n_samples=n_samples,
    )
    train_slice = ds[start:end]
    effective_sample_size = sample_size if sample_size > 0 else len(train_slice)
    return select_train_subset(train_slice, sample_size=effective_sample_size, task="bigmath", outer_step=outer_step)


def _write_metrics_summary(eval_dir: str, *, args: argparse.Namespace, sample_size: int, acc: dict) -> None:
    payload = {
        "eval_split": args.eval_split,
        "decode_tag": args.decode_tag,
        "prompt_variant": args.prompt_variant,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sample_size": sample_size,
        "accuracy": acc["accuracy"],
        "reward": acc["accuracy"],
    }
    with open(os.path.join(eval_dir, "metrics_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run BigMath eval for a single model folder.")
    parser.add_argument("--model-dir", required=True, help="Path to model folder (e.g., .../gradient/s0/sft)")
    parser.add_argument("--eval-dir", required=True, help="Path to eval output folder")
    parser.add_argument("--prompt-variant", choices=["cheat", "no_cheat"], default="cheat")
    parser.add_argument("--decode-tag", default="train_decode")
    parser.add_argument("--eval-split", choices=["test", "train_subset"], default="test")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inference if inference.json exists")
    parser.add_argument("--do-labeling", action="store_true", help="Run counterfactual labeling on correct generations")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of BigMath test samples (<=0 means all)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-token", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--outer-step", type=int, default=0)
    parser.add_argument("--train-slice-size", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    eval_dir = os.path.expanduser(args.eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    inference_path = os.path.join(eval_dir, "inference.json")

    if args.eval_split == "test":
        ds = load_data_bigmath(cheat=args.prompt_variant == "cheat", k=args.max_samples, mix=args.mix)
    else:
        if args.prompt_variant != "cheat":
            raise ValueError("BigMath train_subset evaluation only supports the cheat prompt training distribution.")
        ds = _load_train_subset(
            outer_step=args.outer_step,
            train_slice_size=args.train_slice_size,
            n_samples=args.n_samples,
            sample_size=args.sample_size,
            mix=args.mix,
        )

    for idx, example in enumerate(ds):
        example.setdefault("pid", idx)

    if args.skip_existing and os.path.exists(inference_path):
        print(f"[skip] inference exists: {inference_path}")
    else:
        inference(
            model_name=os.path.expanduser(args.model_dir),
            ds=ds,
            output_dir=eval_dir,
            max_token=args.max_token,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    acc = accuracy(output_dir=eval_dir)
    _write_metrics_summary(eval_dir, args=args, sample_size=len(ds), acc=acc)
    print(f"[done] {acc}")


if __name__ == "__main__":
    main()
