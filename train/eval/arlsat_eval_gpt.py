
import os
import argparse




from train.eval.arlsat_eval import accuracy, gpt_eval


def main(method):
    # test_ds = load_test_data()

    # if method == "gradient":
    # gradient
    for s in range(10):
        load_dir = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/arlsat_d1/gradient/s{s}"
        model_name = load_dir + '/dpo'

        output_dir = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/eval/data/arlsat_d1/{method}/s{s}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Running evaluation on {output_dir}")
        # inference_ds = inference(model_name=model_name, ds=test_ds, output_dir=output_dir)
        gpt_eval(output_dir=output_dir)

        acc = accuracy(output_dir=output_dir, gpt_eval=True)
        print(acc)

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="gradient")

args = parser.parse_args()

main(method=args.method)
