from train.grpo_train import *




parser = argparse.ArgumentParser()
add_grpo_args(parser=parser)
grpo_cfg = parser.parse_args()

# basic setting
step = 0
n_samples = 16
n_grpo_samples = 16
output_dir = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s{step}"
model_name = f"/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/train/output/s{step}/grpo"

# sampling responses


ds = load_data()[(step+1)*n_grpo_samples: (step+1)*n_grpo_samples+n_samples]
ds = Dataset.from_list(ds)

# sampling responses
# sample_batch_size = 4
# sample_responses(model_name=model_name, ds=ds, batch_size=sample_batch_size, output_dir=output_dir)

# load grpo samples
with open(os.path.join(output_dir, 'grpo_samples', 'samples.json'), 'r') as f:
    ds = json.load(f)

# labeling responses with gradient method
trainset = gradient_labeling(output_dir=output_dir, ds=ds, train_type="dpo", method="cluster")





