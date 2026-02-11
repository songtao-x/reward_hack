"""
main function for qwen3_4b model on extra testset (730) with official eval prompt
"""


import os
import json
import argparse


from .rh_model_setting import pipeline, get_trace, get_gradient_score, GradientAnalyzer, gradient_analysis


def main():
    for s in range(5, 50, 5):
        model_name = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf'
        save_dir = f'arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}'

        pipeline(model_name=model_name, save_dir=save_dir, data='arlsat_pro', threshold=0.2904761904761905)


def gradient():
    analyzer = GradientAnalyzer()
    for s in range(5, 50, 5):
        model_name = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf'
        save_dir = f'arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}'

        with open(os.path.join(save_dir, 'true_set.json'), 'r') as f:
            true_set = json.load(f)
        
        with open(os.path.join(save_dir, 'false_set.json'), 'r') as f:
            false_set = json.load(f)

        gradient_analysis(analyzer, model_name=model_name, 
                            true_set=true_set, false_set=false_set, 
                            save_dir=save_dir, get_gradient=True, normalized=True)

        

        
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient_only", action="store_true", help="only used for gradient test")

    args = parser.parse_args()

    if not args.gradient_only:
        main()
    gradient()



