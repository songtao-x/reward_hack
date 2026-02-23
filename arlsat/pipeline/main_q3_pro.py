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


def gradient(use_soft_f1_kmeans=True, clustered_soft_f1_only=False,
             soft_f1_max_iter=200, soft_f1_lr=1e-2, soft_f1_temp=1.0):
    analyzer = GradientAnalyzer()

    for s in range(5, 50, 5):
        model_name = f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/verl/examples/grpo_trainer/checkpoints/test_grpo_arqa/test_grpo_arqa_q3_rh/global_step_{s}/actor/actor_hf'
        save_dir = f'arlsat/pipeline/data/arlsat_q3_pro_test_grpo_normal_step_{s}'

        with open(os.path.join(save_dir, 'true_set.json'), 'r') as f:
            true_set = json.load(f)
        
        with open(os.path.join(save_dir, 'false_set.json'), 'r') as f:
            false_set = json.load(f)
        
        km_cluster = "baseline_km"
        svm_model = "baseline_svm"
        in_test= False

        km_cluster = None
        # svm_model = None
        # in_test = True

        gradient_analysis(analyzer, model_name=model_name, 
                            true_set=true_set, false_set=false_set,
                            save_model="baseline", 
                            baseline_km_centers=km_cluster, baseline_model=None,
                            save_dir=save_dir, get_gradient=False, normalized=True,
                            use_soft_f1_kmeans=use_soft_f1_kmeans,
                            soft_f1_max_iter=soft_f1_max_iter,
                            soft_f1_lr=soft_f1_lr,
                            soft_f1_temp=soft_f1_temp,
                            clustered_soft_f1_only=clustered_soft_f1_only)
        # input()

        

        
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient_only", action="store_true", help="only used for gradient test")
    parser.set_defaults(use_soft_f1_kmeans=True)
    parser.add_argument("--use_soft_f1_kmeans", dest="use_soft_f1_kmeans", action="store_true", help="enable soft-F1-guided kmeans in gradient analysis")
    parser.add_argument("--no_use_soft_f1_kmeans", dest="use_soft_f1_kmeans", action="store_false", help="disable soft-F1-guided kmeans")
    parser.add_argument("--clustered_soft_f1_only", action="store_true", help="compute soft-F1 from vanilla kmeans clustering results (no soft-F1 center optimization)")
    parser.add_argument("--soft_f1_max_iter", type=int, default=200)
    parser.add_argument("--soft_f1_lr", type=float, default=1e-2)
    parser.add_argument("--soft_f1_temp", type=float, default=0.2)

    args = parser.parse_args()
    if args.clustered_soft_f1_only:
        args.use_soft_f1_kmeans = False

    if not args.gradient_only:
        main()
    gradient(use_soft_f1_kmeans=args.use_soft_f1_kmeans,
             clustered_soft_f1_only=args.clustered_soft_f1_only,
             soft_f1_max_iter=args.soft_f1_max_iter,
             soft_f1_lr=args.soft_f1_lr,
             soft_f1_temp=args.soft_f1_temp)
