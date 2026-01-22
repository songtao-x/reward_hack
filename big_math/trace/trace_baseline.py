from .rh_model_setting import baseline_trace
import os
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-4B")
    parser.add_argument('--arlsat', action='store_true', help='whether to run arlsat baseline')
    args = parser.parse_args()

    print(f'Running baseline trace for model: {args.model_name}, arlsat: {args.arlsat}')
    
    save_dir = f'trace/data/{args.model_name}'
    baseline_trace(model_name=args.model_name, save_dir=save_dir, arlsat=args.arlsat)


