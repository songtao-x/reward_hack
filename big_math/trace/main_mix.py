import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import re
import argparse

import random
import glob

from .rh_model_setting import main_arlsat, pipeline_trace
from .gradient import big_math_gradient
from icl.gradient.analysis import GradientAnalyzer


random.seed(224)

if __name__ == '__main__':
    

    analyzer = GradientAnalyzer()

    # for s in range(30, 50, 5):
    #     model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
    #     save_dir = f'trace/data/rloo_mix_step_{s}'
    #     print(f'Processing model: {model_name}')
    #     pipeline_trace(model_name=model_name, save_dir=save_dir)


        


    for s in range(30, 50, 5):
        model_name = f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}"
        save_dir = f'trace/data/rloo_mix_step_{s}'
        big_math_gradient(analyzer, model_name=model_name, save_dir=save_dir, get_gradient=False)







