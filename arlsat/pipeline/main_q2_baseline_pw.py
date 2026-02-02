"""
main function for qwen2.5-3b on proof writer to get baseline trace score
"""



import os
import json
from datasets import Dataset
import numpy as np
import pandas as pd
import tiktoken
import re
import argparse
from tqdm.auto import tqdm

import math
import random
import glob


from collections import defaultdict
from utils_ import result_processer, gpt_completion, gpt_extract_eval_number
from big_math.trace.rh_model_setting import inference_on_ds
from big_math.trace.counterfact import load_arlsat_hint
from .rh_model_setting import pipeline_baseline


def main():
        # cheat
    model_name = "Qwen/Qwen2.5-3B" 
    save_dir = f'arlsat/pipeline/data/proof_q2.5_baseline'

    pipeline_baseline(model_name=model_name, save_dir=save_dir, data='proof')


        

        
if __name__ =="__main__":

    main()



