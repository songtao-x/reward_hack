#!/bin/bash


python -m trace.rh_model_setting \
    > "rh_model_setting_2.log" 2>&1

python -m trace.rh_model_setting \
    --mix \
    > "rh_model_setting_mix_2.log" 2>&1



