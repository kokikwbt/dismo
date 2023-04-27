#!/bin/bash

TRUE=1
FALSE=0

query=$1

# Window Setting
start_point=0
init_wd=104
slide_wd=104
report_wd=4  # fixed
forecast_wd=52

# Options

grid_search=$TRUE
# grid_search=$FALSE

interaction_type="full"
# interaction_type="self" # i.e., no interactions

allow_regime_shift=$TRUE
# allow_regime_shift=$FALSE


python3 main.py --dataset "gtrends" \
                --outdir "out/test/" \
                --query "${query}" \
                --minmax_scale \
                --interaction_type "full" \
                --n_interaction 4 \
                --n_seasonality 5 \
                --min_interaction 2  \
                --max_interaction 8 \
                --min_seasonality 2  \
                --max_seasonality 8 \
                --n_season 52 \
                --max_iter 5 \
                --online_max_iter 5 \
                --max_time 1 \
                --n_trial 5 \
                --start_point $start_point \
                --max_forecast_step $forecast_wd \
                --report_step $report_wd \
                --init_window_size $init_wd \
                --slide_window_size $slide_wd \
                --interaction_type $interaction_type \
                --update_seasonality \
                --regime_shift $allow_regime_shift \
                --grid_search $grid_search
