#!/bin/bash
iteration=$1
overall_prefix=$2
mode=${3:-train}
prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_gan

# WFST decode the phoneme sequences
cd WFST-decoder
python3 scripts/decode.py --set_type test --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
if [ $mode = 'train' ]; then
python3 scripts/decode.py --set_type train --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
fi
cd ../

# Evalution
#python3 eval_per.py --bnd_type $bnd_type --set_type train --lm_type $setting \
                   #--data_path $DATA_PATH --prefix $prefix \
                   #--file_name train_output.txt | tee $DATA_PATH/result/${prefix}.train.log
python3 eval_per.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                   --data_path $DATA_PATH --prefix $prefix \
                   --file_name test_output.txt | tee -a $DATA_PATH/result/${prefix}.log
