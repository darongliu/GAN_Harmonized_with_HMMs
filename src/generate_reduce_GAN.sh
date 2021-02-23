#!/bin/bash
iteration=$1
gan_config=$2
overall_prefix=$3

prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_gan

# Train GAN and output phoneme posterior
cd GAN-based-model

mkdir -p $DATA_PATH/save/${prefix}/
mkdir -p $DATA_PATH/result
cp $gan_config $DATA_PATH/save/${prefix}/gan_config.yaml

python3 main.py --mode test_reduce --cuda_id 0 \
               --bnd_type $bnd_type --iteration $iteration \
               --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${prefix} \
               --config $gan_config \
               --prefix $prefix \
               --overall_prefix $overall_prefix

#python3 main.py --mode test --cuda_id 0 \
               #--bnd_type $bnd_type --iteration $iteration \
               #--setting $setting \
               #--data_dir $DATA_PATH \
               #--save_dir $DATA_PATH/save/${prefix} \
               #--config "./config.yaml"

cd ../