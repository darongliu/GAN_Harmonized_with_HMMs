#!/bin/bash
iteration=$1
gan_config=$2
overall_prefix=$3
bnd_description=$4

prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_gan

# Train GAN and output phoneme posterior
cd GAN-based-model

mkdir -p $DATA_PATH/save/${prefix}/
mkdir -p $DATA_PATH/result
cp $gan_config $DATA_PATH/save/${prefix}/gan_config.yaml

python3 main.py --mode train --cuda_id 0 \
               --bnd_type $bnd_type --iteration $iteration \
               --num_workers $jobs \
               --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${prefix} \
               --config $gan_config \
               --prefix $prefix \
               --overall_prefix $overall_prefix \
               --bnd_description $bnd_description

# python3 main.py --mode load --cuda_id 0 \
#                --bnd_type $bnd_type --iteration $iteration \
#                --num_workers $jobs \
#                --setting $setting \
#                --data_dir $DATA_PATH \
#                --save_dir $DATA_PATH/save/${prefix} \
#                --load_ckpt ckpt_9000.pth \
#                --config $gan_config \
#                --prefix $prefix \
#                --overall_prefix $overall_prefix

#python3 main.py --mode test --cuda_id 0 \
               #--bnd_type $bnd_type --iteration $iteration \
               #--setting $setting \
               #--data_dir $DATA_PATH \
               #--save_dir $DATA_PATH/save/${prefix} \
               #--config "./config.yaml"

cd ../
