# !/bin/bash                                                                                                                2 
current_path=`pwd`
current_dir=`basename "$current_path"`

if [ "GAN_Harmonized_with_HMMs" != "$current_dir" ]; then
    echo "You should run this script in GAN_Harmonized_with_HMMs/ directory!!"
    exit 1
fi

### Setting for Kaldi
. ./cmd.sh
. ./path.sh

### Experimental Setting 
. ./config.sh

### overall prefix
overall_prefix=$1
iteration=$2
load_ckpt=$3 # ckpt name

### Training Process
cd src/GAN-based-model

prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_gan

# inference GAN and output phoneme posterior
python3 main.py --mode eval --cuda_id 0 \
               --bnd_type $bnd_type --iteration $iteration \
               --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${prefix} \
               --config "./config.yaml" \
               --prefix $prefix \
               --load_ckpt $load_ckpt

#python3 main.py --mode test --cuda_id 0 \
               #--bnd_type $bnd_type --iteration $iteration \
               #--setting $setting \
               #--data_dir $DATA_PATH \
               #--save_dir $DATA_PATH/save/${prefix} \
               #--config "./config.yaml"

cd ../ 

# WFST decode the phoneme sequences
cd WFST-decoder
python3 scripts/decode.py --set_type test --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
python3 scripts/decode.py --set_type train --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
cd ../

# Evalution
#python3 eval_per.py --bnd_type $bnd_type --set_type train --lm_type $setting \
                   #--data_path $DATA_PATH --prefix $prefix \
                   #--file_name train_output.txt | tee $DATA_PATH/result/${prefix}.train.log
python3 eval_per.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                   --data_path $DATA_PATH --prefix $prefix \
                   --file_name test_output.txt | tee -a $DATA_PATH/result/${prefix}.log

cd $current_path

