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

### Training Process
cd src/GAN-based-model

prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_gan

# inference GAN and output phoneme posterior
python3 main.py --mode test_posterior --cuda_id 0 \
               --bnd_type $bnd_type --iteration $iteration \
               --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${prefix} \
               --config ./config.yaml \
               --prefix $prefix 

#python3 main.py --mode test --cuda_id 0 \
               #--bnd_type $bnd_type --iteration $iteration \
               #--setting $setting \
               #--data_dir $DATA_PATH \
               #--save_dir $DATA_PATH/save/${prefix} \
               #--config "./config.yaml"

cd $current_path

