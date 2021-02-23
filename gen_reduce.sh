# !/bin/bash

### Setting for Kaldi
. ./cmd.sh
. ./path.sh


### Experimental Setting
. ./config.sh

### overall prefix
gan_config=$(readlink -m $1)
overall_prefix=$2

iteration=1

### Preprocess TIMIT
# bash preprocess.sh


### Training Process
cd src
bash generate_reduce_GAN.sh  $iteration $gan_config $overall_prefix || exit 1
