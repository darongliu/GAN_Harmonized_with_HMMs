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

### wfst decoder
bash train_wfst.sh $iteration $overall_prefix test || exit 1
