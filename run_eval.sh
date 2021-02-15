# !/bin/bash

### Setting for Kaldi
. ./cmd.sh
. ./path.sh


### Experimental Setting
. ./config.sh

### overall prefix
gan_config=$(readlink -m $1)
overall_prefix=$2
ckpt=$3
iteration=$4

if [ $bnd_type == orc ]; then
  iteration=1
fi

cd src

### train GAN model
bash train_GAN_eval.sh  $iteration $gan_config $overall_prefix $ckpt || exit 1