# !/bin/bash

### Setting for Kaldi
. ./cmd.sh
. ./path.sh


### Experimental Setting
. ./config.sh

### overall prefix
gan_config=$(readlink -m $1)
overall_prefix=$2
load_ckpt=$3

if [ $bnd_type == orc ]; then
  total_iter=1
else
  total_iter=1
fi

### Preprocess TIMIT
# bash preprocess.sh


### Training Process
cd src

for iteration in $(seq 1 $total_iter); do
  ### train GAN model
  bash eval_GAN.sh  $iteration $gan_config $overall_prefix $load_ckpt || exit 1

  ### wfst decoder
  # bash train_wfst.sh $iteration $overall_prefix|| exit 1

  ### train HMM and get new boundaries
  # bash train_HMM.sh $iteration $overall_prefix || exit 1
done
