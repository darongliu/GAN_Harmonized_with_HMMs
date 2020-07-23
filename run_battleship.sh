# !/bin/bash

### Setting for Kaldi
. ./cmd.sh
. ./path.sh


### Experimental Setting 
. ./config.sh

### overall prefix
gan_config=$(readlink -m $1)
overall_prefix=$2

if [ $bnd_type == orc ]; then
  total_iter=1
else
  total_iter=3
fi

### Preprocess TIMIT
# bash preprocess.sh


### Training Process
cd src

for iteration in $(seq 1 $total_iter); do
  ### train GAN model
  hrun -G -c 4 -m 32 bash train_GAN.sh  $iteration $gan_config $overall_prefix || exit 1

  ### wfst decoder
  hrun -c 8 -m 32 bash train_wfst.sh $iteration $overall_prefix|| exit 1

  ### train HMM and get new boundaries
  hrun -c 8 -m 32 bash train_HMM.sh $iteration $overall_prefix || exit 1
done
