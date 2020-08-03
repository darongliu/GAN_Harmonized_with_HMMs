# !/bin/bash

### Experimental Setting 
. ./config_battleship.sh

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
for iteration in $(seq 1 $total_iter); do
  ### train GAN model
  hrun -G -c 8 -m 32 bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_GAN.sh '$iteration' '$gan_config' '$overall_prefix || exit 1

  ### wfst decoder
  hrun -c 16 -m 64   bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_wfst.sh '$iteration' '$overall_prefix || exit 1

  ### train HMM and get new boundaries
  hrun -c 16 -m 64   bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_HMM.sh '$iteration' '$overall_prefix || exit 1
done
