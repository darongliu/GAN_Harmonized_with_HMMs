# !/bin/bash

### Experimental Setting
. ./config_battleship.sh

### overall prefix
gan_config=$(readlink -m $1)
overall_prefix=$2
ckpt=$3
iteration=$4

if [ $bnd_type == orc ]; then
  iteration=1

### Preprocess TIMIT
# bash preprocess.sh

### Training Process
### train GAN model
hrun -G -c $jobs -m 32 bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_GAN_eval.sh '$iteration' '$gan_config' '$overall_prefix' '$ckpt  || exit 1
