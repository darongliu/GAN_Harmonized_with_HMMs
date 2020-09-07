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
  ### decide use_posterior_bnd or not
  if [ $bnd_type == orc ]; then
     use_posterior_bnd=0
  else
    use_posterior_bnd="use_posterior_bnd_iter$iteration"
    use_posterior_bnd=${!use_posterior_bnd}
    if [ -z $use_posterior_bnd ]; then
      use_posterior_bnd=0
    fi
  fi

  ### train GAN model
  hrun -G -c $jobs -m 32 bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_GAN.sh '$iteration' '$gan_config' '$overall_prefix' '$use_posterior_bnd || exit 1

  ### wfst decoder
  hrun -c $jobs -m 64   bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_wfst.sh '$iteration' '$overall_prefix || exit 1

  ### train HMM and get new boundaries
  hrun -c $jobs -m 64   bash -c '. ./cmd.sh; . ./path.sh; . ./config_battleship.sh; cd src; bash train_HMM.sh '$iteration' '$overall_prefix || exit 1
done
