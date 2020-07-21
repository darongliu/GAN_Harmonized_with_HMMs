# !/bin/bash
                                                                                                                           2 
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
cd src

### train HMM and get new boundaries
bash train_HMM.sh $iteration $overall_prefix || exit 1

cd $current_path

