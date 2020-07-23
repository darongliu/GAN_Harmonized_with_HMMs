# !/bin/bash                                                                                                                       2 
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
clean_type=$3 # all, wfst, hmm

gan_prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_gan
hmm_prefix=${overall_prefix}${bnd_type}_iter${iteration}_${setting}_hmm

GAN_PATH=$DATA_PATH/save/${gan_prefix}
HMM_PATH=$DATA_PATH/save/${hmm_prefix}

if [ "$clean_type" = "wfst" ]
then
    rm -rf ${GAN_PATH}/decode_test
    rm -rf ${GAN_PATH}/decode_train
    rm -rf ${GAN_PATH}/posterior
    rm -rf ${GAN_PATH}/train.pkl
    rm -rf ${GAN_PATH}/test.pkl
    rm -rf ${GAN_PATH}/train_output.txt
    rm -rf ${GAN_PATH}/test_output.txt
    rm -rf ${DATA_PATH}/result/${gan_prefix}.log
elif [ "$clean_type" = "hmm" ]
then
    rm -rf $HMM_PATH
    rm -rf ${DATA_PATH}/result/${hmm_prefix}.log  
    rm -rf ${DATA_PATH}/timit_for_HMM
elif [ "$clean_type" = "all" ]
then
    rm -rf $DATA_PATH/save
    rm -rf ${DATA_PATH}/timit_for_HMM
    rm -rf ${DATA_PATH}/result/${hmm_prefix}.log  
    rm -rf ${DATA_PATH}/result/${gan_prefix}.log
else
    true
fi

# all
# wfst decoder # decode from 
# hmm 