# !/bin/bash

#Path
export ROOT_DIR=/home/darong/darong/GAN_Harmonized_with_HMMs #abs. path of this github repository
export TIMIT_DIR=/home/darong/other_storage/librispeech_before_process/darong #abs. path of your timit dataset
export DATA_PATH=/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data # contain timit_for_gan

#Boundaries type: orc / uns
export bnd_type=orc

#Setting: match / nonmatch
export setting=match

#Number of jobs
export jobs=8
