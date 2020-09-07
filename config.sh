# !/bin/bash

#Path
export ROOT_DIR=/home/darong/darong/GAN_Harmonized_with_HMMs #abs. path of this github repository
export TIMIT_DIR=/home/darong/other_storage/data/timit #abs. path of your timit dataset
export DATA_PATH=/home/darong/frequent_data/GAN_Harmonized_with_HMMs/data

#Boundaries type: orc / uns
export bnd_type=orc

#Whether to use posterior-based boundary in each iteration
#Default 0 when an iteration is not specified
#Only take effect when bnd_type=uns
export use_posterior_bnd_iter1=1
export use_posterior_bnd_iter2=0
export use_posterior_bnd_iter3=0

#Setting: match / nonmatch
export setting=match

#Number of jobs
export jobs=8
