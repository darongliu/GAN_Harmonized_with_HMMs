# !/bin/bash

#Path
export ROOT_DIR=/home/givebirthday/GAN_Harmonized_with_HMMs #abs. path of this github repository
export TIMIT_DIR=/groups/givebirthday/GAN_Harmonized_with_HMMs/timit #abs. path of your timit dataset
export DATA_PATH=/groups/givebirthday/GAN_Harmonized_with_HMMs/nonmatch-data
#export DATA_PATH=/groups/givebirthday/GAN_Harmonized_with_HMMs/data

#Boundaries type: orc / uns
export bnd_type=uns

#Setting: match / nonmatch
export setting=nonmatch
#export setting=match

#Number of jobs
export jobs=16
