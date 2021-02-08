all_origin_dir='/home/darong/other_storage/librispeech_before_process/un-seg/ss'
all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'

# libri100
python gather_uns_bnd_dayi.py --source_dir_path $all_origin_dir/train-clean-100_0.689 \
                              --target_mfcc_path $all_target_dir/timit-train-mfcc.pkl \
                              --target_meta_path $all_target_dir/timit-train-meta.pkl \
                              --target_uns_bnd_path $all_target_dir/timit-train-uns1-bnd.pkl 

# dev
python gather_uns_bnd_dayi.py --source_dir_path $all_origin_dir/dev-clean_0.696 \
                              --target_meta_path $all_target_dir/timit-dev-meta.pkl \
                              --target_mfcc_path $all_target_dir/timit-dev-mfcc.pkl \
                              --target_uns_bnd_path $all_target_dir/timit-dev-uns1-bnd.pkl 

# test
python gather_uns_bnd_dayi.py --source_dir_path $all_origin_dir/test-clean_0.685 \
                              --target_meta_path $all_target_dir/timit-test-meta.pkl \
                              --target_mfcc_path $all_target_dir/timit-test-mfcc.pkl \
                              --target_uns_bnd_path $all_target_dir/timit-test-uns1-bnd.pkl 