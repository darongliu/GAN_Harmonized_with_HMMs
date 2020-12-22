all_origin_dir='/home/darong/other_storage/librispeech_before_process/darong'
all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'

# libri100
python gather_orc_bnd_and_phn.py --source_dir_path $all_origin_dir/train-clean-100 \
                                 --target_meta_path $all_target_dir/timit-train-meta.pkl \
                                 --target_mfcc_path $all_target_dir/timit-train-mfcc.pkl \
                                 --target_orc_bnd_path $all_target_dir/timit-train-orc1-bnd.pkl \
                                 --target_phn_path $all_target_dir/timit-train-phn.pkl

# dev
python gather_orc_bnd_and_phn.py --source_dir_path $all_origin_dir/dev-clean \
                                 --target_meta_path $all_target_dir/timit-dev-meta.pkl \
                                 --target_mfcc_path $all_target_dir/timit-dev-mfcc.pkl \
                                 --target_orc_bnd_path $all_target_dir/timit-dev-orc1-bnd.pkl \
                                 --target_phn_path $all_target_dir/timit-dev-phn.pkl

# test
python gather_orc_bnd_and_phn.py --source_dir_path $all_origin_dir/test-clean \
                                 --target_meta_path $all_target_dir/timit-test-meta.pkl \
                                 --target_mfcc_path $all_target_dir/timit-test-mfcc.pkl \
                                 --target_orc_bnd_path $all_target_dir/timit-test-orc1-bnd.pkl \
                                 --target_phn_path $all_target_dir/timit-test-phn.pkl