all_pre_dir='/home/darong/other_storage/librispeech_before_process/timit_for_gan_v1'
all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'

# gather libri100
python gather_mfcc_meta.py --pre_meta_path $all_pre_dir/librispeech-train-clean-100-meta.pkl --pre_mfcc_path $all_pre_dir/librispeech-train-clean-100-mfcc.pkl --target_meta_path $all_target_dir/timit-train-meta.pkl --target_mfcc_path $all_target_dir/timit-train-mfcc.pkl

# gather dev
python gather_mfcc_meta.py --pre_meta_path $all_pre_dir/librispeech-dev-clean-meta.pkl --pre_mfcc_path $all_pre_dir/librispeech-dev-clean-mfcc.pkl --target_meta_path $all_target_dir/timit-dev-meta.pkl --target_mfcc_path $all_target_dir/timit-dev-mfcc.pkl

# gather test
python gather_mfcc_meta.py --pre_meta_path $all_pre_dir/librispeech-test-clean-meta.pkl --pre_mfcc_path $all_pre_dir/librispeech-test-clean-mfcc.pkl --target_meta_path $all_target_dir/timit-test-meta.pkl --target_mfcc_path $all_target_dir/timit-test-mfcc.pkl