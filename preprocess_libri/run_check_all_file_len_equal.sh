all_source_dir='/home/darong/other_storage/librispeech_before_process/darong'
all_pre_dir='/home/darong/other_storage/librispeech_before_process/timit_for_gan_v1'

# check libri100
python check_all_file_len_equal.py --source_dir_path $all_source_dir/train-clean-100 --pre_meta_path $all_pre_dir/librispeech-train-clean-100-meta.pkl --pre_mfcc_path $all_pre_dir/librispeech-train-clean-100-mfcc.pkl

# check dev
python check_all_file_len_equal.py --source_dir_path $all_source_dir/dev-clean --pre_meta_path $all_pre_dir/librispeech-dev-clean-meta.pkl --pre_mfcc_path $all_pre_dir/librispeech-dev-clean-mfcc.pkl

# check test
python check_all_file_len_equal.py --source_dir_path $all_source_dir/test-clean --pre_meta_path $all_pre_dir/librispeech-test-clean-meta.pkl --pre_mfcc_path $all_pre_dir/librispeech-test-clean-mfcc.pkl