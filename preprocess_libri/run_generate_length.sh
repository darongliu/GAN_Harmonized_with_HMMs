all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'

# normalize libri100
python generate_length.py --target_mfcc_nor_path $all_target_dir/timit-train-mfcc-nor.pkl --target_length_path $all_target_dir/timit-train-length.pkl

# normalize dev
python generate_length.py --target_mfcc_nor_path $all_target_dir/timit-dev-mfcc-nor.pkl --target_length_path $all_target_dir/timit-dev-length.pkl

# normalize test
python generate_length.py --target_mfcc_nor_path $all_target_dir/timit-test-mfcc-nor.pkl --target_length_path $all_target_dir/timit-test-length.pkl