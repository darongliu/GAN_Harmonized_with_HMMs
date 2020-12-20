all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'
train_meta_path=$all_target_dir/timit-train-meta.pkl

# normalize libri100
python normalize.py --target_train_meta_path $train_meta_path --target_mfcc_path $all_target_dir/timit-train-mfcc.pkl --target_mfcc_nor_path $all_target_dir/timit-train-mfcc-nor.pkl

# normalize dev
python normalize.py --target_train_meta_path $train_meta_path --target_mfcc_path $all_target_dir/timit-dev-mfcc.pkl --target_mfcc_nor_path $all_target_dir/timit-dev-mfcc-nor.pkl

# normalize test
python normalize.py --target_train_meta_path $train_meta_path --target_mfcc_path $all_target_dir/timit-test-mfcc.pkl --target_mfcc_nor_path $all_target_dir/timit-test-mfcc-nor.pkl