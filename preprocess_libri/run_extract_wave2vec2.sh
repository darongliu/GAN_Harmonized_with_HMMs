all_origin_dir='/home/darong/other_storage/librispeech_before_process/darong'
all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'

# libri100
python extract_wave2vec2.py --source_dir_path $all_origin_dir/train-clean-100 \
                            --target_meta_path $all_target_dir/timit-train-meta.pkl \
                            --target_phn_path $all_target_dir/timit-train-phn.pkl \
                            --target_wav2vec2_path $all_target_dir/timit-train-wave2vec2.pkl \
                            --target_wav2vec2_orc_bnd_path $all_target_dir/timit-train-wave2vec2-orc1-bnd.pkl 2> >(tee libri100.err)

# # dev
# python extract_wave2vec2.py --source_dir_path $all_origin_dir/dev-clean \
#                             --target_meta_path $all_target_dir/timit-dev-meta.pkl \
#                             --target_phn_path $all_target_dir/timit-dev-phn.pkl \
#                             --target_wav2vec2_path $all_target_dir/timit-dev-wave2vec2.pkl \
#                             --target_wav2vec2_orc_bnd_path $all_target_dir/timit-dev-wave2vec2-orc1-bnd.pkl 2> >(tee dev.err)

# test
# python extract_wave2vec2.py --source_dir_path $all_origin_dir/test-clean \
#                             --target_meta_path $all_target_dir/timit-test-meta.pkl \
#                             --target_phn_path $all_target_dir/timit-test-phn.pkl \
#                             --target_wav2vec2_path $all_target_dir/timit-test-wave2vec2.pkl \
#                             --target_wav2vec2_orc_bnd_path $all_target_dir/timit-test-wave2vec2-orc1-bnd.pkl 2> >(tee test.err)