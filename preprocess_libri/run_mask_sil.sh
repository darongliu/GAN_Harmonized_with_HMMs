all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'

# libri100
python mask_sil.py --target_uns_bnd_path $all_target_dir/timit-train-uns1-bnd.pkl \
                   --target_orc_bnd_path $all_target_dir/timit-train-orc1-bnd.pkl \
                   --target_phn_path $all_target_dir/timit-train-phn.pkl \
                   --target_masked_uns_bnd_path $all_target_dir/timit-train-mask-sil-uns1-bnd.pkl

# dev
python mask_sil.py --target_uns_bnd_path $all_target_dir/timit-dev-uns1-bnd.pkl \
                   --target_orc_bnd_path $all_target_dir/timit-dev-orc1-bnd.pkl \
                   --target_phn_path $all_target_dir/timit-dev-phn.pkl \
                   --target_masked_uns_bnd_path $all_target_dir/timit-dev-mask-sil-uns1-bnd.pkl

# test
python mask_sil.py --target_uns_bnd_path $all_target_dir/timit-test-uns1-bnd.pkl \
                   --target_orc_bnd_path $all_target_dir/timit-test-orc1-bnd.pkl \
                   --target_phn_path $all_target_dir/timit-test-phn.pkl \
                   --target_masked_uns_bnd_path $all_target_dir/timit-test-mask-sil-uns1-bnd.pkl