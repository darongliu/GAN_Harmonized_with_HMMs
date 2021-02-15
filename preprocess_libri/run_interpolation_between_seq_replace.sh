all_target_dir='/home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/audio'
set e
for orc_weight in 0.25  0.75
do
    echo 'orc weight: '$orc_weight

    # libri100
    python interpolation_between_seq_replace.py \
        --target_uns_bnd_path $all_target_dir/timit-train-uns1-bnd.pkl \
        --target_orc_bnd_path $all_target_dir/timit-train-orc1-bnd.pkl \
        --orc_weight $orc_weight \
        --target_interpolate_uns_bnd_path $all_target_dir/timit-train-inter-replace-$orc_weight-uns1-bnd.pkl

    # dev
    python interpolation_between_seq_replace.py  \
        --target_uns_bnd_path $all_target_dir/timit-dev-uns1-bnd.pkl \
        --target_orc_bnd_path $all_target_dir/timit-dev-orc1-bnd.pkl \
        --orc_weight $orc_weight \
        --target_interpolate_uns_bnd_path $all_target_dir/timit-dev-inter-replace-$orc_weight-uns1-bnd.pkl

    # test
    python interpolation_between_seq_replace.py \
        --target_uns_bnd_path $all_target_dir/timit-test-uns1-bnd.pkl \
        --target_orc_bnd_path $all_target_dir/timit-test-orc1-bnd.pkl \
        --orc_weight $orc_weight \
        --target_interpolate_uns_bnd_path $all_target_dir/timit-test-inter-replace-$orc_weight-uns1-bnd.pkl
done