#python gather_phn_seq_from_forceali_dir.py --input_dir /home/darong/other_storage/librispeech_before_process/darong/dev-clean --output_path /home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/text/dev_lm.48
python gather_phn_seq_from_forceali_dir.py --input_dir /home/darong/other_storage/librispeech_before_process/darong/train-clean-100 --output_path /home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/text/match_lm.48 --output_meta_path /home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/text/match_meta.pkl
python gather_phn_seq_from_forceali_dir.py --input_dir /home/darong/other_storage/librispeech_before_process/darong/train-clean-360 --output_path /home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/text/nonmatch_lm.48 --output_meta_path /home/darong/frequent_data/GAN_Harmonized_with_HMMs_librispeech/data/timit_for_GAN/text/nonmatch_meta.pkl