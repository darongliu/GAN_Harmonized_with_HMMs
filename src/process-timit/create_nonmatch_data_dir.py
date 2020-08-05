import os
import pickle as pk
import shutil
import argparse

def copy_meta(src_path, tgt_path, num):
    a = pk.load(open(src_path, 'rb'))
    a['prefix'] = a['prefix'][:num]
    pk.dump(a, open(tgt_path, 'wb'))

def copy_data(src_path, tgt_path, num):
    a = pk.load(open(src_path, 'rb'))
    pk.dump(a[:num], open(tgt_path, 'wb'))

def create_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',          type=str, default='', help='')
    parser.add_argument('--nonmatch_data_dir', type=str, default='', help='')
    parser.add_argument('--subset_num',        type=int, default=3000, help='')
    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    print('create nonmatch dir')
    create_dir(args.nonmatch_data_dir)

    # copy wfst_data
    print('copy wfst_data dir')
    src_wfst_dir = os.path.join(args.data_dir, 'wfst_data')
    tgt_wfst_dir = os.path.join(args.nonmatch_data_dir, 'wfst_data')
    shutil.copytree(src_wfst_dir, tgt_wfst_dir)

    # copy phn map
    print('copy phn map file')
    src_phn_map = os.path.join(args.data_dir, 'phones.60-48-39.map.txt')
    tgt_phn_map = os.path.join(args.nonmatch_data_dir, 'phones.60-48-39.map.txt')
    shutil.copy(src_phn_map, tgt_phn_map)

    # create timit_for_GAN
    print('create timit_for_GAN dir')
    src_gan_dir = os.path.join(args.data_dir, 'timit_for_GAN')
    tgt_gan_dir = os.path.join(args.nonmatch_data_dir, 'timit_for_GAN')
    create_dir(tgt_gan_dir)

    # copy text dir
    print('copy text dir')
    src_text_dir = os.path.join(src_gan_dir, 'text')
    tgt_text_dir = os.path.join(tgt_gan_dir, 'text')
    shutil.copytree(src_text_dir, tgt_text_dir)

    # create audio dir
    print('create audio dir')
    src_audio_dir = os.path.join(src_gan_dir, 'audio')
    tgt_audio_dir = os.path.join(tgt_gan_dir, 'audio')
    create_dir(tgt_audio_dir)

    # get test and train name 
    print('copy test file')
    all_file_name = os.listdir(src_audio_dir)
    all_test_name = []
    for name in all_file_name:
        if name[:10] == 'timit-test':
            all_test_name.append(name)

    # copy test file
    for name in all_test_name:
        src_file = os.path.join(src_audio_dir, name)
        tgt_file = os.path.join(tgt_audio_dir, name)
        shutil.copy(src_file, tgt_file)

    # copy train file
    print('copy and down sample train file')
    train_meta_name = ['timit-train-meta.pkl']
    all_train_name  = ['timit-train-gas.pkl', 'timit-train-length.pkl', 'timit-train-mfcc-nor.pkl', 
                       'timit-train-orc1-bnd.pkl', 'timit-train-phn.pkl', 'timit-train-uns1-bnd.pkl']
    
    for name in train_meta_name:
        src_file = os.path.join(src_audio_dir, name)
        tgt_file = os.path.join(tgt_audio_dir, name)
        copy_meta(src_file, tgt_file, args.subset_num)

    for name in all_train_name:
        src_file = os.path.join(src_audio_dir, name)
        tgt_file = os.path.join(tgt_audio_dir, name)
        copy_data(src_file, tgt_file, args.subset_num)


  
