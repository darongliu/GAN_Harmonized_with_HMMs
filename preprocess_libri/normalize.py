import os
import argparse
import shutil
import numpy as np
import _pickle as pk

# only for collecting .phn file which is the result of montreal
def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_train_meta_path', type=str, default='', help='')
    parser.add_argument('--target_mfcc_path', type=str, default='', help='')
    parser.add_argument('--target_mfcc_nor_path', type=str, default='', help='')
    return parser

def load(path):
    return pk.load(open(path, 'rb'))

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    # read train meta file
    train_meta = load(args.target_train_meta_path)
    mean = train_meta['mfcc']['mean']
    std  = train_meta['mfcc']['std']

    # load mfcc
    all_mfcc = load(args.target_mfcc_path)

    # get nor mfcc
    all_mfcc_nor = []
    for i in range(len(all_mfcc)):
        mfcc = all_mfcc[i]
        mfcc_nor = (mfcc - mean)/std
        all_mfcc_nor.append(mfcc_nor)

    # dump mfcc nor file
    pk.dump(np.array(all_mfcc_nor, dtype=object), open(args.target_mfcc_nor_path, 'wb'))
