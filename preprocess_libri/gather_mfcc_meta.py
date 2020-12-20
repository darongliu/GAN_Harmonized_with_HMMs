import os
import argparse
import shutil
import numpy as np
import _pickle as pk

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_meta_path'  , type=str, default='', help='')
    parser.add_argument('--pre_mfcc_path', type=str, default='', help='')
    parser.add_argument('--target_meta_path', type=str, default='', help='')
    parser.add_argument('--target_mfcc_path', type=str, default='', help='')
    return parser

def load(path):
    return pk.load(open(path, 'rb'))

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    # copy meta file
    shutil.copyfile(args.pre_meta_path, args.target_meta_path)
    # load mfcc file
    mfcc = load(args.pre_mfcc_path)
    pk.dump(np.array(mfcc, dtype=object), open(args.target_mfcc_path, 'wb'))

    
