import os
import argparse
import shutil
import numpy as np
import _pickle as pk

# only for collecting .phn file which is the result of montreal
def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_mfcc_nor_path', type=str, default='', help='')
    parser.add_argument('--target_length_path', type=str, default='', help='')
    return parser

def load(path):
    return pk.load(open(path, 'rb'))

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    mfcc_nor = load(args.target_mfcc_nor_path)
    length = np.array([len(m) for m in mfcc_nor])

    pk.dump(length, open(args.target_length_path, 'wb'))