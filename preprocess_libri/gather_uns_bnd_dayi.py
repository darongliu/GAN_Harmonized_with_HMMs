import os
import argparse
import _pickle as pk
import numpy as np
from tqdm import tqdm

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir_path'  , type=str, default='', help='')

    parser.add_argument('--target_meta_path', type=str, default='', help='')
    parser.add_argument('--target_mfcc_path', type=str, default='', help='')

    parser.add_argument('--target_uns_bnd_path'  , type=str, default='', help='')
    return parser

def load(path):
    return pk.load(open(path, 'rb'))

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    meta = load(args.target_meta_path)
    mfcc = load(args.target_mfcc_path)
    assert(len(meta['prefix']) == len(mfcc))

    all_bnds = []
    for i in tqdm(range(len(meta['prefix']))):
        prefix = meta['prefix'][i]
        m = mfcc[i]

        bnds = []

        with open(os.path.join(args.source_dir_path, prefix+'.phn')) as f:
            all_lines = f.read().splitlines()
            for line in all_lines:
                elements = line.split()
                assert (len(elements) == 2)
                idx = int(float(elements[0])*100.)
                bnds.append(idx)
            assert((len(m)-1)>=idx)
            if idx != (len(m)-1):
                bnds.append(len(m)-1)
            assert(bnds[0]==0)
            all_bnds.append(bnds)

    all_bnds = np.array(all_bnds, dtype=object)

    pk.dump(all_bnds, open(args.target_uns_bnd_path, 'wb'))

