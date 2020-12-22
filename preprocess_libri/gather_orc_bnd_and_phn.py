import os
import argparse
import _pickle as pk
import numpy as np
from tqdm import tqdm 
from gather_phn_seq_from_forceali_dir import process_phn

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir_path'  , type=str, default='', help='')

    parser.add_argument('--target_meta_path', type=str, default='', help='')
    parser.add_argument('--target_mfcc_path', type=str, default='', help='')

    parser.add_argument('--target_orc_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--target_phn_path'  , type=str, default='', help='')
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
    all_phns = []
    for i in tqdm(range(len(meta['prefix']))):
        prefix = meta['prefix'][i]
        m = mfcc[i]

        bnds = []
        phns = []

        with open(os.path.join(args.source_dir_path, prefix+'.phn')) as f:
            all_lines = f.read().splitlines()
            for line in all_lines:
                elements = line.split()
                assert (len(elements) == 3)
                bnds.append(int(elements[0]))
                phns.append(process_phn(elements[-1]))
            assert(abs(int(elements[1])-len(m)) <= 1)
            assert((len(m)-1)>=int(elements[0]))
            bnds.append(len(m)-1)
            all_bnds.append(bnds)
            all_phns.append(phns)

    all_bnds = np.array(all_bnds, dtype=object)
    all_phns = np.array(all_phns, dtype=object)

    pk.dump(all_bnds, open(args.target_orc_bnd_path, 'wb'))
    pk.dump(all_phns, open(args.target_phn_path, 'wb'))

