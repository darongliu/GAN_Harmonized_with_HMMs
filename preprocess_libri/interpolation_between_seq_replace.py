import random
import os
import argparse
import _pickle as pk
import numpy as np
from tqdm import tqdm
import sys

# only for collecting .phn file which is the result of montreal

def load(path):
    return pk.load(open(path, 'rb'))

def interpolation(orc_bnds, uns_bnds, tolerant_error, orc_weight):
    total_len = uns_bnds[-1]
    replace_len = int(total_len*orc_weight)
    start_idx = random.randint(0, total_len-replace_len)
    new_bnds = []
    # print('start: ', start_idx, 'end: ', start_idx + replace_len, 'len: ', total_len)
    for bnds in uns_bnds:
        if bnds < start_idx - tolerant_error:
            new_bnds.append(bnds)
    for bnds in orc_bnds:
        if bnds >= start_idx and bnds <= start_idx+replace_len:
            new_bnds.append(bnds)
    for bnds in uns_bnds:
        if bnds > start_idx + replace_len + tolerant_error:
            new_bnds.append(bnds)
    if new_bnds[0] != orc_bnds[0]:
        new_bnds[0] = orc_bnds[0]
    if new_bnds[-1] != orc_bnds[-1]:
        new_bnds[-1] = orc_bnds[-1]
    return new_bnds

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_uns_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--target_orc_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--tolerant_error'       , type=int, default=2, help='')
    parser.add_argument('--orc_weight'           , type=float, default=0.5, help='')

    parser.add_argument('--target_interpolate_uns_bnd_path'  , type=str, default='', help='')

    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    all_uns_bnds = load(args.target_uns_bnd_path)
    all_orc_bnds = load(args.target_orc_bnd_path)

    assert(len(all_uns_bnds) == len(all_orc_bnds))

    all_new_uns_bnds = []
    for seq_idx in tqdm(range(len(all_uns_bnds))):
        uns = all_uns_bnds[seq_idx]
        orc = all_orc_bnds[seq_idx]
        # print('idx', seq_idx)
        assert(uns[0]==orc[0]==0)
        assert(uns[-1]==orc[-1])

        inter_uns = interpolation(orc, uns, args.tolerant_error, args.orc_weight)
        all_new_uns_bnds.append(inter_uns)
        # print('new', inter_uns)
        assert(uns[0]==orc[0]==0==inter_uns[0])
        assert(uns[-1]==orc[-1]==inter_uns[-1])
        # print(len(uns), len(inter_uns), len(orc))
        # assert(len(uns)<=len(inter_uns)<=len(orc) or len(uns)>=len(inter_uns)>=len(orc))


    all_new_uns_bnds = np.array(all_new_uns_bnds, dtype=object)
    pk.dump(all_new_uns_bnds, open(args.target_interpolate_uns_bnd_path, 'wb'))

    # stat
    orc_len = np.sum([len(x) for x in all_orc_bnds])
    uns_len = np.sum([len(x) for x in all_uns_bnds])
    new_uns_len = np.sum([len(x) for x in all_new_uns_bnds])
    print('original uns ratio', uns_len/orc_len)
    print('new uns ratio', new_uns_len/orc_len)