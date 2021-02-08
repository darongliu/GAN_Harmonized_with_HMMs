import random
import os
import argparse
import _pickle as pk
import numpy as np
from tqdm import tqdm

# only for collecting .phn file which is the result of montreal

def load(path):
    return pk.load(open(path, 'rb'))

def interpolation(orc_bnds, uns_bnds, tolerant_error, orc_weight):
    d = np.zeros([len(orc_bnds)+1,len(uns_bnds)+1], dtype=np.int32)
    all_idx = np.zeros([len(orc_bnds)+1,len(uns_bnds)+1], dtype=np.int32)

    for i in range(len(orc_bnds)+1):
        d[i][0] = i
        all_idx[i][0] = 2
    for i in range(len(uns_bnds)+1):
        d[0][i] = i
        all_idx[0][i] = 1

    for i in range(1, len(orc_bnds)+1):
        for j in range(1, len(uns_bnds)+1):
            if abs(orc_bnds[i-1] - uns_bnds[j-1]) <= tolerant_error:
                m, index = get_min_and_index([d[i-1][j-1], d[i][j-1]+1, d[i-1][j]+1])
                d[i][j] = m
                if index == 0:
                    index = 3
                all_idx[i][j] = index
            else:
                m, index = get_min_and_index([d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1])
                d[i][j] = m
                all_idx[i][j] = index

    # reverse
    new_boundary = []
    now = (len(orc_bnds), len(uns_bnds))
    while True:
        i, j = now
        if all_idx[i][j] == 0: # replace
            if sample_interpolation(orc_weight):
                new_boundary.append(orc_bnds[i-1])
            else:
                new_boundary.append(uns_bnds[j-1])
            now = (i-1, j-1)
        elif all_idx[i][j] == 1: # insersion error
            if sample_interpolation(orc_weight):
                pass
            else:
                new_boundary.append(uns_bnds[j-1])
            now = (i, j-1)
        elif all_idx[i][j] == 2: # deletion error
            if sample_interpolation(orc_weight):
                new_boundary.append(orc_bnds[i-1])
            now = (i-1, j)
        elif all_idx[i][j] == 3:
            new_boundary.append(orc_bnds[i-1])
            now = (i-1, j-1)
        else:
            raise(NotImplementedError)
        if i == 0 and j == 0:
            break
    
    return reversed(new_boundary)

def get_min_and_index(l):
    m = min(l)
    index = l.index(m)
    return m, index

def sample_interpolation(true_prob):
    assert 0 <= true_prob
    assert 1 >= true_prob
    return random.uniform(0,1) < true_prob

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_uns_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--target_orc_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--tolerant_error'       , type=float, default='', help='')
    parser.add_argument('--orc_weight'           , type=float, default='', help='')

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

        inter_uns = interpolation(orc_bnds, uns_bnds, args.tolerant_error, args.orc_weight)
        all_new_uns_bnds.append(inter_uns)

    all_new_uns_bnds = np.array(all_new_uns_bnds, dtype=object)
    pk.dump(all_new_uns_bnds, open(args.target_interpolate_uns_bnd_path, 'wb'))