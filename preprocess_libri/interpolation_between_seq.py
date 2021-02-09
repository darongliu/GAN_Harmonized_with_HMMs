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
    d = np.zeros([len(orc_bnds)+1,len(uns_bnds)+1], dtype=np.int32)
    all_idx = np.zeros([len(orc_bnds)+1,len(uns_bnds)+1], dtype=np.int32)

    for i in range(len(orc_bnds)+1):
        d[i][0] = i
        all_idx[i][0] = 1
    for i in range(len(uns_bnds)+1):
        d[0][i] = i
        all_idx[0][i] = 0

    for i in range(1, len(orc_bnds)+1):
        for j in range(1, len(uns_bnds)+1):
            if abs(orc_bnds[i-1] - uns_bnds[j-1]) <= tolerant_error:
                m, index = get_min_and_index([d[i][j-1]+1, d[i-1][j]+1, d[i-1][j-1]], [uns_bnds[j-1], orc_bnds[i-1], orc_bnds[i-1]])
                d[i][j] = m
                all_idx[i][j] = index
            else:
                m, index = get_min_and_index([d[i][j-1]+1, d[i-1][j]+1], [uns_bnds[j-1], orc_bnds[i-1]])
                d[i][j] = m
                all_idx[i][j] = index

    # reverse
    new_boundary = []
    now = (len(orc_bnds), len(uns_bnds))
    while True:
        i, j = now
        if all_idx[i][j] == 0: # insersion error
            if sample_interpolation(orc_weight):
                pass
            else:
                # print(new_boundary, uns_bnds[j-1])
                assert len(new_boundary)==0 or (uns_bnds[j-1]) < new_boundary[-1]
                new_boundary.append(uns_bnds[j-1])
            now = (i, j-1)
        elif all_idx[i][j] == 1: # deletion error
            if sample_interpolation(orc_weight):
                # print(new_boundary, orc_bnds[i-1])
                assert len(new_boundary)==0 or (orc_bnds[i-1]) < new_boundary[-1]
                new_boundary.append(orc_bnds[i-1])
            now = (i-1, j)
        elif all_idx[i][j] == 2:
            # print(new_boundary, orc_bnds[i-1])
            assert len(new_boundary)==0 or (orc_bnds[i-1]) < new_boundary[-1]
            new_boundary.append(orc_bnds[i-1])
            now = (i-1, j-1)
        else:
            raise(NotImplementedError)

        if now[0] == now[1] == 0:
            break
    
    return list(reversed(new_boundary))

def get_min_and_index(l, add_bnd):
    index = 0
    m = np.inf
    b = 0
    for i in range(len(l)):
        if l[i] < m:
            index = i
            m = l[i]
            b = add_bnd[i]
        elif l[i] == m and add_bnd[i] >= b:
            index = i
            m = l[i]
            b = add_bnd[i]            

    return m, index

def sample_interpolation(true_prob):
    assert 0 <= true_prob
    assert 1 >= true_prob
    return random.uniform(0,1) < true_prob

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

        inter_uns = interpolation(orc[1:-1], uns[1:-1], args.tolerant_error, args.orc_weight)
        inter_uns = [orc[0]] + inter_uns + [orc[-1]]
        all_new_uns_bnds.append(inter_uns)
        # print('new', inter_uns)
        # assert(len(uns)<=len(inter_uns)<=len(orc) or len(uns)>=len(inter_uns)>=len(orc))


    all_new_uns_bnds = np.array(all_new_uns_bnds, dtype=object)
    pk.dump(all_new_uns_bnds, open(args.target_interpolate_uns_bnd_path, 'wb'))

    # stat
    orc_len = np.sum([len(x) for x in all_orc_bnds])
    uns_len = np.sum([len(x) for x in all_uns_bnds])
    new_uns_len = np.sum([len(x) for x in all_new_uns_bnds])
    print('original uns ratio', uns_len/orc_len)
    print('new uns ratio', new_uns_len/orc_len)