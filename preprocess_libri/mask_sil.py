import os
import argparse
import _pickle as pk
import numpy as np
from tqdm import tqdm

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_uns_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--target_orc_bnd_path'  , type=str, default='', help='')
    parser.add_argument('--target_phn_path'  , type=str, default='', help='')

    parser.add_argument('--target_masked_uns_bnd_path'  , type=str, default='', help='')

    return parser

def load(path):
    return pk.load(open(path, 'rb'))

if __name__ == '__main__':
    TOLERANT=5
    parser = addParser()
    args = parser.parse_args()

    a = load(args.target_uns_bnd_path)
    b = load(args.target_orc_bnd_path)
    all_phn = load(args.target_phn_path)

    new_a = []

    assert(len(a) == len(b) == len(all_phn))
    for seq_idx in tqdm(range(len(a))):
        uns = a[seq_idx]
        orc = b[seq_idx]
        phn = all_phn[seq_idx]
        try:
            assert uns[-1]==orc[-1]
        except:
            print('seq_idx', seq_idx)
            assert uns[-1]==orc[-1]
        new_uns = []
        uns_idx = 0  # the idx after the last silence # not including overlapping
        for i in range(len(orc)-1):
            if phn[i] != 'sil':
                continue
            sil_start = orc[i]
            sil_end = orc[i+1]
            #print('seq idx', i)
            #print('sil_start', sil_start, sil_end)
            # get the boundary b4 sthubart of silence
            prev_uns_idx = uns_idx
            while True:
                if uns[uns_idx] >= sil_start-TOLERANT:
                    break
                else:
                    uns_idx += 1
            new_uns += uns[prev_uns_idx:uns_idx]
            new_uns.append(sil_start)
            #print('right b4', uns_idx)
                    # get the boundary after end
            while True:
                if uns_idx == len(uns) or uns[uns_idx] > sil_end+TOLERANT:
                    break
                else:
                    uns_idx += 1
            new_uns.append(sil_end)
            #print('right after', uns_idx)
        new_uns += uns[uns_idx:]
        assert(new_uns[-1]==orc[-1]==uns[-1])
        new_a.append(new_uns)

    all_bnds = np.array(new_a, dtype=object)
    pk.dump(new_a, open(args.target_masked_uns_bnd_path, 'wb'))

    # stat
    orc_len = np.sum([len(x) for x in b])
    uns_len = np.sum([len(x) for x in a])
    new_uns_len = np.sum([len(x) for x in new_a])
    print('original uns ratio', uns_len/orc_len)
    print('new uns ratio', new_uns_len/orc_len)