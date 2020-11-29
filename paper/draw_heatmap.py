import torch
import numpy as np
import argparse
import _pickle as pk

def read_phn_map(phn_map_path):
    phn_mapping_48_39 = {}
    phn_mapping_39_48 = {}
    with open(phn_map_path, 'r') as f:
        for line in f:
            if line.strip() != "":
                p60, p48, p39 = line.split()
                phn_mapping_48_39[p48] = p39
                if p39 in phn_mapping_39_48.keys():
                    phn_mapping_39_48[p39].add(p48)
                else:
                    phn_mapping_39_48[p39] = set([p48])

    all_phn_48 = list(phn_mapping_48_39.keys())
    all_phn_39 = list(phn_mapping_39_48.keys())
    assert(len(all_phn_48) == 48)
    assert(len(all_phn_39) == 39)

    return phn_mapping_48_39, phn_mapping_39_48, all_phn_48, all_phn_39
    '''
    all_phn = list(phn_mapping.keys())
    assert(len(all_phn) == 48)
    self.phn_size = len(all_phn)
    self.phn2idx        = dict(zip(all_phn, range(len(all_phn))))
    self.idx2phn        = dict(zip(range(len(all_phn)), all_phn))
    self.phn_mapping    = dict([(i, phn_mapping[phn]) for i, phn in enumerate(all_phn)])
    self.sil_idx = self.phn2idx['sil']
    '''

def read_pickle(path):
    return pk.load(open(path,'rb'))

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m'   , '--phn_map_path', type=str, default='', help='')
    parser.add_argument('-post', '--posterior_path'  , type=str, default='', help='')
    parser.add_argument('-ob'  , '--oracle_boundary_path' , type=str, default='', help='')
    parser.add_argument('-op'  , '--oracle_phn_path' , type=str, default='', help='')


    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    phn_mapping_48_39, phn_mapping_39_48, all_phn_48, all_phn_39 = read_phn_map(args.phn_map_path)
    phn_2_idx_48 = dict(zip(all_phn_48, range(len(all_phn_48))))

    all_post = read_pickle(args.posterior_path)
    all_ob = read_pickle(args.oracle_boundary_path)
    all_op = read_pickle(args.oracle_phn_path)

    # create dict
    all_48_accu = {} # accumulate
    all_48_count = {}
    for p48 in all_phn_48:
        all_48_accu[p48] = 0
        all_48_count[p48] = 0

    assert (len(all_post) == len(all_ob) == len(all_op))

    for post, ob, op in zip(all_post, all_ob, all_op):
        assert len(ob) == len(op) + 1
        for prev_b, b, p in zip(ob[:-1], ob[1:], op):
            all_48_accu[p] += post[prev_b:b].sum(0)
            all_48_count[p] += (b-prev_b)

    all_39_count = []
    for p39 in all_phn_39:
        total = 0
        for p48 in phn_mapping_39_48[p39]:
            total += all_48_count[p48]
        all_39_count.append(total)
    # ranking
    zipped_lists = zip(all_39_count, all_phn_39)
    sorted_zipped_lists = sorted(zipped_lists, reverse=True)
    new_all_phn_39 = [p39 for _, p39 in sorted_zipped_lists]

    # new
    new_all_39_post = []
    for p39 in new_all_phn_39:
        new_accu = 0
        new_count = 0
        for p48 in phn_mapping_39_48[p39]:
            new_accu += all_48_accu[p48]
            new_count += all_48_count[p48]
        new_post = new_accu/new_count

        # compute
        new_post_39 = np.zeros(39)
        for i, p39 in enumerate(new_all_phn_39):
            for p48 in phn_mapping_39_48[p39]:
                new_post_39[i] += new_post[phn_2_idx_48[p48]]

        new_all_39_post.append(new_post_39)
    new_all_39_post = np.stack(new_all_39_post, 0)

    # draw
    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    ax = sns.heatmap(new_all_39_post, xticklabels=new_all_phn_39,yticklabels=new_all_phn_39, cmap="YlGnBu")
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.tight_layout()
    plt.savefig('./heatmap.png')
    #plt.show()
    
    




