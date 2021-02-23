from scipy.stats import entropy
import argparse
import _pickle as pk
import numpy as np

def read_pickle(path):
    return pk.load(open(path,'rb'))

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p' , '--posterior_path' , type=str, default='', help='')
    parser.add_argument('-l', '--length_path', type=str, default='', help='')

    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    posterior = read_pickle(args.posterior_path)
    length = read_pickle(args.length_path)

    assert(len(posterior) == len(length))

    total_entropy = 0
    count = 0
    for i in range(len(posterior)):
        for j in range(length[i]):
            e = entropy(posterior[i][j])
            if np.isnan(e):
                print('isnan', posterior[i][j], i, j)
                continue
            count += 1
            total_entropy += e
    print('entropy:', total_entropy/count)
