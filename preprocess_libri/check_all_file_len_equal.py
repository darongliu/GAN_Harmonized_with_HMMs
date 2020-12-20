import os
import argparse
import _pickle as pk

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir_path'  , type=str, default='', help='')
    parser.add_argument('--pre_meta_path'  , type=str, default='', help='')
    parser.add_argument('--pre_mfcc_path'  , type=str, default='', help='')
    return parser

def load(path):
    return pk.load(open(path, 'rb'))

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    print(args)

    meta = load(args.pre_meta_path)
    meta_len = len(meta['prefix'])
    mfcc_len = len(load(args.pre_mfcc_path))
    assert(meta_len==mfcc_len)

    for prefix in meta['prefix']:
        assert(os.path.isfile(os.path.join(args.source_dir_path, prefix+'.phn')))