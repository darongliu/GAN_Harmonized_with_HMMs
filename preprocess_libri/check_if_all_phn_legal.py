import os
import argparse
import _pickle as pk

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_text_path'  , type=str, default='', help='')
    parser.add_argument('--phn_mapping_path', type=str, default='', help='')
    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    # read mapping file
    all_phn = []
    for line in open(args.phn_mapping_path, 'r').read().splitlines():
        all_phn.append(line.split()[1])
    all_phn = set(all_phn)

    i = 0
    for line in open(args.lm_text_path, 'r').read().splitlines():
        if i % 200 == 0:
            print(i)
        i += 1
        for p in line.split():
            try:
                assert(p in all_phn)
            except:
                print(p)
                import sys
                sys.exit()

    