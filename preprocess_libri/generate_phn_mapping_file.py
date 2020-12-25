import os
import argparse

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phn_list_path'  , type=str, default='', help='')
    parser.add_argument('--phn_mapping_path', type=str, default='', help='')
    return parser


if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    # read phn_list
    all_new_lines = []
    with open(args.phn_list_path, 'r') as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            phn = line.strip()
            if phn == 'SIL':
                phn = 'sil'
            if phn == 'SPN':
                phn = 'spn'

            reduce_phn = phn
            if phn == 'sil' or phn == 'spn':
                reduce_phn = 'sil'
            if phn[-1] in '012':
                reduce_phn = phn[:-1]

            new_line = phn + ' ' + phn + ' ' + reduce_phn
            all_new_lines.append(new_line)

    with open(args.phn_mapping_path, 'w') as f:
        for line in all_new_lines:
            f.write(line)
            f.write('\n')
