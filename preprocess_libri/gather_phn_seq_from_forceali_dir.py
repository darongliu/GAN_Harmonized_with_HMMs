import os
import argparse
import _pickle as pk

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir'  , type=str, default='', help='')
    parser.add_argument('--output_path', type=str, default='', help='')
    parser.add_argument('--output_meta_path', type=str, default='', help='')
    return parser

def process_phn(phn):
    if phn == 'spn':
        return 'SPN'
    elif phn == 'sil':
        return 'SIL'
    elif phn == 'sp':
        return 'SIL'
    else:
        return phn

def get_phn_from_line(line):
    # get phoneme from line
    # we can further process the phn
    all_elements = line.split()
    assert(len(all_elements) == 3)

    return process_phn(all_elements[-1])

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    all_output_lines = []
    count = 0
    all_name = []
    for f_name in os.listdir(args.input_dir):
        if '.phn' in f_name:
            all_name.append(os.path.join(args.input_dir, f_name))
            if count % 200 == 0:
                print(f'process {count}')
            count += 1

            new_line = ''
            for line in open(os.path.join(args.input_dir, f_name)).read().splitlines():
                new_line += get_phn_from_line(line)
                new_line += ' '
            all_output_lines.append(new_line.strip())

    with open(args.output_path, 'w') as f:
        for line in all_output_lines:
            f.write(line)
            f.write('\n')
        print(f'output total {len(all_output_lines)} lines')
    
    pk.dump(all_name, open(args.output_meta_path, 'wb'))
