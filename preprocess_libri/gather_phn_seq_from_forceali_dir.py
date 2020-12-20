import os
import argparse

# only for collecting .phn file which is the result of montreal

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir'  , type=str, default='', help='')
    parser.add_argument('--output_path', type=str, default='', help='')
    return parser

def get_phn_from_line(line):
    # get phoneme from line
    # we can further process the phn
    all_elements = line.split()
    assert(len(all_elements) == 3)

    return all_elements[-1]

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    all_output_lines = []
    count = 0
    for f_name in os.listdir(args.input_dir):
        if '.phn' in f_name:
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
