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

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    all_output_lines = []
    count = 0
    all_name = []
    for f_name in os.listdir(args.input_dir):
        if '.txt' in f_name:
            all_name.append(os.path.join(args.input_dir, f_name))
            if count % 200 == 0:
                print(f'process {count}')
            count += 1

            line = open(os.path.join(args.input_dir, f_name)).read().splitlines()
            assert(len(line)==1)
            new_line = ' '.join(line.split()[2:])
            all_output_lines.append(new_line.strip())

    with open(args.output_path, 'w') as f:
        for line in all_output_lines:
            f.write(line)
            f.write('\n')
        print(f'output total {len(all_output_lines)} lines')
    
    pk.dump(all_name, open(args.output_meta_path, 'wb'))
