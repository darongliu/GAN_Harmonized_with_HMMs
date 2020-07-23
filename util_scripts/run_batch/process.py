import argparse
import yaml

parser = argparse.ArgumentParser(description='modify yaml')
parser.add_argument('--config', default='./src/GAN-based-model/config.yaml', type=str, metavar='PATH',
                    help='base config file')
parser.add_argument('--gan_gumbel', default='', type=str)
#parser.add_argument('--prior', type=float, help='the dirichlet prior')
parser.add_argument('--intra_gumbel', default='', type=str)
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='output config file')


args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
if args.gan_gumbel != '':
    config['gan_gumbel'] = args.gan_gumbel
if args.intra_gumbel != '':
    config['intra_gumbel'] = args.intra_gumbel
# config['training']['neighbor_kl']['max_beta'] = args.neighbor_kl
# if args.use_lstm != '':
#     if args.use_lstm == 'True' or args.use_lstm == 'true':
#         config['model']['encoder']['use_lstm'] = True
#         config['model']['decoder']['use_lstm'] = True
#     else:
#         config['model']['encoder']['use_lstm'] = False
#         config['model']['decoder']['use_lstm'] = False
# #config['training']['prior_alpha'] = args.prior
# config['training']['neighbor_kl']['one_side_kernel_size'] = args.window

yaml.dump(config, open(args.output, 'w'))
