import numpy as np
import os
import json
import argparse
import yaml

from src.data.dataset import PickleDataset
from src.data.dataLoader import DataLoader
from src.models.uns_model import UnsModel
from src.models.sup_model import SupModel


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str, default='train', help='')
    parser.add_argument('--model_type',     type=str, default='uns', help='')
    parser.add_argument('--cuda_id',        type=str, default='0', help='')
    parser.add_argument('--bnd_type',       type=str, default='orc', help='')
    parser.add_argument('--setting',        type=str, default='match', help='')
    parser.add_argument('--iteration',      type=int, default=1, help='')
    parser.add_argument('--aug',            action='store_true', help='')
    parser.add_argument('--data_dir',       type=str, default=f'/home/r06942045/myProjects/GAN_Harmonized_with_HMMs/data') 
    parser.add_argument('--save_dir',       type=str, default=f'/home/r06942045/myProjects/GAN_Harmonized_with_HMMs/data/save/test_model') 
    parser.add_argument('--load_ckpt',      type=str, default=f'ckpt_9000.pth') 
    parser.add_argument('--config',         type=str, default=f'/home/r06942045/myProjects/GAN_Harmonized_with_HMMs/src/GAN-based-model/config.yaml') 
    parser.add_argument('--prefix',         type=str, default=f'', help='used for output fer result') 
    parser.add_argument('--overall_prefix',         type=str, default=f'', help='used for read correct bnd') 
    return parser

def print_bar():
    print ('='*80)

def print_model_parameter(config):
    print ('Model Parameter:')
    print (f'   generator first layer:     {config.gen_hidden_size}')
    print (f'   frame temperature:         {config.frame_temp}')
    print (f'   intra-segment loss ratio:  {config.seg_loss_ratio}')
    print (f'   gradient penalty ratio:    {config.penalty_ratio}')

    print (f'   discriminator model type:     {config.model_type}')
    print (f'   use maxlen:                   {config.use_maxlen}')
    for key in config[config.model_type].keys():
        print (f'{key}:     {config[config.model_type][key]}')
    print_bar()

def print_training_parameter(args, config):
    print ('Training Parameter:')
    print (f'   batch_size:             {config.batch_size}')
    if args.model_type == 'sup':
        print (f'   epoch:                  {config.epoch}')
        print (f'   learning rate(sup):     {config.sup_lr}')
    elif args.model_type == 'uns':
        print (f'   repeat:                 {config.repeat}')
        print (f'   step:                   {config.step}')
        print (f'   learning rate(gen):     {config.gen_lr}')
        print (f'   learning rate(dis):     {config.dis_lr}')
        print (f'   dis iteration:          {config.dis_iter}')
        print (f'   gen iteration:          {config.gen_iter}')
        print (f'   setting:                {args.setting}')
        print (f'   aug:                    {args.aug}')
        print (f'   bound type:             {args.bnd_type}')
        print (f'   data_dir:               {args.data_dir}')
        print (f'   save_dir:               {args.save_dir}')
        print (f'   config_path:            {args.config}')
        print (f'   prefix:                 {args.prefix}')
        print (f'   overall prefix:                 {args.overall_prefix}')
    print_bar()     


if __name__ == "__main__":
    parser = addParser()
    args = parser.parse_args()
    config = read_config(args.config)

    ######################################################################
    # Environment & argument settings
    #
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    if args.iteration == 1:
        train_bnd_path    = f'{args.data_dir}/timit_for_GAN/audio/timit-train-{args.bnd_type}{args.iteration}-bnd.pkl'
    else:
        train_bnd_path    = f'{args.data_dir}/timit_for_GAN/audio/{args.overall_prefix}timit-train-{args.bnd_type}{args.iteration}-bnd.pkl'
    test_bnd_path     = f'{args.data_dir}/timit_for_GAN/audio/timit-test-{args.bnd_type}{args.iteration}-bnd.pkl'
    output_path       = f'{args.save_dir}/train.pkl'
    phn_map_path      = f'{args.data_dir}/phones.60-48-39.map.txt'

    target_path = os.path.join(args.data_dir, 'timit_for_GAN/text', args.setting+'_lm.48')
    data_length = None
    print_bar()

    ######################################################################
    # Build dataset
    #
    if args.mode=='train' or args.mode=='load':
        # load train
        train_data_set = PickleDataset(config,
                                       os.path.join(args.data_dir, config.train_feat_path),
                                       os.path.join(args.data_dir, config.train_phn_path),
                                       os.path.join(args.data_dir, config.train_orc_bnd_path),
                                       train_bnd_path=train_bnd_path,
                                       target_path=target_path,
                                       data_length=data_length, 
                                       phn_map_path=phn_map_path,
                                       name='DATA LOADER(train)')
        train_data_set.print_parameter(True)
        # load dev
        dev_data_set = PickleDataset(config,
                                     os.path.join(args.data_dir, config.dev_feat_path),
                                     os.path.join(args.data_dir, config.dev_phn_path),
                                     os.path.join(args.data_dir, config.dev_orc_bnd_path),
                                     phn_map_path=phn_map_path,
                                     name='DATA LOADER(dev)',
                                     mode='dev')
        # load test
        test_data_set = PickleDataset(config,
                                     os.path.join(args.data_dir, config.test_feat_path),
                                     os.path.join(args.data_dir, config.test_phn_path),
                                     os.path.join(args.data_dir, config.test_orc_bnd_path),
                                     phn_map_path=phn_map_path,
                                     name='DATA LOADER(test)',
                                     mode='dev')
        dev_data_set.print_parameter()
        test_data_set.print_parameter()
    else:
        # load train for evalution
        train_data_set = PickleDataset(config,
                                       os.path.join(args.data_dir, config.train_feat_path),
                                       os.path.join(args.data_dir, config.train_phn_path),
                                       os.path.join(args.data_dir, config.train_orc_bnd_path),
                                       train_bnd_path=train_bnd_path if args.mode == 'test_reduce' else None, 
                                       phn_map_path=phn_map_path,
                                       name='DATA LOADER(evaluation train)',
                                       mode='dev')
        test_data_set = PickleDataset(config,
                                     os.path.join(args.data_dir, config.test_feat_path),
                                     os.path.join(args.data_dir, config.test_phn_path),
                                     os.path.join(args.data_dir, config.test_orc_bnd_path),
                                     train_bnd_path=test_bnd_path if args.mode == 'test_reduce' else None, 
                                     phn_map_path=phn_map_path,
                                     name='DATA LOADER(evaluation test)',
                                     mode='dev')
        train_data_set.print_parameter()
        test_data_set.print_parameter()
    config.feat_dim = train_data_set.feat_dim * config.concat_window
    config.phn_size = train_data_set.phn_size
    config.mfcc_dim = train_data_set.feat_dim
    config.save_path = f'{args.save_dir}/model'
    config.load_path = f'{config.save_path}/{args.load_ckpt}'


    ######################################################################
    # Build model
    #
    if args.model_type == 'sup':
        g = SupModel(config)
    else:
        g = UnsModel(config)
    print_bar()
    print_model_parameter(config)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print ('Building Session...')
    print_bar()

    if args.prefix == '':
        fer_result_path = ''
    else:
        fer_result_path = os.path.join(args.data_dir, 'result', args.prefix+'.log')

    if args.mode == 'train':
        print_training_parameter(args, config)
        g.train(train_data_set, dev_data_set, args.aug)
        print_training_parameter(args, config)
        g.test(train_data_set, f'{args.save_dir}/train.pkl')
        g.test(test_data_set, f'{args.save_dir}/test.pkl', fer_result_path) # fer is report on dev set

    elif args.mode == 'load':
        print_training_parameter(args, config)
        g.load_ckpt(config.load_path)
        g.train(train_data_set, dev_data_set, args.aug)
        print_training_parameter(args, config)
        g.test(train_data_set, f'{args.save_dir}/train.pkl')
        g.test(test_data_set, f'{args.save_dir}/test.pkl', fer_result_path) # fer is report on dev set

    elif args.mode == 'eval':
        g.load_ckpt(config.load_path)
        g.test(train_data_set, f'{args.save_dir}/train.pkl')
        g.test(test_data_set, f'{args.save_dir}/test.pkl', fer_result_path) # fer is report on dev set

    elif args.mode == 'test_reduce':
        g.test_reduce(train_data_set, \
                      f'{args.save_dir}/train_origin.pkl', \
                      f'{args.save_dir}/train_reduce.pkl', \
                      f'{args.save_dir}/train_reduce_length.pkl', \
                      f'{args.save_dir}/train_extend.pkl')

        g.test_reduce(test_data_set, \
                      f'{args.save_dir}/test_origin.pkl', \
                      f'{args.save_dir}/test_reduce.pkl', \
                      f'{args.save_dir}/test_reduce_length.pkl', \
                      f'{args.save_dir}/test_extend.pkl')
                    
        # g.test_posterior(train_data_set, f'{args.save_dir}/train.pkl')
        # g.test_posterior(test_data_set, f'{args.save_dir}/test.pkl') # fer is report on dev set