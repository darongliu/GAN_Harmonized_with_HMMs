import sys
import pickle
import numpy as np
import random
from collections import Counter

import torch
from torch.utils.data import Dataset


class PickleDataset(Dataset):
    """
    Instance Variables:
        self.phn_size       : 48
        self.phn2idx        : 48 phns -> 48 indices
        self.idx2phn        : 48 indices -> 48 phns
        self.phn_mapping    : 48 indices -> 39 phns
        self.sil_idx        : 
        self.feat_dim       : 39
        self.source         : source dataset
        self.target         : target dataset
        self.dev            : dev dataset
    """
    def __init__(self, config,
                 feat_path,                                 # feat data path
                 phn_path,                                  # phone data path
                 orc_bnd_path,                              # oracle boundary data path
                 train_bnd_path='',                       # pretrained boundary data path
                 target_path=None,                          # text data path
                 data_length=None,                          # num of non-matching data, None would be set to len(feats)
                 phn_map_path='./phones.60-48-39.map.txt',
                 name='DATA LOADER',
                 mode='train'):
        super().__init__()
        args = locals()
        self.add_to_self_attr(args)

        cout_word = f'{name}: loading    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.read_phn_map(phn_map_path)

        feats     = pickle.load(open(feat_path, 'rb'))
        orc_bnd   = pickle.load(open(orc_bnd_path, 'rb'))
        phn_label = pickle.load(open(phn_path, 'rb'))
        train_bound = pickle.load(open(train_bnd_path, 'rb'))
        assert (len(feats) == len(orc_bnd) == len(phn_label) == len(train_bound))
        
        self.data_length = len(feats) if not data_length else data_length
        self.process_label(orc_bnd[:self.data_length], phn_label[:self.data_length])
        self.process_train_bnd(feats[:self.data_length], train_bound[:self.data_length])

        if target_path:
            self.process_target(target_path)

        self.create_datasets(mode)

        sys.stdout.write('\b' * len(cout_word))
        cout_word = f'{name}: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()
        print ('='*80)

    def add_to_self_attr(self, args):
        for k, v in args.items():
            if k not in ['self', '__class__']:
                if k == 'config':
                    self.concat_window   = v.concat_window     # concat window size
                    self.phn_max_length  = v.phn_max_length    # max length of phone sequence
                    self.feat_max_length = v.feat_max_length   # max length of feat sequence
                    self.sample_var      = v.sample_var
                    self.target_augment_prob = v.target_augment_prob
                else:
                    setattr(self, k, v)

    def read_phn_map(self, phn_map_path):
        phn_mapping = {}
        with open(phn_map_path, 'r') as f:
            for line in f:
                if line.strip() != "":
                    p60, p48, p39 = line.split()
                    phn_mapping[p48] = p39

        all_phn = list(phn_mapping.keys())
        assert(len(all_phn) == 48)
        self.phn_size = len(all_phn)
        self.phn2idx        = dict(zip(all_phn, range(len(all_phn))))
        self.idx2phn        = dict(zip(range(len(all_phn)), all_phn))
        self.phn_mapping    = dict([(i, phn_mapping[phn]) for i, phn in enumerate(all_phn)])
        self.sil_idx = self.phn2idx['sil']

    def process_label(self, orc_bnd, phn_label):
        assert len(orc_bnd) == len(phn_label) == self.data_length
        self.frame_labels = []
        for bnd, phn in zip(orc_bnd, phn_label):
            assert len(bnd) == len(phn) + 1
            frame_label = []
            if bnd[0] != 0:
                bnd[0] = 0
            for prev_b, b, p in zip(bnd, bnd[1:], phn):
                frame_label += [self.phn2idx[p]] * (b-prev_b)
            frame_label += [self.phn2idx[phn[-1]]]
            self.frame_labels.append(torch.tensor(frame_label))

    def process_train_bnd(self, feats, train_bnd):
        assert len(feats) == self.data_length == len(train_bnd) == len(self.frame_labels)
        half_window = (self.concat_window-1) // 2
        self.feat_dim = feats[0].shape[-1]
        self.feats = []
        self.feats_segment_len = []
        self.feats_labels = []
        i = 0
        for feat, bnd, frame_label in zip(feats, train_bnd, self.frame_labels):
            segment_feature = []
            segment_len = []
            segment_label = []

            _feat_ = np.concatenate([np.tile(feat[0], (half_window, 1)), feat,
                                     np.tile(feat[-1], (half_window, 1))], axis=0)
            feature = torch.tensor([np.reshape(_feat_[l : l+self.concat_window], [-1])
                                    for l in range(len(feat))])[:self.feat_max_length]

            for prev_b, b in zip(bnd[:-1], bnd[1:]):
                if min(prev_b, b) >= len(frame_label):
                    break
                if prev_b == b:
                    segment_feature.append(feature[prev_b:prev_b+1])
                    segment_len.append(1)
                    segment_label.append(frame_label[prev_b])
                else:
                    segment_feature.append(feature[min(prev_b, b):max(prev_b, b)])
                    segment_len.append(abs(prev_b - b))
                    # print(i, len(frame_label), prev_b, b)
                    segment_label.append(Counter(list(frame_label[min(prev_b, b):max(prev_b, b)])).most_common(1)[0][0])
            self.feats.append(segment_feature) 
            self.feats_segment_len.append(torch.tensor(segment_len))
            self.feats_labels.append(torch.tensor(segment_label))
            i += 1

    def process_target(self, target_path):
        target_data = [line.strip().split() for line in open(target_path, 'r')]
        target_data = [[self.phn2idx[t] for t in target] for target in target_data]
        self.target_data = [torch.tensor(target).int() for target in target_data]

    def create_datasets(self, mode):
        if mode == 'train':
            self.source = SourceDataset(self.feats,
                                        self.feats_segment_len)
            self.target = TargetDataset(self.target_data, self.sil_idx, augment_prob=self.target_augment_prob)
        self.dev = DevDataset(self.feats, self.feats_segment_len, self.frame_labels, self.feats_labels)

    def print_parameter(self, target=False):
        print ('Data Loader Parameter:')
        print (f'   phoneme number:  {self.phn_size}')
        print (f'   phoneme length:  {self.phn_max_length}')
        print (f'   feature dim:     {self.feat_dim * self.concat_window}')
        print (f'   feature windows: {self.concat_window}')
        print (f'   feature length:  {self.feat_max_length}')
        print (f'   source size:     {self.data_length}')
        if target:
            print (f'   target size:     {len(self.target)}')
        print (f'   feat_path:       {self.feat_path}')
        print (f'   phn_path:        {self.phn_path}')
        print (f'   orc_bnd_path:    {self.orc_bnd_path}')
        print (f'   train_bnd_path:  {self.train_bnd_path}')
        print (f'   target_path:     {self.target_path}')
        print ('='*80)


class TargetDataset(Dataset):
    def __init__(self, target_data, sil_idx, augment_prob=None):
        self.target_data = target_data
        self.sil_idx = sil_idx
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        feat = self._data_augmentation(self.target_data[index])
        return feat, len(feat)

    def _data_augmentation(self, seq):
        if self.augment_prob is None or self.augment_prob == 'None':
            return seq
        new_seq = [] 
        for s in seq:
            if s == self.sil_idx:
                # new_seq.extend([s]*np.random.choice([0, 1, 2], p=[0.04, 0.8, 0.16]))
                new_seq.extend([s])
            else:
                new_seq.extend([s]*np.random.choice(list(range(len(self.augment_prob))), p=self.augment_prob))
        return torch.tensor(new_seq)

class SourceDataset(Dataset):
    def __init__(self, feats, feats_segment_len):
        self.feats = feats
        self.feats_segment_len = feats_segment_len
        assert len(feats) == len(self.feats_segment_len)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        feat = self.feats[index]
        feat_segment_len = self.feats_segment_len[index]
        train_seq_length = len(feat)
        return feat, feat_segment_len, train_seq_length


class DevDataset(Dataset):
    def __init__(self, feats, feats_segment_len, frame_labels, feats_labels):
        self.feats = feats
        self.feats_segment_len = feats_segment_len
        self.frame_labels = frame_labels
        self.feats_labels = feats_labels
        assert len(feats) == len(self.feats_segment_len) == len(frame_labels) == len(feats_labels)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        feat = self.feats[index]
        feat_segment_len = self.feats_segment_len[index]
        train_seq_length = len(feat)
        frame_label = self.frame_labels[index]
        feats_label = self.feats_labels[index]
        origin_length = len(frame_label)
        return feat, feat_segment_len, train_seq_length, feats_label, origin_length, frame_label
