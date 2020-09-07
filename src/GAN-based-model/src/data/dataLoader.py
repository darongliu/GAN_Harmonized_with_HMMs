import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler, BatchSampler
from src.data.dataset import PickleDataset
from functools import partial
import random
import numpy as np


def _collate_segmented_source_fn(l, repeat=6):
    batch_size  = len(l)
    feats, _, bnds, bnd_ranges, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0).repeat(repeat * 2, 1, 1)
    bnds = pad_sequence(bnds, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    bnd_ranges = pad_sequence(bnd_ranges, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    seq_lengths = torch.tensor(seq_lengths).repeat(repeat * 2).int()

    random_pick = torch.clamp(torch.randn_like(bnds) * 0.2 + 0.5, 0, 1)
    sample_frame = torch.round(bnds + random_pick * bnd_ranges).long()
    sample_source = feats[torch.arange(batch_size * 2 * repeat).reshape(-1, 1), sample_frame]
    intra_diff_num = (sample_frame[:batch_size * repeat] != sample_frame[batch_size * repeat:]).sum(1).int()
    return sample_source, seq_lengths, intra_diff_num 

def _collate_non_segmented_source_fn(l, repeat, sample_range):
    batch_size = len(l)
    feats, feats_length, _, _, _ = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0)
    feats_length = torch.tensor(feats_length).int()

    def _generate_rand_id(max_length, feats_length, sample_range=[2,5]):
        current = 0
        all_idx = []
        while True:
            if current >= max_length:
                break
            all_idx.append(current)

            # update current
            current += np.random.randint(sample_range[0], sample_range[1]+1)
        all_idx = torch.tensor(all_idx)

        length = torch.lt(all_idx.unsqueeze(0), feats_length.unsqueeze(1))
        length = length.int().sum(-1)

        return all_idx, length

    max_length = feats.shape[1]
    all_idx_tensor = []
    all_new_feat_length = []
    for _ in range(repeat):
        idx_tensor, length = _generate_rand_id(max_length, feats_length, sample_range=sample_range)
        all_idx_tensor.append(idx_tensor)
        all_new_feat_length.append(length)
    all_idx_tensor = pad_sequence(all_idx_tensor, batch_first=True, padding_value=-1)
    # all_idx_tensor: (repeat, seqlen)
    sampled_feats = feats[:, all_idx_tensor.view(-1), :].view(-1, all_idx_tensor.size(-1), feats.size(-1))
    # sampled_feats: (batch_size * repeat, seqlen, featdim)
    sampled_feats_length = torch.stack(all_new_feat_length).transpose(0, 1).reshape(-1)
    # sampled_feats_length: (batch_size * repeat)

    return sampled_feats, sampled_feats_length, None

def _collate_target_fn(l):
    feats, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0).long()
    seq_lengths = torch.tensor(seq_lengths).int()
    return feats, seq_lengths

def _collate_sup_fn(l, padding_label=-100):
    feats, frame_labels = zip(*l)
    lengths = torch.tensor([len(feat) for feat in feats])

    feats = pad_sequence(feats, batch_first=True, padding_value=0)
    frame_labels = pad_sequence(frame_labels, batch_first=True, padding_value=padding_label)
    return feats, frame_labels, lengths


class RandomBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield random.sample(range(len(self.data_source)), self.batch_size)

    def __len__(self):
        return len(self.data_source)


def get_data_loader(dataset, batch_size, repeat=6, sample_range=[2,5], use_posterior_bnd=False, random_batch=True, shuffle=False, drop_last=True):
    assert random_batch
    if use_posterior_bnd:
        source_collate_fn = partial(_collate_non_segmented_source_fn, repeat=repeat, sample_range=sample_range)
        src_batch_size = batch_size
        tgt_batch_size = batch_size * repeat
    else:
        source_collate_fn = partial(_collate_segmented_source_fn, repeat=repeat)
        src_batch_size = batch_size // 2
        tgt_batch_size = batch_size * repeat
    
    target_collate_fn = _collate_target_fn
    if random_batch:
        source = DataLoader(dataset.source,
                            batch_sampler=RandomBatchSampler(dataset.source, src_batch_size),
                            collate_fn=source_collate_fn,
                            num_workers=8)
        target = DataLoader(dataset.target,
                            batch_sampler=RandomBatchSampler(dataset.target, tgt_batch_size),
                            collate_fn=target_collate_fn,
                            num_workers=8)
    return source, target

def get_dev_data_loader(dataset, batch_size, shuffle=False, drop_last=False):
    _collate_dev_fn = partial(_collate_sup_fn, padding_label=-100)
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_dev_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader

def get_sup_data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_sup_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader


if __name__ == '__main__':
    # run in src/GAN-based-model/  # and import to ipython
    import yaml
    import os
    from src.data.dataset import PickleDataset

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    def read_config(path):
        return AttrDict(yaml.load(open(path, 'r')))

    data_dir = '/home/darong/frequent_data/GAN_Harmonized_with_HMMs/data'
    config_path = '/home/darong/darong/GAN_Harmonized_with_HMMs/src/GAN-based-model/config.yaml'
    config = read_config(config_path)

    train_bnd_path = f'{data_dir}/timit_for_GAN/audio/timit-train-orc1-bnd.pkl'
    target_path = os.path.join(data_dir, 'timit_for_GAN/text', 'match_lm.48')
    data_length = None
    phn_map_path      = f'{data_dir}/phones.60-48-39.map.txt'

    train_data_set = PickleDataset(config,
                                os.path.join(data_dir, config.train_feat_path),
                                os.path.join(data_dir, config.train_phn_path),
                                os.path.join(data_dir, config.train_orc_bnd_path),
                                train_bnd_path=train_bnd_path,
                                target_path=target_path,
                                data_length=data_length, 
                                phn_map_path=phn_map_path,
                                name='DATA LOADER(train)')

    train_source, train_target = get_data_loader(train_data_set,
                                                batch_size=config.batch_size,
                                                repeat=config.repeat,
                                                use_posterior_bnd=True)

    train_source, train_target = iter(train_source), iter(train_target)

    # test source
    feats, all_idx_tensor, all_new_feat_length = next(train_source)
    print('feat shape', feats.shape)
    print('all idx tensor', all_idx_tensor)
    print('all new feat length', all_new_feat_length)
    feats, all_idx_tensor, all_new_feat_length = next(train_source)
    print('feat shape', feats.shape)
    print('all idx tensor', all_idx_tensor)
    print('all new feat length', all_new_feat_length)
    # target
    target_idx, target_len = next(train_target)
    target_idx, target_len = next(train_target)
