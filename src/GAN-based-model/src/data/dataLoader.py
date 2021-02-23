import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler, BatchSampler
from src.data.dataset import PickleDataset
from functools import partial
import random


def _collate_source_fn(l, repeat=6, use_avg=False):
    batch_size  = len(l)
    feats, bnds, bnd_ranges, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0)
    seq_lengths = torch.tensor(seq_lengths).int() # length after reduce

    # for inter loss
    bnds = pad_sequence(bnds, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    bnd_ranges = pad_sequence(bnd_ranges, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    random_pick = torch.clamp(torch.randn_like(bnds) * 0.2 + 0.5, 0, 1)
    sample_frame = torch.round(bnds + random_pick * bnd_ranges).long()
    # sample_source = feats[torch.arange(batch_size * 2 * repeat).reshape(-1, 1), sample_frame]
    intra_diff_num = (sample_frame[:batch_size * repeat] != sample_frame[batch_size * repeat:]).sum(1).int()

    # for avg reduce
    # gen max possible weight
    if use_avg:
        max_bnd_range = int(torch.max(bnd_ranges))
        bnd_range2weight = {1:torch.tensor([1.])}

        all_seq_bnd_idx = torch.zeros(feats.shape[0], feats.shape[1], dtype=int)
        #all_seq_bnd_idx.fill_(-1)
        all_seq_bnd_weight = torch.zeros(feats.shape[0], feats.shape[1])
        for seq_idx in range(len(feats)):
            owe = 0
            count = int(bnds[seq_idx][0])

            for bnd_idx in range(int(seq_lengths[seq_idx])):
                r = int(bnd_ranges[seq_idx][bnd_idx])
                if r == 0:
                    r = 1
                    if count != 0:
                        count -= 1
                    else:
                        owe += 1
                else:
                    if r>0 and owe > 0:
                        re = min(owe, r-1)
                        owe = owe - re
                        r = r - re
                if r > 0:
                    if r not in bnd_range2weight.keys():
                        dist = ((torch.arange(r))/(r-1)-0.5)/0.2
                        x = torch.exp(-(dist**2)/2)
                        bnd_range2weight[r] = x/x.mean()
                    all_seq_bnd_idx[seq_idx][count:count+r] = bnd_idx
                    all_seq_bnd_weight[seq_idx][count:count+r] = bnd_range2weight[r]
                count += r
            if owe != 0:
                print('bnds,', bnds[seq_idx][:int(seq_lengths[seq_idx])], 'range, ', bnd_ranges[seq_idx][:int(seq_lengths[seq_idx])])
                assert(owe == 0)
    else:
        all_seq_bnd_idx = None
        all_seq_bnd_weight = None

    return feats, seq_lengths, intra_diff_num, sample_frame, all_seq_bnd_idx, all_seq_bnd_weight

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


def get_data_loader(dataset, batch_size, repeat=6, random_batch=True, shuffle=False, drop_last=True, use_avg=False):
    assert random_batch
    source_collate_fn = partial(_collate_source_fn, repeat=repeat, use_avg=use_avg)
    target_collate_fn = _collate_target_fn
    if random_batch:
        source_batch_size = batch_size//2 if not use_avg else batch_size
        target_batch_size = batch_size*repeat if not use_avg else batch_size
        source = DataLoader(dataset.source,
                            batch_sampler=RandomBatchSampler(dataset.source, source_batch_size),
                            collate_fn=source_collate_fn,
                            num_workers=8)
        target = DataLoader(dataset.target,
                            batch_sampler=RandomBatchSampler(dataset.target, target_batch_size),
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
