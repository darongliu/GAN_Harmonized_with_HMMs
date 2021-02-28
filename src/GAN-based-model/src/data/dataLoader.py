import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler, BatchSampler
from src.data.dataset import PickleDataset
from functools import partial
import random


def _collate_source_fn(l, repeat=6):
    feats, feats_segment_len, seq_lengths = zip(*l)

    recover_idx_0 = [torch.tensor([i]*len(f)) for i, f in enumerate(feats)]
    recover_idx_1 = [torch.arange(len(f)) for f in feats]

    # concat
    feats = [j for i in feats for j in i]
    feats = pad_sequence(feats, batch_first=True, padding_value=0)

    feats_segment_len = torch.cat(feats_segment_len).int() # length after reduce
    recover_idx_0 = torch.cat(recover_idx_0).long() # length after reduce
    recover_idx_1 = torch.cat(recover_idx_1).long() # length after reduce

    seq_lengths = torch.tensor(seq_lengths).int()

    # reorder according to length
    indice = torch.sort(feats_segment_len, descending=True)[1]
    return feats[indice], feats_segment_len[indice], recover_idx_0[indice], recover_idx_1[indice], seq_lengths

def _collate_dev_fn(l, padding_label=-100):
    feats, feats_segment_len, seq_lengths, feats_labels, origin_lengths, frame_labels = zip(*l)

    recover_idx_0 = [torch.tensor([i]*len(f)) for i, f in enumerate(feats)]
    recover_idx_1 = [torch.arange(len(f)) for f in feats]

    # concat
    feats = [j for i in feats for j in i]
    feats = pad_sequence(feats, batch_first=True, padding_value=0)

    feats_segment_len = torch.cat(feats_segment_len).int() # length after reduce
    recover_idx_0 = torch.cat(recover_idx_0).long() # length after reduce
    recover_idx_1 = torch.cat(recover_idx_1).long() # length after reduce

    seq_lengths = torch.tensor(seq_lengths).int()
    origin_lengths = torch.tensor(origin_lengths).int()

    frame_labels = pad_sequence(frame_labels, batch_first=True, padding_value=padding_label)
    feats_labels = pad_sequence(feats_labels, batch_first=True, padding_value=padding_label)

    # reorder according to length
    indice = torch.sort(feats_segment_len, descending=True)[1]
    return feats[indice], feats_segment_len[indice], recover_idx_0[indice], recover_idx_1[indice], seq_lengths, feats_labels, origin_lengths, frame_labels

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


def get_data_loader(dataset, batch_size, repeat=6, random_batch=True, shuffle=False, drop_last=True):
    assert random_batch
    source_collate_fn = partial(_collate_source_fn)
    target_collate_fn = _collate_target_fn
    if random_batch:
        source_batch_size = batch_size
        target_batch_size = batch_size
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
    # _collate_dev_fn = partial(_collate_dev_fn, padding_label=-100)
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_dev_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader

def get_sup_data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_sup_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader
