import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def posteriors_to_right_locations(posteriors, half_kernel_size, window_per_phn):
    """
    Arguments:
        posteriors:
            (batch_size, seqlen, class_num)

    Outputs:
        locations:
            (batch_size, seqlen, half_kernel_size, half_window_size):
                half_kernel_size = (kernel_size - 1) // 2
                half_window_size = window_per_phn * half_kernel_size
    """
    batch_size, seqlen, class_num = posteriors.shape
    padding = posteriors.new_zeros(batch_size, window_per_phn * half_kernel_size, class_num)
    padded_posteriors = torch.cat([posteriors, padding], dim=1)
    padded_seqlen = padded_posteriors.size(1)
    
    locations = [posteriors.new_zeros(batch_size, padded_seqlen, window_per_phn * half_kernel_size)]
    same_to_right = (padded_posteriors[:, :padded_seqlen - 1, :] * padded_posteriors[:, 1:, :]).sum(dim=-1)
    same_to_right = F.pad(same_to_right, (0, 1))

    cumulatively_same = 1
    for i in range(window_per_phn):
        same = same_to_right[:, i : seqlen + i]
        locations[0][:, :seqlen, i] = cumulatively_same * (1 - same)
        cumulatively_same = cumulatively_same * same

    for j in range(1, half_kernel_size):
        windows = []
        for i in range(window_per_phn):
            window = posteriors.new_zeros(*locations[0].shape)
            window[:, :seqlen, i+1:j*window_per_phn+i+1] = locations[j-1][:, i+1:seqlen+i+1, :j*window_per_phn] * locations[0][:, :seqlen, i:i+1]
            windows.append(window)
        locations.append(torch.stack(windows, dim=0).sum(dim=0))

    return torch.stack(locations, dim=2)[:, :seqlen, :, :]


def posteriors_to_locations(posteriors, kernel_size, window_per_phn):
    """
    Arguments:
        posteriors:
            (batch_size, seqlen, class_num)

    Outputs:
        locations:
            The outputs of posteriors_to_right_locations() on both left and right sides
            [(batch_size, seqlen, half_kernel_size, half_window_size), ...] of length 2:
                half_kernel_size = (kernel_size - 1) // 2
                half_window_size = window_per_phn * half_kernel_size
    """
    batch_size, seqlen, class_num = posteriors.shape
    half_kernel_size = (kernel_size - 1) // 2

    left_locations, right_locations = None, None
    if half_kernel_size > 0:
        left_locations = posteriors_to_right_locations(
            posteriors.flip(dims=[1]), half_kernel_size, window_per_phn
        ).flip(dims=[1, 2, 3])
        right_locations = posteriors_to_right_locations(
            posteriors, half_kernel_size, window_per_phn
        )

    return [left_locations, right_locations]


def locations_to_neighborhood(features, locations, out_kernel_size=None, padding='zeros'):
    """
    Arguments:
        features:
            (batch_size, seqlen, feat_dim)
        locations:
            The output of posteriors_to_locations()
            [(batch_size, seqlen, half_kernel_size, half_window_size), ...] of length 2:
                half_kernel_size = (kernel_size - 1) // 2
                half_window_size = window_per_phn * half_kernel_size

    Outputs:
        neighborhood:
            (batch_size, seqlen, kernel_size, feat_dim)
    """
    locations = slice_locations(locations, out_kernel_size)

    left_locations, right_locations = locations
    batch_size, seqlen, feat_dim = features.shape
    _, _, half_kernel_size, half_window_size = left_locations.shape
    
    if padding == 'zeros':
        padding = features.new_zeros(batch_size, half_window_size, feat_dim)
        padded_features = torch.cat([padding, features, padding], dim=1)
    elif padding == 'replicate':
        left_padding = features[:, 0:1, :].expand(batch_size, half_window_size, feat_dim)
        right_padding = features[:, -1:, :].expand(batch_size, half_window_size, feat_dim)
        padded_features = torch.cat([left_padding, features, right_padding], dim=1)
    else:
        raise NotImplementedError
    
    left_neighborhood = 0
    right_neighborhood = 0
    for i in range(half_window_size):
        left_neighborhood = left_neighborhood + (
            left_locations[:, :, :, i:i+1] * 
            padded_features[:, i:seqlen+i, :].unsqueeze(2)
        )
        right_neighborhood = right_neighborhood + (
            right_locations[:, :, :, i:i+1] * 
            padded_features[:, half_window_size+i+1:half_window_size+seqlen+i+1, :].unsqueeze(2)
        )
    neighborhood = torch.cat([left_neighborhood, features.unsqueeze(2), right_neighborhood], dim=2)

    return neighborhood


def slice_locations(locations, out_kernel_size=None):
    if out_kernel_size is None:
        return locations

    left_locations, right_locations = locations
    out_half_kernel_size = (out_kernel_size - 1) // 2
    out_half_window_size = left_locations.size(-1) // left_locations.size(-2) * out_half_kernel_size
    left_locations = left_locations[:, :, -out_half_kernel_size:, -out_half_window_size:]
    right_locations = right_locations[:, :, :out_half_kernel_size, :out_half_window_size]

    return [left_locations, right_locations]


def locations_to_full_locations(locations, out_kernel_size=None):
    locations = slice_locations(locations, out_kernel_size)

    left_locations, right_locations = locations
    batch_size, seqlen, half_kernel_size, half_window_size = left_locations.shape

    full_locations = left_locations.new_zeros(batch_size, seqlen, 2 * half_kernel_size + 1, 2 * half_window_size + 1)
    full_locations[:, :, half_kernel_size, half_window_size] = 1
    if half_kernel_size > 0:
        full_locations[:, :, :half_kernel_size, :half_window_size] = left_locations
        full_locations[:, :, half_kernel_size+1:, half_window_size+1:] = right_locations

    return full_locations


if __name__ == '__main__':
    import os
    import sys
    import argparse
    from time import time
    from ipdb import set_trace

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--debug_path', default='debug')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--phn_num', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=9)
    parser.add_argument('--out_kernel_size', type=int, default=5)
    parser.add_argument('--window_per_phn', type=int, default=7)
    parser.add_argument('--idx', type=int, default=50)
    args = parser.parse_args()

    if not os.path.isdir(args.debug_path):
        os.makedirs(args.debug_path)

    list_of_idx = [0] * 5 + [1] * 3 + [2] * 6 + [3] * 2
    phones = torch.LongTensor(list_of_idx * (100 // len(list_of_idx))).to(args.device)
    phones_onehot = F.one_hot(phones, num_classes=args.phn_num).unsqueeze(0).expand(args.batch_size, -1, -1)

    start = time()
    locations = posteriors_to_locations(phones_onehot, args.kernel_size, args.window_per_phn)
    print(f'Calculate locations: {time() - start}')
    
    start = time()
    padding = phones_onehot.new_zeros(args.batch_size, (args.out_kernel_size - 1) // 2 * args.window_per_phn, args.phn_num)
    padded_phones_onehot = torch.cat([padding, phones_onehot, padding], dim=1)
    is_boundaries = 1 - (padded_phones_onehot[:, :-1, :] * padded_phones_onehot[:, 1:, :]).sum(dim=-1)

    seqlen = phones_onehot.size(1)
    padding_seqlen = padding.size(1)
    padded_seqlen = padded_phones_onehot.size(1)

    select_instance = 0
    full_locations = locations_to_full_locations(locations, args.out_kernel_size)
    for k in range(args.out_kernel_size):
        location = full_locations[select_instance, args.idx, k, :].detach().cpu()
        half_window = (location.size(-1) - 1) // 2
        abs_start = args.idx - half_window + padding_seqlen
        abs_end = args.idx + half_window + 1 + padding_seqlen

        abs_location = torch.cat([torch.zeros(abs_start), location, torch.zeros(padded_seqlen - abs_end)], dim=-1)
        is_boundary = is_boundaries[select_instance]
        assert abs_location.size(0) == is_boundary.size(0) + 1

        plt.figure(figsize=(20, 10))
        plt.plot(range(len(abs_location)), abs_location)
        for x, is_bnd in enumerate(is_boundary):
            if is_bnd.item():
                plt.axvline(x=float(x) + 0.5, color='r')

        plt.axvline(x=args.idx + padding_seqlen, color='g')
        plt.savefig(f'{args.debug_path}/{k}.png')
    print(f'Draw location distributions: {time() - start}')

    start = time()
    neighborhood = locations_to_neighborhood(phones_onehot, locations, args.out_kernel_size).detach().cpu()
    print(f'Calculate posterior-neighborhood: {time() - start}')

    start = time()
    phoneseq = torch.LongTensor([phones[0]] + [phones[i] for i in range(1, phones.size(0)) if phones[i] != phones[i - 1]])
    phoneseq_onehot = F.one_hot(phoneseq, num_classes=args.phn_num).unsqueeze(0).expand(args.batch_size, -1, -1)
    phoneseq_unfolded = F.unfold(
        phoneseq_onehot.float().transpose(1, 2).unsqueeze(-1),
        kernel_size=(args.out_kernel_size, 1),
        padding=((args.out_kernel_size - 1) // 2, 0),
    ).long().view(args.batch_size, args.phn_num, args.out_kernel_size, -1).transpose(1, 3)

    pivot = 0
    real_neighborhood = [phoneseq_unfolded[:, pivot, :, :]]
    for i in range(1, phones.size(0)):
        if phones[i] == phones[i - 1]:
            real_neighborhood.append(real_neighborhood[-1])
        else:
            pivot += 1
            real_neighborhood.append(phoneseq_unfolded[:, pivot, :, :])
    real_neighborhood = torch.stack(real_neighborhood, dim=1)
    print(f'Calculate real-neighborhood: {time() - start}')
    
    if torch.allclose(neighborhood, real_neighborhood):
        print('Test passed')
    else:
        print('Test failed')
