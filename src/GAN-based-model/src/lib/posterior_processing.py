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
                half_window_size = half_kernel_size * window_per_phn
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
        for i in range(j * window_per_phn):
            window = posteriors.new_zeros(*locations[0].shape)
            window[:, :seqlen, i+1:window_per_phn+i+1] = locations[j-1][:, :seqlen, i:i+1] * locations[0][:, i+1:seqlen+i+1, :window_per_phn]
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
            (batch_size, seqlen, kernel_size, window_size):
                window_size = 2 * window_per_phn * half_kernel_size + 1
                half_kernel_size = (kernel_size - 1) // 2
    """
    batch_size, seqlen, class_num = posteriors.shape
    half_kernel_size = (kernel_size-1) // 2

    all_locations = posteriors.new_zeros(batch_size, seqlen, kernel_size, 2 * window_per_phn * half_kernel_size + 1)
    center_index = (all_locations.size(-1) - 1) // 2
    all_locations[:, :, half_kernel_size, center_index] = 1
    if half_kernel_size > 0:
        all_locations[:, :, half_kernel_size+1:, center_index+1:] = posteriors_to_right_locations(
            posteriors, half_kernel_size, window_per_phn
        )
        all_locations[:, :, :half_kernel_size, :center_index] = posteriors_to_right_locations(
            posteriors.flip(dims=[1]), half_kernel_size, window_per_phn
        ).flip(dims=[1, 2, 3])

    return all_locations


def locations_to_neighborhood(features, locations):
    """
    Arguments:
        features:
            (batch_size, seqlen, feat_dim)
        locations:
            (batch_size, seqlen, kernel_size, window_size):
                kernel_size = width of a neighborhood
                window_size = width of the frame-distribution for a specific neighbor

    Outputs:
        neighborhood:
            (batch_size, seqlen, kernel_size, feat_dim)
    """
    batch_size, seqlen, feat_dim = features.shape
    _, _, kernel_size, window_size = locations.shape
    half_window_size = (window_size - 1) // 2
    
    padding = features.new_zeros(batch_size, half_window_size, feat_dim)
    padded_features = torch.cat([padding, features, padding], dim=1)

    neighborhood = 0
    for i in range(window_size):
        neighborhood = neighborhood + (
            locations[:, :, :, i].unsqueeze(-1) * 
            padded_features[:, i : seqlen + i, :].unsqueeze(2)
        )

    return neighborhood


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
    padding = phones_onehot.new_zeros(args.batch_size, (args.kernel_size - 1) // 2 * args.window_per_phn, args.phn_num)
    padded_phones_onehot = torch.cat([padding, phones_onehot, padding], dim=1)
    is_boundaries = 1 - (padded_phones_onehot[:, :-1, :] * padded_phones_onehot[:, 1:, :]).sum(dim=-1)

    seqlen = phones_onehot.size(1)
    padding_seqlen = padding.size(1)
    padded_seqlen = padded_phones_onehot.size(1)

    select_instance = 0
    for k in range(args.kernel_size):
        location = locations[select_instance, args.idx, k, :].detach().cpu()
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
    neighborhood = locations_to_neighborhood(phones_onehot, locations).detach().cpu()
    print(f'Calculate posterior-neighborhood: {time() - start}')

    start = time()
    phoneseq = torch.LongTensor([phones[0]] + [phones[i] for i in range(1, phones.size(0)) if phones[i] != phones[i - 1]])
    phoneseq_onehot = F.one_hot(phoneseq, num_classes=args.phn_num).unsqueeze(0).expand(args.batch_size, -1, -1)
    phoneseq_unfolded = F.unfold(
        phoneseq_onehot.float().transpose(1, 2).unsqueeze(-1),
        kernel_size=(args.kernel_size, 1),
        padding=((args.kernel_size - 1) // 2, 0),
    ).long().view(args.batch_size, args.phn_num, args.kernel_size, -1).transpose(1, 3)

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
