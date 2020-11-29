import torch
import numpy as np
import argparse
import _pickle as pk

"""
Metric definitions from: An improved speech segmentation quality measure: The R-value
http://users.spa.aalto.fi/orasanen/papers/IS09_r_value.pdf
"""


def boundary2location(boundaries):
    maxlen = max([max(boundary) for boundary in boundaries]) + 1
    locations = torch.zeros(len(boundaries), maxlen).long()
    for bsxid, boundary in enumerate(boundaries):
        locations[bsxid, boundary] = 1
    return locations


def compute_alignment(predicts_loc, reals_loc, one2one=True, T=2):
    assert (len(predicts_loc) == len(reals_loc))
    hits = torch.zeros(len(predicts_loc)).long()
    detecteds = []
    for bsxid, (predict_loc, real_loc) in enumerate(zip(predicts_loc, reals_loc)):
        detected = []
        real_positions = real_loc.nonzero().squeeze()
        for idx in range(len(real_positions)):
            now_pos = real_positions[idx]
            start = max(now_pos - T, 0)
            end = now_pos + (T+1)
            if one2one:
                if idx > 0 and real_positions[idx - 1] + (T+2) >= start:
                    start = (real_positions[idx] + real_positions[idx - 1]) // 2
                if idx < len(real_positions) - 1 and real_positions[idx + 1] - (T+1) <= end:
                    end = (real_positions[idx] + real_positions[idx + 1]) // 2
            if predict_loc[start:end].sum() > 0:
                hits[bsxid] += 1
                detected.append(now_pos.item())
        detecteds.append(detected)
    return hits, detecteds


def compare_boundaries(predicts, reals, one2one=True, T=2):
    predicts_loc = boundary2location(predicts)
    reals_loc = boundary2location(reals)
    hits, detecteds = compute_alignment(predicts_loc, reals_loc, one2one)
    NC = hits.sum().float()
    NT = predicts_loc.sum().float()
    NG = reals_loc.sum().float()
    return torch.tensor([NC, NT, NG])


def compute_R(tensor):
    NC, NT, NG = tensor
    HR = (NC / NG) * 100.0
    OS = ((NT / NG) - 1) * 100.0
    r1 = np.sqrt(np.power((100.0 - HR), 2) + np.power(OS, 2))
    r2 = (-OS + HR - 100.0) / np.sqrt(2)
    R = 1 - (np.abs(r1) + np.abs(r2)) / 200.0
    return R.item()


def compute_F(tensor):
    NC, NT, NG = tensor
    PRC = NC / NT
    RCL = NC / NG
    F = (2 * PRC * RCL) / (PRC + RCL)
    return [F.item(), PRC.item(), RCL.item()]


def read_pickle(path):
    return pk.load(open(path,'rb'))


def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--orc_boundary_path', type=str, default='', help='')
    parser.add_argument('-u', '--uns_boundary_path', type=str, default='', help='')
    parser.add_argument('-p', '--periodic', action="store_true")

    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    orc = read_pickle(args.orc_boundary_path)

    if args.uns_boundary_path != '':
        uns = read_pickle(args.uns_boundary_path)

        stats = compare_boundaries(orc, uns)
        R = compute_R(stats)
        F, precision, recall = compute_F(stats)

        print(f'uns: R: {R}, F: {F}, PRC: {precision}, RCL: {recall}')

    if args.periodic:
        uns = []
        for utt in orc:
            m = max(utt)
            index = 0
            uns_utt = []
            while index <= m:
                uns_utt.append(index)
                index += 8
            uns.append(uns_utt)
        uns = np.array(uns, dtype=object)

        stats = compare_boundaries(uns, orc)
        R = compute_R(stats)
        F, precision, recall = compute_F(stats)

        print(f'periodic: R: {R}, F: {F}, PRC: {precision}, RCL: {recall}')

