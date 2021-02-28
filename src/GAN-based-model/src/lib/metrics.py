import torch

import editdistance as ed
from itertools import groupby
from typing import List


def get_phoneseq(frame_labels):
    phone_seq = [key for key, group in groupby(frame_labels)]
    return phone_seq

def frame_eval(pred, frame_label, length):
    """ Calculate FER with count of errors and total counts """
    pred = [p[:l] for p, l in zip(pred, length)]
    frame_label = [f[:l] for f, l in zip(frame_label, length)]
    frame_error, frame_num = calc_fer(pred, frame_label)
    return frame_error, frame_num, frame_error / frame_num

def calc_fer(prediction : List[torch.Tensor],
            ground_truth : List[torch.Tensor]) -> float :

    frame_error = 0
    frame_num = 0
    for p, g in zip(prediction, ground_truth):
        l = g.shape[0]
        p = p[:l]
        frame_error += sum(p!=g)
        frame_num += l
    return frame_error, frame_num

def per_eval(pred, frame_label, pred_length, frame_label_length=None):
    """ Calculate PER with count of errors and total counts """
    if frame_label_length is None:
        frame_label_length = pred_length
    pred = [p[:l] for p, l in zip(pred, pred_length)]
    label = [f[:l] for f, l in zip(frame_label, frame_label_length)]
    phone_error, phone_num = calc_per(pred, label)
    return phone_error, phone_num, phone_error / phone_num

def calc_per(prediction : List[torch.Tensor],
            ground_truth : List[torch.Tensor]) -> float :
    phone_error = 0
    phone_num = 0
    prediction = [get_phoneseq(p) for p in prediction]
    ground_truth = [get_phoneseq(p) for p in ground_truth]
    for p, l in zip(prediction, ground_truth):
        phone_error += ed.eval(p, l)
        phone_num += len(l)
    return phone_error, phone_num

def calc_acc(prediction : List[torch.Tensor],
            ground_truth : List[torch.Tensor]) -> float :
    acc = []
    for p, g in zip(prediction, ground_truth):
        l = g.shape[0]
        p = p[:l]
        acc.append(sum(p==g)/l)
    return sum(acc) / (len(acc) + 1e-10)

