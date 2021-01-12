# extract features: https://github.com/pytorch/fairseq/blob/v0.10.1/fairseq/models/wav2vec/wav2vec2.py
import torch
import librosa
from torch.nn.utils.rnn import pad_sequence 
import fairseq
import os
import argparse
import _pickle as pk
import numpy as np
from tqdm import tqdm 
import sys

class Extractor():
    def __init__(self, ckpt='./libri960_big.pt'):
        self.ckpt = ckpt
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt])
        self.model = model[0].cuda()

    def extract_features_gpu(self, path_list):
        # https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/benchmark/upstream/wav2vec2/expert.py
        wavs = [torch.tensor(librosa.core.load(path, sr=16000)[0]) for path in path_list]
        # print([len(w) for w in wavs])
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0),
            wav_lengths.unsqueeze(1)
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        
        self.model = self.model.cuda()
        features, feat_padding_mask = self.model.extract_features(padded_wav.cuda(), wav_padding_mask.cuda())
        feat_lengths = (features.size(1) - feat_padding_mask.sum(dim=-1)).tolist()

        features = [feat[:length].cpu().detach().numpy() for feat, length in zip(features, feat_lengths)]
        # print([w.shape[0] for w in features])
        return features

    def extract_features_cpu(self, path_list):
        # https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/benchmark/upstream/wav2vec2/expert.py
        wavs = [torch.tensor(librosa.core.load(path, sr=16000)[0]) for path in path_list]
        # print([len(w) for w in wavs])
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0),
            wav_lengths.unsqueeze(1)
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        
        self.model = self.model.cpu()
        features, feat_padding_mask = self.model.extract_features(padded_wav, wav_padding_mask)
        feat_lengths = (features.size(1) - feat_padding_mask.sum(dim=-1)).tolist()

        features = [feat[:length].cpu().detach().numpy() for feat, length in zip(features, feat_lengths)]
        # print([w.shape[0] for w in features])
        return features

    def reload_model(self):
        del self.model
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt])
        self.model = model[0].cuda()


def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch'  , type=int, default=1, help='')
    parser.add_argument('--split'  , type=int, default=1, help='')
    parser.add_argument('--source_dir_path'  , type=str, default='', help='')

    # generated
    parser.add_argument('--target_meta_path', type=str, default='', help='')
    parser.add_argument('--target_phn_path'  , type=str, default='', help='')

    # to generate
    parser.add_argument('--target_wav2vec2_path', type=str, default='', help='')
    parser.add_argument('--target_wav2vec2_orc_bnd_path'  , type=str, default='', help='')
    return parser

def load(path):
    return pk.load(open(path, 'rb'))

def chunkIt(seq, num):
    avg = len(seq) // num
    out = []

    count = 0
    for i in range(num):
        if i == num-1:
            out.append(seq[count:])
        else:
            out.append(seq[count:count+avg])
            count += avg
    assert(len(out) == num)

    for i, x in enumerate([j for i in out for j in i]):
        assert(x==seq[i])
    return out


if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    all_meta_before_split = load(args.target_meta_path)
    all_phn_before_split = load(args.target_phn_path)
    assert(len(all_meta_before_split['prefix']) == len(all_phn_before_split))

    # split meta and phn
    all_meta_split = chunkIt(all_meta_before_split['prefix'], args.split)
    all_phn_split  = chunkIt(all_phn_before_split, args.split)

    # generate extractor class
    extractor = Extractor()

    all_split_all_wav2vec2_bnds = []
    for split_idx in range(args.split):
        print('proprocessing split', split_idx)
        extractor.reload_model()
        print('finish reloading')
        all_meta = all_meta_split[split_idx]
        all_phn = all_phn_split[split_idx]

        # create extractor
        print('extract wave2vec2')
        all_wav2vec2 = []
        batch_path = []
        for i in tqdm(range(len(all_meta)), file=sys.stdout):
            prefix = all_meta[i]
            wav_path = os.path.join(args.source_dir_path, prefix+'.wav')
            batch_path.append(wav_path)

            if len(batch_path) >= args.batch or (i == len(all_meta)-1):
                try:
                    feats = extractor.extract_features_gpu(batch_path)
                except RuntimeError as e:
                    print('cuda oom for', wav_path, ', use cpu extract', file=sys.stderr)
                    feats = extractor.extract_features_cpu(batch_path)
                all_wav2vec2 += feats
                batch_path = []
        
        # generate phn 
        print('extract bnd')
        all_wav2vec2_bnds = []
        for i in tqdm(range(len(all_meta)), file=sys.stdout):
            prefix = all_meta[i]
            wav2vec2 = all_wav2vec2[i]
            phns = all_phn[i]

            wav2vec2_bnds = []
            
            with open(os.path.join(args.source_dir_path, prefix+'.phn')) as f:
                all_lines = f.read().splitlines()
                for i, line in enumerate(all_lines):
                    elements = line.split()
                    assert (len(elements) == 3)
                    idx = int(elements[0])//2
                    if i == 0: 
                        assert(idx == 0)
                    else:
                        if idx < wav2vec2_bnds[-1]+1:
                            print('little phn duration: prefix: ', prefix, 'line: ', i, file=sys.stderr)
                            idx = wav2vec2_bnds[-1]+1
                    wav2vec2_bnds.append(idx)
                try:
                    final_idx = int(elements[1])//2
                    assert(abs(final_idx-len(wav2vec2)) <= 1)
                    assert((len(wav2vec2)-1)>=wav2vec2_bnds[-1])
                    wav2vec2_bnds.append(len(wav2vec2)-1)
                    assert(len(wav2vec2_bnds) == len(phns)+1)
                except:
                    print('assert error, prefix:', prefix, file=sys.stderr)
                all_wav2vec2_bnds.append(wav2vec2_bnds)
        
        # process split
        all_split_all_wav2vec2_bnds += all_wav2vec2_bnds
        if args.split == 1:
            pk.dump(np.array(all_wav2vec2, dtype=object), open(args.target_wav2vec2_path, 'wb'))
        else:
            pk.dump(np.array(all_wav2vec2, dtype=object), open(args.target_wav2vec2_path+'.'+str(split_idx+1), 'wb'))
            print('finish')

    pk.dump(np.array(all_split_all_wav2vec2_bnds, dtype=object), open(args.target_wav2vec2_orc_bnd_path, 'wb'))
