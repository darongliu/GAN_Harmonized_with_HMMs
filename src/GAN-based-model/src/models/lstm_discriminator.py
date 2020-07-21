import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.utils import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMDiscriminator(nn.Module):

    def __init__(self, phn_size, dis_emb_dim, hidden_dim, layer_num, bidirectional=True):
        super().__init__()
        #self.max_len = max_len

        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        self.lstms

        self.lrelu_1 = nn.LeakyReLU()
        self.conv_2 = nn.ModuleList([
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
        ])
        self.flatten = nn.Flatten()
        self.lrelu_2 = nn.LeakyReLU()
        if ngram is not None:
            self.linear = nn.Linear(ngram*4*hidden_dim2, 1)
        elif max_len is not None:
            self.linear = nn.Linear(max_len*4*hidden_dim2, 1)
        else:
            self.linear = nn.Linear(4*hidden_dim2, 1)

    def embedding(self, x):
        return x @ self.emb_bag.weight

    def mask_pool(self, output, lengths=None):
        """ Mean pooling of masked elements """
        if input_len is None:
            return output.mean(1)
        mask = get_mask_from_lengths(lengths).unsqueeze(-1)
        output = output * mask
        return output.sum(1) / mask.sum(1)
    
    def forward(self, inputs, inputs_len=None):
        outputs = self.embedding(inputs)
        outputs = outputs.transpose(1, 2)
        outputs = torch.cat([conv(outputs) for conv in self.conv_1], dim=1)
        outputs = self.lrelu_1(outputs)
        outputs = torch.cat([conv(outputs) for conv in self.conv_2], dim=1)
        outputs = outputs.transpose(1, 2)
        if self.ngram is not None or self.max_len is not None:
            # (B, T, D) -> (B, T*D)
            outputs = self.flatten(outputs)
        else:
            # (B, T, D) -> (B, D)
            output = self.mask_pool(outputs, inputs_len)
        outputs = self.lrelu_2(outputs)
        outputs = self.linear(outputs)
        return outputs

    def forward_no_bnd(self, inputs, inputs_len=None):
        pass