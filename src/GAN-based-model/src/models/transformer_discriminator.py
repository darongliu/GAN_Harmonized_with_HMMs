import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from src.lib.utils import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerDiscriminator(nn.Module):

    def __init__(self, phn_size, dis_emb_dim, num_head, hidden_dim, num_layers, max_len=None):
        super().__init__()
        self.max_len = max_len
        self.dis_emb_dim = dis_emb_dim
        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        
        encoder_layers = TransformerEncoderLayer(dis_emb_dim, num_head, hidden_dim, dropout=0)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        output_dim = dis_emb_dim

        self.flatten = nn.Flatten()

        if max_len is not None:
            self.linear = nn.Linear(max_len*output_dim, 1)
        else:
            self.linear = nn.Linear(output_dim, 1)
        self._spec_init()

    def _spec_init(self):
        for name, para in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(para.data)
            elif 'bias' in name:
                nn.init.zeros_(para.data)
    
    def positional_encoding(self, x):
        l = x.size(1)
        dim = self.dis_emb_dim

        pe = torch.zeros(l, dim).cuda()
        position = torch.arange(0, l, dtype=torch.float).unsqueeze(1).cuda()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)).cuda()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x+pe.unsqueeze(0)

    def embedding(self, x):
        return x @ self.emb_bag.weight

    def get_mask(self, outputs, lengths):
        mask = get_mask_from_lengths(lengths)
        if mask.shape[1] < outputs.shape[1]:
            mask = torch.cat([mask, mask.new_zeros(mask.shape[0], outputs.shape[1]-mask.shape[1])], dim=1)
        return mask

    def mask_pool(self, outputs, lengths=None):
        """ Mean pooling of masked elements """
        if lengths is None:
            return outputs.mean(1)
        mask = self.get_mask(outputs, lengths).unsqueeze(-1)
        outputs = outputs * mask
        return outputs.sum(1) / mask.sum(1)
    
    def forward(self, inputs, inputs_len=None):
        outputs = self.embedding(inputs)
        outputs = self.positional_encoding(outputs)
        mask = self.get_mask(outputs, inputs_len)
        outputs = self.transformer_encoder(outputs.transpose(0, 1), src_key_padding_mask=mask)
        outputs = outputs.transpose(0, 1)

        if self.max_len is not None:
            # (B, T, D) -> (B, T*D)
            outputs = self.flatten(outputs)
        else:
            # (B, T, D) -> (B, D)
            outputs = self.mask_pool(outputs, inputs_len)
        outputs = self.linear(outputs)

        return outputs

if __name__ == '__main__':
    import torch
    from src.models.transformer_discriminator import TransformerDiscriminator
    # prepare data
    batch_size = 3
    T = 7
    phn_size = 4
    feat = torch.rand(batch_size, T, phn_size).cuda()
    feat = torch.nn.functional.softmax(feat, dim=-1)
    feat_length = torch.randint(low=1, high=T, size=(batch_size,)).cuda()
    print('feat_length b4 sorting', feat_length)
    feat_length = torch.sort(feat_length, descending=True)[0]
    print('feat_length', feat_length)

    dis_emb_dim = 16
    num_head = 8
    hidden_dim = 15
    num_layers = 2
    model  = TransformerDiscriminator(phn_size, dis_emb_dim, num_head, hidden_dim, num_layers, max_len=T).cuda()

    result = model(feat, inputs_len=feat_length)
