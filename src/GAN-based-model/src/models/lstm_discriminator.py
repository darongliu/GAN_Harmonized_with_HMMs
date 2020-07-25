import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.utils import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMDiscriminator(nn.Module):

    def __init__(self, phn_size, dis_emb_dim, hidden_size, num_layers, bidirectional=True, max_len=None):
        super().__init__()
        self.max_len = max_len

        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        self.recurrent_layer = nn.LSTM(input_size=dis_emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        output_dim = 2*hidden_size if bidirectional else hidden_size

        self.flatten = nn.Flatten()

        if max_len is not None:
            self.linear = nn.Linear(max_len*output_dim, 1)
        else:
            self.linear = nn.Linear(output_dim, 1)

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
        l = outputs.size(1)
        output_packed = nn.utils.rnn.pack_padded_sequence(outputs, lengths=inputs_len, batch_first=True)
        memory_packed, h = self.recurrent_layer(output_packed) # h: [batch x 1 x hdim]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(memory_packed, batch_first=True, total_length=l)

        if self.max_len is not None:
            # (B, T, D) -> (B, T*D)
            outputs = self.flatten(outputs)
        outputs = self.linear(outputs)

        return outputs

if __name__ == '__main__':
    # prepare data
    batch_size = 3
    T = 7
    phn_size = 4
    feat = torch.rand(batch_size, T, phn_size)
    feat = torch.nn.functional.softmax(feat, dim=-1)
    feat_length = torch.randint(low=1, high=T, size=(batch_size,))
    print('feat_length b4 sorting', feat_length)
    feat_length = torch.sort(feat_length, descending=True)[0]
    print('feat_length', feat_length)

    dis_emb_dim = 10
    hidden_size = 15
    num_layers = 2
    model  = LSTMDiscriminator(phn_size, dis_emb_dim, hidden_size, num_layers, max_len=T)

    result = model(feat, inputs_len=feat_length)

