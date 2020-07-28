import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.utils import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMDiscriminator(nn.Module):

    def __init__(self, phn_size, dis_emb_dim, hidden_dim, num_layers, bidirectional=True, max_len=None):
        super().__init__()
        self.max_len = max_len
        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        
        self.recurrent_layer = nn.LSTM(input_size=dis_emb_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        output_dim = 2*hidden_dim if bidirectional else hidden_dim

        self.flatten = nn.Flatten()

        if max_len is not None:
            self.linear = nn.Linear(max_len*output_dim, 1)
        else:
            self.linear = nn.Linear(output_dim, 1)
        self._spec_init()

    def _spec_init(self):
        for name, para in self.recurrent_layer.named_parameters():
            if len(para.shape) >= 2:
                nn.init.orthogonal_(para.data)
            else:
                nn.init.normal_(para.data)

        for name, para in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(para.data)
            elif 'bias' in name:
                nn.init.zeros_(para.data)

    def embedding(self, x):
        return x @ self.emb_bag.weight

    def mask_pool(self, outputs, lengths=None):
        """ Mean pooling of masked elements """
        if lengths is None:
            return outputs.mean(1)
        mask = get_mask_from_lengths(lengths).unsqueeze(-1)
        if mask.shape[1] < outputs.shape[1]:
            mask = torch.cat([mask, mask.new_zeros(mask.shape[0], outputs.shape[1]-mask.shape[1], 1)], dim=1)
        outputs = outputs * mask
        return outputs.sum(1) / mask.sum(1)
    
    def forward(self, inputs, inputs_len=None):
        outputs = self.embedding(inputs)
        l = outputs.size(1)
        output_packed = nn.utils.rnn.pack_padded_sequence(outputs, lengths=inputs_len, batch_first=True)
        memory_packed, h = self.recurrent_layer(output_packed) # h: [batch x 1 x hdim]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(memory_packed, batch_first=True, total_length=l)

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
    from src.models.lstm_discriminator import LSTMDiscriminator
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

    dis_emb_dim = 10
    hidden_dim = 15
    num_layers = 2
    model  = LSTMDiscriminator(phn_size, dis_emb_dim, hidden_dim, num_layers, max_len=T).cuda()

    result = model(feat, inputs_len=feat_length)

