import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.modules import MLP
from src.lib.utils import masked_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Frame2Phn(nn.Module):
    def __init__(self, input_size, output_size, hidden):
        super().__init__()
        if type(hidden) == int:
            hidden = str(hidden)
        elif type(hidden) == list:
            hidden = '_'.join([str(h) for h in hidden])
        self.model = MLP(input_size, output_size, hidden)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        """
        Inputs: sampled features
            x: (batch, timesteps, feature_size)
            mask_len: list of int, lengths

        Outputs:
            prob: (batch, timesteps, phn_size)
        """
        output = self.model(x)
        prob = self.softmax(output) # no gumbel

        return prob

    def calc_seq_loss(self, x, y):
        x = x[:, :y.size(1)]
        output = self.model(x)
        prob = self.softmax(output)
        output = output.transpose(1,2)
        seq_loss = F.cross_entropy(output, y)
        return seq_loss, prob

class Frame2PhnLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden, num_layers=1):
        super().__init__()
        assert type(hidden) == int
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.num_layers = num_layers
        self.hidden = hidden
        self.output = torch.nn.Linear(hidden, output_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, mask_len=None):
        """
        Inputs: sampled features
            x: (batch, timesteps, feature_size)
            mask_len: list of int, lengths

        Outputs:
            prob: (batch, timesteps, phn_size)
        """
        batch_size = x.shape[0]
        h0 = x.new_zeros(self.num_layers, batch_size, self.hidden)
        c0 = x.new_zeros(self.num_layers, batch_size, self.hidden)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, mask_len.cpu(), batch_first=True)
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        output = h_n[:, -1, :] # [B x hidden]
        output = self.output(output)
        prob = self.softmax(output) # no gumbel
        return prob