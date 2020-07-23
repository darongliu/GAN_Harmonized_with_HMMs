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

    def sample_noise(self, x, eps=1e-20):
        U = torch.rand_like(x)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, x, temp=1, mask_len=None):
        """
        Inputs: sampled features
            x: (batch, timesteps, feature_size)
            mask_len: list of int, lengths

        Outputs:
            prob: (batch, timesteps, phn_size)
        """
        output = self.model(x)
        prob = self.softmax(output) # no gumbel

        gumbel_output = output + self.sample_noise(output)
        soft_prob = self.softmax(gumbel_output / temp)

        hard_prob = torch.nn.functional.one_hot(torch.max(soft_prob, dim=-1)[-1], soft_prob.shape[-1])
        hard_prob = hard_prob.cuda().float()
        hard_prob = (hard_prob - soft_prob).detach() + soft_prob

        if mask_len is not None:
            prob = masked_out(prob, mask_len)
            soft_prob = masked_out(soft_prob, mask_len)
            hard_prob = masked_out(hard_prob, mask_len)

        return prob, soft_prob, hard_prob

    def calc_seq_loss(self, x, y):
        x = x[:, :y.size(1)]
        output = self.model(x)
        prob = self.softmax(output)
        output = output.transpose(1,2)
        seq_loss = F.cross_entropy(output, y)
        return seq_loss, prob
