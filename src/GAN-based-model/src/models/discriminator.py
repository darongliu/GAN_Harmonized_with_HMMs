import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.utils import get_mask_from_lengths
from src.lib.modules import Conv1dWrapper
from src.lib.posterior_processing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SECOND_CONV_NUM = 4
FIRST_CONV_MIN_KERNEL = 3
SECOND_CONV_KERNEL = 3

class ResBlock(nn.Module):
    def __init__(self, idim, odim, kernel, use_batchnorm, res=True):
        super().__init__()
        self.res = res
        self.lrelu = nn.LeakyReLU()

        if use_batchnorm:
            self.res_block = nn.Sequential(
                nn.Conv1d(idim, odim, kernel, padding=(kernel-1)//2),
                nn.BatchNorm1d(odim), 
                nn.LeakyReLU(),
                nn.Conv1d(odim, odim, kernel, padding=(kernel-1)//2),
                nn.BatchNorm1d(odim)
            )
        else:
            self.res_block = nn.Sequential(
                nn.Conv1d(idim, odim, kernel, padding=(kernel-1)//2),
                nn.LeakyReLU(),
                nn.Conv1d(odim, odim, kernel, padding=(kernel-1)//2),
            )

    def forward(self, inputs):
        outputs = self.res_block(inputs)
        if self.res:
            outputs = outputs + inputs
        return self.lrelu(outputs)


class Discriminator(nn.Module):
    """ Not used.
    Arguments:
        dim: channels
    """
    def __init__(self, phn_size, dis_emb_dim, hidden_dim, num_layers, kernel, use_batchnorm=False, max_len=None):
        super().__init__()
        self.max_len = max_len
        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)

        self.blocks = nn.Sequential(
            ResBlock(dis_emb_dim, hidden_dim, kernel, use_batchnorm=use_batchnorm, res=False),
            *[ResBlock(hidden_dim,  hidden_dim, kernel, use_batchnorm=use_batchnorm) for i in range(num_layers-1)])
        self.flatten = nn.Flatten()

        if max_len is not None:
            self.linear = nn.Linear(max_len*hidden_dim, 1)
        else:
            self.linear = nn.Linear(hidden_dim, 1)
        self._spec_init()

    def _spec_init(self):
        pass
                
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
        outputs = outputs.transpose(1, 2)
        outputs = self.blocks(outputs)
        outputs = outputs.transpose(1, 2)
        if self.max_len is not None:
            # (B, T, D) -> (B, T*D)
            outputs = self.flatten(outputs)
        else:
            # (B, T, D) -> (B, D)
            outputs = self.mask_pool(outputs, inputs_len)
        outputs = self.linear(outputs)
        return outputs

class WeakDiscriminator(nn.Module):
    """ Two layers with second layer very few channels
    Arguments:
        dim: channels
    """
    def __init__(self, phn_size, dis_emb_dim, hidden_dim1, hidden_dim2, use_second_conv=True, max_len=None,
                 max_conv_bank_kernel=9, window_per_phn=3, local_discriminate=False, eps=1e-8, **kwargs):
        # max_conv_kernel: means conv bank kernel is 3,5,...max_conv_kernel
        super().__init__()
        self.max_len = max_len
        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        self.use_second_conv = use_second_conv
        self.max_conv_bank_kernel = max_conv_bank_kernel
        self.window_per_phn = window_per_phn
        self.local_discriminate = local_discriminate
        self.eps = eps

        assert(max_conv_bank_kernel % 2 == 1)
        self.conv_1 = nn.ModuleList([Conv1dWrapper(dis_emb_dim, hidden_dim1, k) for k in range(FIRST_CONV_MIN_KERNEL, max_conv_bank_kernel + 1, 2)])
        self.lrelu_1 = nn.LeakyReLU()
        if use_second_conv:
            self.conv_2 = nn.ModuleList([Conv1dWrapper(len(self.conv_1) * hidden_dim1, hidden_dim2, SECOND_CONV_KERNEL) for i in range(SECOND_CONV_NUM)])
            self.lrelu_2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        if not self.local_discriminate:
            hidden_dim = hidden_dim2 if use_second_conv else hidden_dim1
            if max_len is not None:
                    self.linear = nn.Linear(max_len * SECOND_CONV_NUM * hidden_dim, 1)
            else:
                    self.linear = nn.Linear(SECOND_CONV_NUM * hidden_dim, 1)
        else:
            hidden_dim = SECOND_CONV_NUM * hidden_dim2 if use_second_conv else len(self.conv_1) * hidden_dim1
            self.linear = nn.Linear(hidden_dim, 1)        

        self._spec_init()

    def _spec_init(self):
        for name, para in self.named_parameters():
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
    
    def forward(self, inputs, inputs_len=None, posteriors=None, balance_ratio=None):
        outputs = self.embedding(inputs)

        if posteriors is not None:
            locations = posteriors_to_locations(posteriors, self.max_conv_bank_kernel, self.window_per_phn)
            # locations: (batch_size, seqlen, kernel_size, window_size)
            outputs = locations_to_neighborhood(outputs, locations)
            # outputs: (batch_size, seqlen, kernel_size, class_num)

        outputs = torch.cat([conv(outputs) for conv in self.conv_1], dim=-1)
        outputs = self.lrelu_1(outputs)
        # outputs: (batch_size, seqlen, featdim)
        
        if self.use_second_conv:

            if posteriors is not None:
                outputs = locations_to_neighborhood(outputs, locations, SECOND_CONV_KERNEL)

            outputs = torch.cat([conv(outputs) for conv in self.conv_2], dim=-1)
            outputs = self.lrelu_2(outputs)
            # outputs: (batch_size, seqlen, featdim)
        
        if not self.local_discriminate:
            if self.max_len is not None:
                # (B, T, D) -> (B, T*D)
                outputs = self.flatten(outputs)
            else:
                # (B, T, D) -> (B, D)
                outputs = self.mask_pool(outputs, inputs_len)
            outputs = self.linear(outputs).squeeze(-1)
        else:
            scores = self.linear(outputs).squeeze(-1)
            # scores: (batch_size, seqlen)

            length_masks = torch.lt(torch.arange(inputs_len.max().item()).unsqueeze(0), inputs_len.unsqueeze(1)).to(inputs.device)
            scores = scores * length_masks

            if posteriors is not None and balance_ratio is not None:
                # reweight scores according to repeated frame num
                abs_positions = torch.arange(scores.size(1)).to(scores.device).unsqueeze(0).expand_as(scores).unsqueeze(-1)
                kernel3_positions = locations_to_neighborhood(abs_positions.float(), locations, 3, 'replicate').squeeze(-1)
                left_position, _, right_position = kernel3_positions.chunk(3, dim=-1)
                phone_interval = torch.max(right_position - left_position, scores.new_ones(1)).squeeze(-1)
                scores = scores / (1 + (phone_interval - 1) * balance_ratio)

            outputs = scores.sum(dim=-1) / inputs_len.to(inputs.device)
        return outputs


class LocalDiscriminator(nn.Module):
    """ Two layers with second layer very few channels
    Arguments:
        dim: channels
    """
    def __init__(self, phn_size, dis_emb_dim, hidden_dim1, hidden_dim2, use_second_conv=True, max_len=None,
                 max_conv_bank_kernel=9, window_per_phn=3, gb_tau=0.5, gb_hard=True, eps=1e-8, **kwargs):
        # max_conv_kernel: means conv bank kernel is 3,5,...max_conv_kernel
        super().__init__()
        self.max_len = max_len
        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        self.use_second_conv = use_second_conv
        self.max_conv_bank_kernel = max_conv_bank_kernel
        self.window_per_phn = window_per_phn
        self.gb_tau = gb_tau
        self.gb_hard = gb_hard
        self.eps = eps

        assert(max_conv_bank_kernel % 2 == 1)
        self.conv_1 = nn.ModuleList([Conv1dWrapper(dis_emb_dim, hidden_dim1, k, SECOND_CONV_KERNEL if use_second_conv else 1) 
                                    for k in range(FIRST_CONV_MIN_KERNEL, max_conv_bank_kernel + 1, 2)])
        self.lrelu_1 = nn.LeakyReLU()
        if use_second_conv:
            self.conv_2 = nn.ModuleList([Conv1dWrapper(len(self.conv_1) * hidden_dim1, hidden_dim2, SECOND_CONV_KERNEL) for i in range(SECOND_CONV_NUM)])
            self.lrelu_2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        hidden_dim = SECOND_CONV_NUM * hidden_dim2 if use_second_conv else len(self.conv_1) * hidden_dim1
        self.linear = nn.Linear(hidden_dim, 1)        

        self._spec_init()

    def _spec_init(self):
        for name, para in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(para.data)
            elif 'bias' in name:
                nn.init.zeros_(para.data)

    def embedding(self, x):
        return x @ self.emb_bag.weight

    def forward(self, inputs, inputs_len=None, posteriors=None, balance_ratio=None):
        if posteriors is not None:
            subseqlen = self.max_conv_bank_kernel + 2 if self.use_second_conv else 0
            locations = posteriors_to_locations(posteriors, subseqlen, self.window_per_phn)
            # locations: (batch_size, seqlen, kernel_size, window_size)
            inputs = locations_to_neighborhood(posteriors, locations)
            # inputs: (batch_size, seqlen, kernel_size, class_num)
            inputs = F.gumbel_softmax((inputs + self.eps).log(), self.gb_tau, self.gb_hard, dim=-1)

        outputs = self.embedding(inputs)
        outputs = torch.cat([conv(outputs) for conv in self.conv_1], dim=-1)
        outputs = self.lrelu_1(outputs)
        # outputs: (batch_size, seqlen, sliding_window_num, featdim)
        
        if self.use_second_conv:
            outputs = torch.cat([conv(outputs) for conv in self.conv_2], dim=-1)
            outputs = self.lrelu_2(outputs)
            # outputs: (batch_size, seqlen, featdim)
        
        scores = self.linear(outputs).squeeze(-1)
        # scores: (batch_size, seqlen)

        length_masks = torch.lt(torch.arange(inputs_len.max().item()).unsqueeze(0), inputs_len.unsqueeze(1)).to(inputs.device)
        scores = scores * length_masks

        if posteriors is not None and balance_ratio is not None:
            # reweight scores according to repeated frame num
            abs_positions = torch.arange(scores.size(1)).to(scores.device).unsqueeze(0).expand_as(scores).unsqueeze(-1)
            kernel3_positions = locations_to_neighborhood(abs_positions.float(), locations, 3, 'replicate').squeeze(-1)
            left_position, _, right_position = kernel3_positions.chunk(3, dim=-1)
            phone_interval = torch.max(right_position - left_position, scores.new_ones(1)).squeeze(-1)
            scores = scores / (1 + (phone_interval - 1) * balance_ratio)

        outputs = scores.sum(dim=-1) / inputs_len.to(inputs.device)
        return outputs


if __name__ == '__main__':
    import torch
    from src.models.discriminator import WeakDiscriminator, Discriminator
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
    kernel = 3
    use_batchnorm = False
    model  = Discriminator(phn_size, dis_emb_dim, hidden_dim, num_layers, kernel, use_batchnorm=use_batchnorm, max_len=T).cuda()

    result = model(feat, inputs_len=feat_length)
    print(result)

    dis_emb_dim = 10
    hidden_dim = 15
    use_second_conv=False
    model  = WeakDiscriminator(phn_size, dis_emb_dim, hidden_dim, hidden_dim, use_second_conv=use_second_conv, max_len=T).cuda()

    result = model(feat, inputs_len=feat_length)
    print(result)