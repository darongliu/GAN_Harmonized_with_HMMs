import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.utils import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, phn_size, dis_emb_dim, hidden_dim1, hidden_dim2, use_second_conv=True, max_len=None):
        super().__init__()
        self.max_len = max_len
        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        self.use_second_conv = use_second_conv

        self.conv_1 = nn.ModuleList([
            nn.Conv1d(dis_emb_dim, hidden_dim1, 3, padding=1),
            nn.Conv1d(dis_emb_dim, hidden_dim1, 5, padding=2),
            nn.Conv1d(dis_emb_dim, hidden_dim1, 7, padding=3),
            nn.Conv1d(dis_emb_dim, hidden_dim1, 9, padding=4),
        ])
        self.lrelu_1 = nn.LeakyReLU()
        if use_second_conv:
            self.conv_2 = nn.ModuleList([
                nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
                nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
                nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
                nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            ])
            self.lrelu_2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        hidden_dim = hidden_dim2 if use_second_conv else hidden_dim1
        if max_len is not None:
            self.linear = nn.Linear(max_len*4*hidden_dim, 1)
        else:
            self.linear = nn.Linear(4*hidden_dim, 1)
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
    
    def forward(self, inputs, inputs_len=None):
        outputs = self.embedding(inputs)
        outputs = outputs.transpose(1, 2)
        outputs = torch.cat([conv(outputs) for conv in self.conv_1], dim=1)
        outputs = self.lrelu_1(outputs)
        if self.use_second_conv:
            outputs = torch.cat([conv(outputs) for conv in self.conv_2], dim=1)
            outputs = self.lrelu_2(outputs)
        outputs = outputs.transpose(1, 2)
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