import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discriminator import WeakDiscriminator, Discriminator, LocalDiscriminator
from src.models.lstm_discriminator import LSTMDiscriminator
from src.models.transformer_discriminator import TransformerDiscriminator
from src.models.generator import Frame2Phn
from src.lib.utils import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DisWrapper(nn.Module):
    def __init__(self, phn_size, max_len, model_type, use_posterior_bnd=False, **model_config):
        super().__init__()
        self.model_type = model_type
        self.model_config = model_config
        if model_type == 'cnn':
            self.model =     Discriminator(phn_size=phn_size, **model_config, max_len=max_len)
        elif model_type == 'lstm':
            self.model = LSTMDiscriminator(phn_size=phn_size, **model_config, max_len=max_len)
        elif model_type == 'transformer':
            self.model = TransformerDiscriminator(phn_size=phn_size, **model_config, max_len=max_len)
        elif model_type == 'weak_cnn':
            self.model = WeakDiscriminator(phn_size=phn_size, local_discriminate=use_posterior_bnd, **model_config, max_len=max_len)
        elif model_type == 'local_cnn':
            self.model = LocalDiscriminator(phn_size=phn_size, local_discriminate=use_posterior_bnd, **model_config, max_len=max_len)
        else:
            raise NotImplementedError

    def calc_gp(self, real, real_len, fake, fake_len, prob=None, balance_ratio=None):
        """
        Inputs:
            real, fake: torch.tensor, padded sequence
            real_len, fake_len: torch.LongTensor
        """
        batch_size = min(real.size(0), fake.size(0))
        cut_len = min(real.size(1), fake.size(1))
        real, real_len = real[:batch_size, :cut_len], real_len[:batch_size]
        fake, fake_len = fake[:batch_size, :cut_len], fake_len[:batch_size]

        size = [batch_size] + [1] * (real.dim()-1)
        alpha = torch.rand(size).to(device)

        inter = real + alpha * (fake - real)
        inter_prob = None
        if prob is not None:
            inter_prob = real + alpha * (prob - real)

        inter_len = torch.stack([real_len, fake_len]).min(0)[0]
        if self.model_type == 'lstm':
            inter, inter_len = self._sort_sequences(inter, inter_len)

        inter_pred = self.model(inter, inter_len, inter_prob, balance_ratio)

        gp_target = inter
        if self.model_config.get('gp_target') == 'posterior':
            assert inter_prob is not None
            gp_target = inter_prob

        gradients = torch.autograd.grad(outputs=inter_pred, inputs=gp_target,
            grad_outputs=torch.ones(inter_pred.size()).to(device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calc_phone_posterior_gp(self, real, real_len, fake, fake_len, prob, balance_ratio=None):
        if self.model_config.get('gp_level') == 'phone':
            return self.model.calc_gp(real, real_len, fake, fake_len, prob, balance_ratio)
        else:
            frames_per_phone = max(round(fake_len.max().item() / real_len.max().item()), 1)
            expanded_real = real.unsqueeze(2).expand(-1, -1, frames_per_phone, -1).reshape(real.size(0), -1, real.size(-1))
            estimated_phone_len = real_len * frames_per_phone
            min_len = torch.min(estimated_phone_len.long(), fake_len.long())
            return self.calc_gp(expanded_real[:, :min_len.max()], min_len,
                                fake[:, :min_len.max()], min_len,
                                prob[:, :min_len.max()], balance_ratio)

    def calc_g_loss(self, real, real_len, fake, fake_len, prob=None, balance_ratio=None):
        """
        Inputs:
            real: list of (len, phn_size)
            fake: list of (len, phn_size)
        """
        if self.model.max_len is not None and prob is None:
            real = pad_sequence(real, max_len=self.model.max_len)
            fake = pad_sequence(fake, max_len=self.model.max_len)
            real_len = torch.clamp(real_len, 0, self.model.max_len)
            fake_len = torch.clamp(fake_len, 0, self.model.max_len)

        if self.model_type == 'lstm':
            real, real_len = self._sort_sequences(real, real_len)
            fake, fake_len = self._sort_sequences(fake, fake_len)

        real_pred = self.model(real, real_len)
        if prob is None:
            fake_pred = self.model(fake, fake_len)
        else:
            fake_pred = self.model(fake, fake_len, prob, balance_ratio)

        g_loss = real_pred.mean() - fake_pred.mean()
        return g_loss

    def calc_d_loss(self, real, real_len, fake, fake_len, prob=None, balance_ratio=None):
        """
        Inputs:
            real: list of (len, phn_size)
            fake: list of (len, phn_size)
        """
        if self.model.max_len is not None and prob is None:
            real = pad_sequence(real, max_len=self.model.max_len)
            fake = pad_sequence(fake, max_len=self.model.max_len)
            real_len = torch.clamp(real_len, 0, self.model.max_len)
            fake_len = torch.clamp(fake_len, 0, self.model.max_len)

        if self.model_type == 'lstm':
            real, real_len = self._sort_sequences(real, real_len)
            fake, fake_len = self._sort_sequences(fake, fake_len)

        real_pred = self.model(real, real_len)
        if prob is None:
            fake_pred = self.model(fake, fake_len)
            gp_loss = self.calc_gp(real, real_len, fake, fake_len)
        else:
            fake_pred = self.model(fake, fake_len, prob, balance_ratio)
            gp_loss = self.calc_phone_posterior_gp(real, real_len, fake, fake_len, prob, balance_ratio)

        d_loss = fake_pred.mean() - real_pred.mean()
        return d_loss, gp_loss

    def _sort_sequences(self, inputs, lengths):
        """sort_sequences
        Sort sequences according to lengths descendingly.

        :param inputs (Tensor): input sequences, size [B, T, D]
        :param lengths (Tensor): length of each sequence, size [B]
        """
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        return inputs[sorted_idx], lengths_sorted


class GenWrapper(nn.Module):
    """ Input is already sampled repeatedly (2*batch*repeat, ...). """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Frame2Phn(*args, **kwargs)
        self._spec_init()

    def _spec_init(self):
        for name, para in self.model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(para.data)
            elif 'bias' in name:
                nn.init.zeros_(para.data)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # count loss with softmax_logits
    # https://github.com/gary083/GAN_Harmonized_with_HMMs/blob/master/src/GAN-based-model/uns_model.py#L35
    # https://github.com/gary083/GAN_Harmonized_with_HMMs/blob/master/src/GAN-based-model/lib/module.py#L175
    def calc_intra_loss(self, prob1, prob2, intra_diff_num):
        intra = ((prob1 - prob2) ** 2).sum() / intra_diff_num.sum()
        return intra
