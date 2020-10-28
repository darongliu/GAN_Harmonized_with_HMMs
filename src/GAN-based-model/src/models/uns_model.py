import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import re
import sys
import random
import _pickle as pk
import numpy as np
from collections import defaultdict
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from src.data.dataLoader import get_data_loader, get_dev_data_loader
from src.models.gan_wrapper import GenWrapper, DisWrapper
from src.lib.utils import gen_real_sample, pad_sequence
from src.lib.metrics import frame_eval, per_eval
from src.lib.posterior_processing import stats_locations

import torch_optimizer as optim # from https://github.com/jettify/pytorch-optimizer.git

LOG_TEXT_NUM = 3
MAX_SEQLEN = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UnsModel(nn.Module):

    def __init__(self, config, use_posterior_bnd=False):
        super().__init__()
        cout_word = 'UNSUPERVISED MODEL: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.config = config
        self.use_posterior_bnd = use_posterior_bnd
        self.step = 0

        self.gen_model = GenWrapper(config.feat_dim,
                                    config.phn_size,
                                    config.gen_hidden_size).to(device)
        model_type = config['model_type']
        max_len = config.phn_max_length if config.use_maxlen else None
        self.dis_model = DisWrapper(phn_size=config.phn_size,
                                    max_len=max_len,
                                    model_type=model_type,
                                    use_posterior_bnd=use_posterior_bnd,
                                    **config[model_type]).to(device)

        if config.optimizer == 'radam':
            self.gen_optim = optim.RAdam(self.gen_model.parameters(),
                                         lr=config.gen_lr, betas=(0.5, 0.9))
            self.dis_optim = optim.RAdam(self.dis_model.parameters(),
                                         lr=config.dis_lr, betas=(0.5, 0.9))
        elif config.optimizer == 'sgd':
            self.gen_optim = torch.optim.SGD(self.gen_model.parameters(),
                                         lr=config.gen_lr, momentum=config.momentum)
            self.dis_optim = torch.optim.SGD(self.dis_model.parameters(),
                                         lr=config.dis_lr, momentum=config.momentum)
        else:
            self.gen_optim = torch.optim.Adam(self.gen_model.parameters(),
                                              lr=config.gen_lr, betas=(0.5, 0.9))
            self.dis_optim = torch.optim.Adam(self.dis_model.parameters(),
                                              lr=config.dis_lr, betas=(0.5, 0.9))

        def get_raising_scheduler(total_steps, start_raise_ratio, end_raise_ratio):
            start_point = round(total_steps * start_raise_ratio)
            end_point = round(total_steps * end_raise_ratio)
            raising_ratio = torch.arange(end_point - start_point).true_divide(end_point - start_point)
            fixed_zero = torch.zeros(start_point)
            fixed_full = torch.ones(total_steps - end_point)
            scheduling = torch.cat([fixed_zero, raising_ratio, fixed_full], dim=0).to(device)
            assert scheduling.size(0) == total_steps
            return scheduling

        self.frame_balance_schedular = None
        if hasattr(self.config, 'frame_balance_scheduler'):
            frame_balance_ratio_scheduler = get_raising_scheduler(self.config.step, *self.config.frame_balance_scheduler)
            self.frame_balance_schedular = frame_balance_ratio_scheduler * self.config.frame_balance_ratio

        sys.stdout.write('\b'*len(cout_word))
        cout_word = 'UNSUPERVISED MODEL: finish     '
        sys.stdout.write(cout_word+'\n')
        sys.stdout.flush()

    def forward(self, sample_feat, sample_len, target_idx, target_len, frame_temp,
                intra_diff_num, step, train_generator=True):
        sample_feat = sample_feat.to(device)
        sample_len = sample_len.to(device)
        target_idx = target_idx.to(device)
        target_len = target_len.to(device)

        prob, soft_prob, hard_prob = self.gen_model(sample_feat, frame_temp, sample_len) # masked
        sample_len_masks = torch.lt(torch.arange(sample_len.max()).to(device).unsqueeze(0), sample_len.unsqueeze(-1))

        if self.config.gan_gumbel == 'soft':
            fake_sample = soft_prob
        elif self.config.gan_gumbel == 'hard':
            fake_sample = hard_prob
        else:
            fake_sample = prob

        if self.config.intra_gumbel == 'soft':
            intra_sample = soft_prob
        elif self.config.intra_gumbel == 'hard':
            intra_sample = hard_prob
        else:
            intra_sample = prob

        
        real_sample = gen_real_sample(target_idx, target_len, self.config.phn_size).to(device) # masked one-hot vector
        balance_ratio = None if self.frame_balance_schedular is None else self.frame_balance_schedular[step-1]  # for step is 1 to total_steps

        if train_generator:
            g_loss, locations = self.dis_model.calc_g_loss(real_sample, target_len,
                                                           fake_sample, sample_len,
                                                           prob if self.use_posterior_bnd else None,
                                                           balance_ratio)
            
            if not self.use_posterior_bnd:
                batch_size = fake_sample.size(0) // 2
                intra_diff_num = intra_diff_num.to(device)
                g_seg_regu = self.gen_model.calc_intra_loss(intra_sample[:batch_size],
                                                            intra_sample[batch_size:],
                                                            intra_diff_num)
            else:
                estimated_phone_num = ((1 - (prob[:, :-1, :] * prob[:, 1:, :]).sum(dim=-1)) * sample_len_masks[:, :-1]).sum(dim=-1) + 1
                estimated_frame_phone_ratio = torch.true_divide(sample_len, estimated_phone_num).mean(dim=0, keepdim=True)
                g_seg_regu = F.l1_loss(estimated_frame_phone_ratio, torch.ones(1).to(device) * self.config.frame_phone_ratio)
                g_same_regu = ((1 - (prob[:, :-1, :] * prob[:, 1:, :]).sum(dim=-1)) * sample_len_masks[:, :-1]).sum(dim=-1).true_divide(sample_len).mean()
                valid_indices = sample_len_masks.nonzero(as_tuple=True)

                valid_prob = prob[valid_indices].view(-1, prob.size(-1))
                g_maxprob_regu = valid_prob.max(dim=-1).values.mean()
                g_entropy_regu = -((valid_prob + 1e-8).log() * valid_prob).sum(dim=-1).mean()

                avg_prob = valid_prob.mean(dim=0)
                g_avg_maxprob_regu = avg_prob.max(dim=-1).values.mean()
                g_avg_entropy_regu = -((avg_prob + 1e-8).log() * avg_prob).sum()
                
                locations_stats = stats_locations(locations, valid_indices)
                for i, (maxprob, entropy) in enumerate(zip(*locations_stats)):
                    locals()[f'g_neighbor_{i}_maxprob_regu'] = maxprob
                    locals()[f'g_neighbor_{i}_entropy_regu'] = entropy
                g_neighbor_avg_maxprob_regu = torch.stack(locations_stats[0], dim=0).mean()
                g_neighbor_avg_entropy_regu = torch.stack(locations_stats[1], dim=0).mean()
            
            vs = locals()
            regus = {key: vs[key] for key in list(vs.keys()) if 'regu' in key}
            return g_loss, regus, prob
        
        else:
            d_loss, d_gp_regu = self.dis_model.calc_d_loss(real_sample, target_len,
                                                           fake_sample, sample_len,
                                                           prob if self.use_posterior_bnd else None,
                                                           balance_ratio)
            vs = locals()
            regus = {key: vs[key] for key in list(vs.keys()) if 'regu' in key}
            return d_loss, regus


    def train(self, train_data_set, dev_data_set=None, aug=False):
        print ('TRAINING(unsupervised)...')
        self.log_writer = SummaryWriter(self.config.save_path)

        train_source, train_target = get_data_loader(train_data_set,
                                                     batch_size=self.config.batch_size,
                                                     repeat=self.config.repeat,
                                                     sample_range=self.config.sample_range,
                                                     use_posterior_bnd=self.use_posterior_bnd)
        train_source, train_target = iter(train_source), iter(train_target)

        max_fer = 100.0
        frame_temp = self.config.frame_temp

        self.gen_model.train()
        self.dis_model.train()

        logging = defaultdict(list)
        length_mask = torch.arange(MAX_SEQLEN).to(device)
        for _ in trange(self.config.step, dynamic_ncols=True):
            self.step += 1
            #if self.step == 8000: fram_temp = 0.8
            #if self.step == 12000: fram_temp = 0.7

            for _ in range(self.config.dis_iter):
                self.dis_optim.zero_grad()
                sample_feat, sample_len, intra_diff_num = next(train_source)
                target_idx, target_len = next(train_target)

                d_loss, d_regus = self.forward(
                    sample_feat, sample_len,
                    target_idx, target_len,
                    frame_temp, intra_diff_num,
                    self.step, train_generator=False
                )
                locals().update(d_regus)

                d_total_loss = d_loss
                for key in d_regus.keys():
                    if d_regus[key] is not None:
                        ratio = getattr(self.config, key.replace('regu', 'ratio'), 0)
                        d_total_loss  = d_total_loss + ratio * d_regus[key]
                d_total_loss.backward()

                d_clip_grad = nn.utils.clip_grad_norm_(self.dis_model.parameters(), 5.0)
                self.dis_optim.step()

            for _ in range(self.config.gen_iter):
                self.gen_optim.zero_grad()
                sample_feat, sample_len, intra_diff_num = next(train_source)
                target_idx, target_len = next(train_target)

                g_loss, g_regus, fake_probs = self.forward(
                    sample_feat, sample_len,
                    target_idx, target_len,
                    frame_temp, intra_diff_num,
                    self.step, train_generator=True
                )
                locals().update(g_regus)

                g_total_loss = g_loss
                for key in g_regus.keys():
                    if g_regus[key] is not None:
                        ratio = getattr(self.config, key.replace('regu', 'ratio'), 0)
                        g_total_loss  = g_total_loss + ratio * g_regus[key]
                g_total_loss.backward()

                g_clip_grad = nn.utils.clip_grad_norm_(self.gen_model.parameters(), 5.0)
                self.gen_optim.step()

            vs = locals()
            for key in list(vs.keys()):
                if re.fullmatch('(g|d)_.*(loss|regu|grad)', key) and vs[key] is not None:
                    logging[f'{key.split("_")[0]}/{key}'].append(vs[key].item() if type(vs[key]) is torch.Tensor else vs[key]) # accumulate the value
            
            if self.step % self.config.print_step == 0:
                # log scalars
                for key, values in logging.items():
                    self.log_writer.add_scalar(key, torch.FloatTensor(values).mean().item(), self.step) # logging the avg of several steps
                logging = defaultdict(list)

                # log the ratio: len(predicted fake phoneme sequence)/len(oracle fake phoneme sequence)
                fake_pred = fake_probs.detach().argmax(dim=-1)
                length_masks = torch.lt(length_mask[:max(sample_len)].unsqueeze(0), sample_len.to(device).unsqueeze(-1))
                fake_phone_num = ((fake_pred[:, :-1] != fake_pred[:, 1:]) * length_masks[:, :-1]).sum(dim=-1).float().mean().item() + 1
                self.log_writer.add_scalar('g/fake_real_phone_ratio', fake_phone_num / target_len.float().mean().item(), self.step)

            if self.step % self.config.eval_step == 0:
                step_fer = self.dev(dev_data_set)
                self.log_writer.add_scalar('fer', step_fer, self.step)
                
                if step_fer < max_fer: 
                    max_fer = step_fer
                    self.save_ckpt()
                
                self.gen_model.train()

        print ('='*80)

    def dev(self, dev_data_set):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.gen_model.eval()
        fers, fnums = 0, 0
        log_indices = random.sample(range(len(dev_source)), LOG_TEXT_NUM)
        for batch_idx, (feat, frame_label, length) in enumerate(dev_source):
            prob, soft_prob, hard_prob = self.gen_model(feat.to(device), mask_len=length)
            if self.config.output_gumbel == 'soft':
                prob = soft_prob
            elif self.config.output_gumbel == 'hard':
                prob = hard_prob
            prob = prob.detach().cpu()
            pred = prob.argmax(-1)
            frame_error, frame_num, _ = frame_eval(pred.numpy(), frame_label.numpy(), length)
            fers += frame_error
            fnums += frame_num

            if batch_idx in log_indices:
                real, fake = frame_label[0], pred[0]
                l = (real == -100).nonzero(as_tuple=False).view(-1)[0]
                text = (
                    '**REAL**  \n' + ' '.join([dev_data_set.idx2phn[idx] for idx in real[:l].tolist()]) + '  \n' +
                    '**FAKE**  \n' + ' '.join([dev_data_set.idx2phn[idx] for idx in fake[:l].tolist()])
                )
                sample_idx = log_indices.index(batch_idx)
                self.log_writer.add_text(f'sample_{sample_idx}', text, self.step)

        step_fer = fers / fnums * 100
        return step_fer

    def test(self, dev_data_set, file_path, fer_result_path=''):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.gen_model.eval()
        fers, fnums = 0, 0
        fers_39, fnums_39 = 0, 0
        pers_39, pnums_39 = 0, 0
        probs = []
        # calc all phn 39 and give index
        all_p39 = list(set(dev_data_set.phn_mapping.values()))
        phn392idx = dict(zip(all_p39, range(len(all_p39))))

        for feat, frame_label, length in dev_source:
            feat = pad_sequence(feat, max_len=self.config.feat_max_length)
            prob, soft_prob, hard_prob  = self.gen_model(feat.to(device), mask_len=length)
            if self.config.output_gumbel == 'soft':
                prob = soft_prob
            elif self.config.output_gumbel == 'hard':
                prob = hard_prob
            prob = prob.detach().cpu().numpy()
            pred = prob.argmax(-1)
            frame_label = frame_label.numpy()
            frame_error, frame_num, _ = frame_eval(pred, frame_label, length)

            probs.extend(prob)
            fers += frame_error
            fnums += frame_num

            # 39
            pred_39 = np.zeros_like(pred)
            l, w  = pred.shape
            for i in range(l):
                for j in range(w):
                    pred_39[i][j] = phn392idx[dev_data_set.phn_mapping[pred[i][j]]]

            frame_label_39 = np.zeros_like(frame_label)
            l, w  = frame_label.shape
            for i in range(l):
                for j in range(w):
                    if frame_label[i][j] != -100:
                        frame_label_39[i][j] = phn392idx[dev_data_set.phn_mapping[frame_label[i][j]]]
            frame_error_39, frame_num_39, _ = frame_eval(pred_39, frame_label_39, length)
            phone_error_39, phone_num_39, _ = per_eval(pred_39, frame_label_39, length)

            fers_39 += frame_error_39
            fnums_39 += frame_num_39
            pers_39 += phone_error_39
            pnums_39 += phone_num_39
        step_fer = fers / fnums * 100
        step_fer_39 = fers_39 / fnums_39 * 100
        step_per_39 = pers_39 / pnums_39 * 100
        print('fer:', step_fer)
        print('fer on phn 39: ', step_fer_39)
        print('per on phn 39: ', step_per_39)
        if fer_result_path != '':
            with open(fer_result_path, 'w') as f:
                f.write('frame error rate: '+str(step_fer)+'\n')
                f.write('frame error rate on phn 39: '+str(step_fer_39)+'\n')
                f.write('phone error rate on phn 39: '+str(step_per_39)+'\n')
        print(np.array(probs).shape)
        pk.dump(np.array(probs), open(file_path, 'wb'))

    def test_posterior(self, dev_data_set, file_path):
        # do not output or write anything
        # used to calc fer, per for already generated posterior
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        fers, fnums = 0, 0
        fers_39, fnums_39 = 0, 0
        pers_39, pnums_39 = 0, 0

        all_probs = pk.load(open(file_path, 'rb'))

        # calc all phn 39 and give index
        all_p39 = list(set(dev_data_set.phn_mapping.values()))
        phn392idx = dict(zip(all_p39, range(len(all_p39))))

        count = 0
        for _, frame_label, length in dev_source:
            prob = all_probs[count:count+len(length)]
            count += len(length)
            pred = prob.argmax(-1)
            frame_label = frame_label.numpy()
            frame_error, frame_num, _ = frame_eval(pred, frame_label, length)

            fers += frame_error
            fnums += frame_num

            # 39
            pred_39 = np.zeros_like(pred)
            l, w  = pred.shape
            for i in range(l):
                for j in range(w):
                    pred_39[i][j] = phn392idx[dev_data_set.phn_mapping[pred[i][j]]]

            frame_label_39 = np.zeros_like(frame_label)
            l, w  = frame_label.shape
            for i in range(l):
                for j in range(w):
                    if frame_label[i][j] != -100:
                        frame_label_39[i][j] = phn392idx[dev_data_set.phn_mapping[frame_label[i][j]]]
            frame_error_39, frame_num_39, _ = frame_eval(pred_39, frame_label_39, length)
            phone_error_39, phone_num_39, _ = per_eval(pred_39, frame_label_39, length)

            fers_39 += frame_error_39
            fnums_39 += frame_num_39
            pers_39 += phone_error_39
            pnums_39 += phone_num_39
        step_fer = fers / fnums * 100
        step_fer_39 = fers_39 / fnums_39 * 100
        step_per_39 = pers_39 / pnums_39 * 100
        print('fer:', step_fer)
        print('fer on phn 39: ', step_fer_39)
        print('per on phn 39: ', step_per_39)

    def save_ckpt(self):
        ckpt_path = os.path.join(self.config.save_path, "ckpt_{}.pth".format(self.step))
        torch.save({
            "gen_model": self.gen_model.state_dict(),
            "dis_model": self.dis_model.state_dict(),
            "gen_optim": self.gen_optim.state_dict(),
            "dis_optim": self.dis_optim.state_dict(),
            "step": self.step
        }, ckpt_path)

    def load_ckpt(self, load_path):
        print('\033[K' + "[INFO]", "Load model from: " + load_path)
        ckpt = torch.load(load_path)
        self.gen_model.load_state_dict(ckpt['gen_model'])
        self.dis_model.load_state_dict(ckpt['dis_model'])
        self.gen_optim.load_state_dict(ckpt['gen_optim'])
        self.dis_optim.load_state_dict(ckpt['dis_optim'])
        self.step = ckpt['step']

    def write_log(self, val_name, val_dict):
        self.log_writer.add_scalars(val_name, val_dict, self.step)

