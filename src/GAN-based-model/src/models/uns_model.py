import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import _pickle as pk
import numpy as np
from collections import defaultdict
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from src.data.dataLoader import get_data_loader, get_dev_data_loader
from src.models.gan_wrapper import GenWrapper, DisWrapper
from src.lib.utils import gen_real_sample, pad_sequence
from src.lib.metrics import frame_eval, per_eval

import torch_optimizer as optim # from https://github.com/jettify/pytorch-optimizer.git

LOG_TEXT_NUM = 5
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

        prob, soft_prob, hard_prob = self.gen_model(sample_feat, frame_temp, sample_len)

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

        
        real_sample = gen_real_sample(target_idx, target_len, self.config.phn_size).to(device)
        balance_ratio = None if self.frame_balance_schedular is None else self.frame_balance_schedular[step]

        if train_generator:
            g_loss = self.dis_model.calc_g_loss(real_sample, target_len,
                                                fake_sample, sample_len,
                                                prob if self.use_posterior_bnd else None,
                                                balance_ratio)
            seg_loss = None
            same_loss = None
            
            if not self.use_posterior_bnd:
                batch_size = fake_sample.size(0) // 2
                intra_diff_num = intra_diff_num.to(device)
                seg_loss = self.gen_model.calc_intra_loss(intra_sample[:batch_size],
                                                          intra_sample[batch_size:],
                                                          intra_diff_num)
            else:
                estimated_phone_num = (1 - (prob[:, :-1, :] * prob[:, 1:, :]).sum(dim=-1)).sum(dim=-1) + 1
                estimated_frame_phone_ratio = torch.true_divide(sample_len.to(device), estimated_phone_num).mean(dim=0, keepdim=True)
                seg_loss = F.l1_loss(estimated_frame_phone_ratio, torch.ones(1).to(device) * self.config.frame_phone_ratio)
                same_loss = (1 - (prob[:, :-1, :] * prob[:, 1:, :]).sum(dim=-1)).mean()
            
            return g_loss, seg_loss, same_loss, fake_sample
        
        else:
            d_loss, gp_loss = self.dis_model.calc_d_loss(real_sample, target_len,
                                                         fake_sample, sample_len,
                                                         prob if self.use_posterior_bnd else None,
                                                         balance_ratio)
            
            return d_loss, gp_loss


    def train(self, train_data_set, dev_data_set=None, aug=False):
        print ('TRAINING(unsupervised)...')
        self.log_writer = SummaryWriter(self.config.save_path)

        train_source, train_target = get_data_loader(train_data_set,
                                                     batch_size=self.config.batch_size,
                                                     repeat=self.config.repeat,
                                                     use_posterior_bnd=self.use_posterior_bnd)
        train_source, train_target = iter(train_source), iter(train_target)

        max_fer = 100.0
        frame_temp = self.config.frame_temp

        self.gen_model.train()
        self.dis_model.train()

        logging = defaultdict(list)
        for _ in trange(self.config.step, dynamic_ncols=True):
            self.step += 1
            #if self.step == 8000: fram_temp = 0.8
            #if self.step == 12000: fram_temp = 0.7

            for _ in range(self.config.dis_iter):
                self.dis_optim.zero_grad()
                sample_feat, sample_len, intra_diff_num = next(train_source)
                target_idx, target_len = next(train_target)

                dis_loss, dis_gp_loss = self.forward(sample_feat, sample_len,
                                                     target_idx, target_len,
                                                     frame_temp, intra_diff_num,
                                                     self.step, train_generator=False)
                dis_total_loss = dis_loss
                dis_total_loss = dis_total_loss + (0 if dis_gp_loss is None else self.config.penalty_ratio * dis_gp_loss)
                dis_total_loss.backward()

                dis_clip_grad = nn.utils.clip_grad_norm_(self.dis_model.parameters(), 5.0)
                self.dis_optim.step()

            for _ in range(self.config.gen_iter):
                self.gen_optim.zero_grad()
                sample_feat, sample_len, intra_diff_num = next(train_source)
                target_idx, target_len = next(train_target)

                gen_loss, gen_seg_loss, gen_same_loss, fake_sample = self.forward(sample_feat, sample_len,
                                                                                  target_idx, target_len,
                                                                                  frame_temp, intra_diff_num,
                                                                                  self.step, train_generator=True)
                gen_total_loss = gen_loss
                gen_total_loss = gen_total_loss + (0 if gen_seg_loss is None else self.config.seg_loss_ratio * gen_seg_loss)
                gen_total_loss = gen_total_loss + (0 if gen_same_loss is None else self.config.same_loss_ratio * gen_same_loss)
                gen_total_loss.backward()

                gen_clip_grad = nn.utils.clip_grad_norm_(self.gen_model.parameters(), 5.0)
                self.gen_optim.step()

            vs = locals()
            for name in list(vs.keys()):
                if ('loss' in name or 'grad' in name) and vs[name] is not None:
                    logging[f'{name.split("_")[0]}/{name}'].append(vs[name].item() if type(vs[name]) is torch.Tensor else vs[name])
            
            if self.step % self.config.print_step == 0:
                # log scalars
                for name, values in logging.items():
                    self.log_writer.add_scalar(name, torch.FloatTensor(values).mean().item(), self.step)
                logging = defaultdict(list)

            if self.step % self.config.eval_step == 0:
                step_fer = self.dev(dev_data_set)
                self.log_writer.add_scalar('fer', step_fer, self.step)
                
                if step_fer < max_fer: 
                    max_fer = step_fer
                    self.save_ckpt()
                
                self.gen_model.train()

                # log fake frame-sequences
                all_boundaries = []
                for idx, (fake, fake_len) in enumerate(zip(fake_sample.detach().cpu()[:LOG_TEXT_NUM], sample_len.detach().cpu()[:LOG_TEXT_NUM])):
                    fake_seq = fake[:fake_len].argmax(dim=-1)
                    all_boundaries.append(torch.nonzero((fake_seq[1:] != fake_seq[:-1]).long(), as_tuple=True)[0])
                    self.log_writer.add_text(f'fake_sample_{idx}', ' '.join([train_data_set.idx2phn[idx] for idx in fake_seq.tolist()]), self.step)
                fake_phone_num = torch.FloatTensor([len(bs) + 1 for bs in all_boundaries]).mean().item()
                self.log_writer.add_scalar('gen/fake_real_phone_ratio', fake_phone_num / target_len.float().mean().item(), self.step)

        print ('='*80)

    def dev(self, dev_data_set):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.gen_model.eval()
        fers, fnums = 0, 0
        for feat, frame_label, length in dev_source:
            prob, soft_prob, hard_prob  = self.gen_model(feat.to(device), mask_len=length)
            if self.config.output_gumbel == 'soft':
                prob = soft_prob
            elif self.config.output_gumbel == 'hard':
                prob = hard_prob
            prob = prob.detach().cpu().numpy()
            pred = prob.argmax(-1)
            frame_label = frame_label.numpy()
            frame_error, frame_num, _ = frame_eval(pred, frame_label, length)
            fers += frame_error
            fnums += frame_num
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

