"""
@Time: 2019/12/17 9:26
@Author: Wang Peiyi
@Site :
@File : sorting_model.py.py
"""
import torch
import torch.nn as nn
from config import opt

from utils import sinkhorn_ops


class SortingModel(nn.Module):

    def __init__(self, opt):
        super(SortingModel, self).__init__()
        self.opt = opt
        self.net = nn.Sequential(
            ###################################################################
            # the function here: 实现原文《LEARNING LATENT PERMUTATIONS WITH GUMBEL- SINKHORN NETWORKS》Figure 1的g1函数
            ##########################################################################################################
            nn.Linear(1, self.opt.n_units),
            nn.ReLU(),
            nn.Dropout(self.opt.drop_prob),

            ##########################################################################################################
            # the function here: 实现原文Figure 1中的g2函数, 输出为【g(X^, θ)】，
            # 它是(batch_size * n_numbers, n_numbers)的Tensor
            ##########################################################################################################
            nn.Linear(self.opt.n_units, self.opt.n_numbers),
            nn.Dropout(self.opt.drop_prob)
        )

        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data)
                nn.init.xavier_normal(m.weight.data)
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()

    def compute_l2_soft_losses(self,
                               soft_pers_inf,
                               _ordered_tiled,
                               _random_tiled):
        """计算重构的l2损失
        @param soft_pers_inf:   一个(batch_size, n_sample, n, n)的tensor， 对应着模型
                                重构的ordered到random的置换矩阵
        @param _ordered_tiled:  一个(n_sample*batch_size, n)的Tensor, 对应着扩展后的有序数据
        @param _random_tiled:   一个(n_sample*batch_size, n)的Tensor, 对应着扩展后的无需数据

        @return: 一个floatTensor值，对应着损失
        """
        n = soft_pers_inf.size(2)
        inv_soft_perms = soft_pers_inf.permute(
            0, 1, 3, 2)    # (batch_size, n_sample, n, n)
        inv_soft_perms = inv_soft_perms.permute(
            1, 0, 2, 3)   # (n_sample, batch_Size, n, n)
        # (n_sample*batch_size, n, n)
        inv_soft_perms_flat = inv_soft_perms.view(-1, n, n)

        # (n_sample*batch_size, n, 1)
        _ordered_tiled = _ordered_tiled.view(-1, n, 1)
        _random_tiled = _random_tiled.view(-1, n, 1)

        rec_titled = torch.matmul(inv_soft_perms_flat, _random_tiled)
        loss_l2 = torch.mean(torch.pow(_ordered_tiled - rec_titled, 2))

        return loss_l2

    def compute_hard_losses(self,
                            log_alpha_w_noise,
                            _random_tiled,
                            _hard_perms,
                            _ordered_tiled):
        """

        @param log_alpha_w_noise: 一个(batch_size, n_samples, N, N)的Tensor
                                (加入的gumbel分布噪声
        @return: 所有都是一个float值
                l1_diff: l1距离
                l2sh_diff: l2平方距离
                prob_wrong: 错误率
                prop_any_wrong: 平均一个序列有几个排错了
                kendall_tau: 平均kendall相关系数
        """
        log_alpha_w_noise = log_alpha_w_noise.permute(
            1, 0, 2, 3)  # (n_sample, batch_size, n, n)
        # (n_sample * batch_size, n, n)
        log_alpha_w_noise_flat = log_alpha_w_noise.view(
            -1, self.opt.n_numbers, self.opt.n_numbers)

        hard_perms_inf = sinkhorn_ops.matching(
            log_alpha_w_noise_flat)  # (n_sample * batch_size, n)
        inverse_hard_perms_inf = sinkhorn_ops.invert_listperm(
            hard_perms_inf)  # (n_sample * batch_size, n)
        hard_perms_tiled = _hard_perms.repeat(
            self.opt.samples_per_num,
            1)  # (n_sample * batch_size, n)
        ordered_inf_tiled = sinkhorn_ops.permute_batch_list(
            _random_tiled, inverse_hard_perms_inf)  # (n_samle * batch_size, n)

        l1_diff = torch.mean(torch.abs(_ordered_tiled - ordered_inf_tiled))
        l2sh_diff = torch.mean(
            torch.pow(
                _ordered_tiled -
                ordered_inf_tiled,
                2))
        diff_perms = torch.abs(hard_perms_tiled - inverse_hard_perms_inf)
        prop_wrong = torch.mean(torch.sign(diff_perms).float())   # (所有的错误率)
        prop_any_worng = torch.mean(
            torch.sum(
                torch.sign(diff_perms).float(),
                1))  # 单个样本中的错误数量
        kendall_tau = torch.mean(
            sinkhorn_ops.kendall_tau(
                hard_perms_tiled,
                inverse_hard_perms_inf))  # (n_sample * batch_size, 1)
        return l1_diff, l2sh_diff, prop_wrong, prop_any_worng, kendall_tau

    def forward(self, datas, is_train=True):
        _ordered, _random, _hard_perms, _ordered_tiled, _random_tiled = datas
        random_flattened = _random.view(-1, 1)  # (batch_size * n_numbers, 1)
        # (batch_size * n_numbers, n_numbers)
        log_alpha = self.net(random_flattened)
        # (batch_size, n_numbers, n_numbers)
        log_alpha = log_alpha.view(-1, self.opt.n_numbers, self.opt.n_numbers)
        _soft_pers_inf, _log_alpha_w_noise = sinkhorn_ops.gumbel_sinkhorn(log_alpha,
                                                                          self.opt.temperature,
                                                                          self.opt.samples_per_num,
                                                                          self.opt.noise_factor,
                                                                          self.opt.n_iter_sinkhorn,
                                                                          squeeze=False)
        # _soft_pers_inf: (batch_size, n_sample, n_number, n_number)
        # _ordered_tiled: (n_sample * batch_size, n_number)
        # _random_tiled: (n_sample * batch_size, n_number)
        if is_train:
            return self.compute_l2_soft_losses(_soft_pers_inf,
                                               _ordered_tiled,
                                               _random_tiled)

        return self.compute_hard_losses(_log_alpha_w_noise,
                                        _random_tiled,
                                        _hard_perms,
                                        _ordered_tiled)

