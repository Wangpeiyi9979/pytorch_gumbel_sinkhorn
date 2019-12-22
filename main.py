"""
@Time: 2019/12/17 10:39
@Author: Wang Peiyi
@Site : 
@File : main.py
"""

##########################################################################################################
# TODO: 执行以下命令:python main go --epoches=2700 --noise_factor=0.0,其在训练集上
#       最后效果竟然比执行:python main go --epoches=500 --noise_factor=0.0差，不明白
#       是为什么。
##########################################################################################################

import fire
import torch
import numpy as np
import torch.optim as optim

from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR

from config import opt
from sorting import SortingModel
from utils import sinkhorn_ops

def setup_seed(seed):
    """Funciton:固定随机种子，使得模型每次结果唯一
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, datas, optimizer, opt):
    model.train()
    loss = model(datas, is_train=True)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
    optimizer.step()
    return loss

def test(model, datas):
    model.eval()
    ##########################################################################################################
    # TODO: 弄明白噪声影响。这里，噪声会严重影响表现，如果噪声依然为1，效果为42.5/50, 如果噪声为0, 则为8.3/50。
    #       解答，噪声在训练时有正则项的作用。但是在测试时，加的噪声太大了，大的有到1这样的数量级，足以严重影响神经网络输出
    ##########################################################################################################
    model.opt.noise_factor=0
    all_loss = model(datas, is_train=False)
    return all_loss

def go(**kwargs):

    opt.parse(kwargs)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    setup_seed(opt.seed)

    model = SortingModel(opt)
    n_params = 0
    for p in model.parameters():
        n_params += np.prod(p.size())
    print('# of parameters: {}'.format(n_params))

    if opt.use_gpu:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # data
    _ordered, _random, _hard_perms \
        = sinkhorn_ops.sample_uniform_and_order(opt.batch_size, opt.n_numbers, opt.prob_inc)
    _ordered_tiled = _ordered.repeat(opt.samples_per_num, 1)  # (n_sample * batch_size, n_numbers)
    _random_tiled = _random.repeat(opt.samples_per_num, 1)  # (n_sample * batch_size, n_numbers)

    if opt.use_gpu:
        _ordered_tiled = _ordered_tiled.cuda()
        _ordered = _ordered.cuda()
        _random_tiled = _random_tiled.cuda()
        _random = _random.cuda()
        _hard_perms = _hard_perms.cuda()

    datas = [_ordered, _random, _hard_perms, _ordered_tiled, _random_tiled]

    print("start training...")
    for i in range(opt.epoches):
        loss = train(model, datas, optimizer, opt)
        print('epoch:{}, train loss :{}'.format(i+1, loss.item()))
    print("finish train..")

    print("evaluating in train data...")
    l1_diff, l2sh_diff, prop_wrong, prop_any_worng, kendall_tau = test(model, datas)
    print("l1_diff:{:.2}, l2sh_diff:{:.4}, prop_wrong:{:.4}, per wrong number:{}/{}, kendall_tau:{:.4}".format(
        l1_diff, l2sh_diff, prop_wrong, prop_any_worng, opt.n_numbers, kendall_tau
    ))


    print("evaluating in test data...")
    _ordered, _random, _hard_perms \
        = sinkhorn_ops.sample_uniform_and_order(opt.batch_size, opt.n_numbers, opt.prob_inc)
    _ordered_tiled = _ordered.repeat(opt.samples_per_num, 1)  # (n_sample * batch_size, n_numbers)
    _random_tiled = _random.repeat(opt.samples_per_num, 1)  # (n_sample * batch_size, n_numbers)

    if opt.use_gpu:
        _ordered_tiled = _ordered_tiled.cuda()
        _ordered = _ordered.cuda()
        _random_tiled = _random_tiled.cuda()
        _random = _random.cuda()
        _hard_perms = _hard_perms.cuda()

    datas = [_ordered, _random, _hard_perms, _ordered_tiled, _random_tiled]
    l1_diff, l2sh_diff, prop_wrong, prop_any_worng, kendall_tau = test(model, datas)
    print("l1_diff:{:.2}, l2sh_diff:{:.4}, prop_wrong:{:.4}, per wrong number:{}/{}, kendall_tau:{:.4}".format(
                l1_diff, l2sh_diff, prop_wrong, prop_any_worng, opt.n_numbers, kendall_tau
    ))

if __name__ == '__main__':
    fire.Fire()