"""
@Time: 2019/12/16 20:40
@Author: Wang Peiyi
@Site : BeiJin
@File : sinkhorn_ops.py.py
"""

import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau


# noinspection PyUnresolvedReferences
def sample_uniform_and_order(n_lists, n_numbers, prob_inc):
    """从均匀分布中抽样数字，并且将它们排序
    返回的数据要么升序要么降序，升序的概率为prob_inc

    input:
        n_lists: 一个整数， 需要排序的list的数目。
        n_numbers: 一个整数， 每个list里的元素数目。
        prob_inc: 排序后的n_lists个list中，是升序的list的比例
    return:
        ordered: 一个(n_lists, n_numbers)的tensor, 每个list要么升序，要么降序
        random: 一个(n_lists, n_numbers)的tensor, 原始的均匀分布中抽样出的数字
        permutes: 一个(n_lists, n_numbers)的intTensor，
                      ordered元素在原始random中的下标，即满足
                      random[i][permutes[i]] == ordered[i]

        ps: 源码中写成了 ordered[i][permutes[i]] == random[i], 应该是写错了
    """

    bern = torch.distributions.Bernoulli(probs=torch.ones(n_lists, 1) * prob_inc).sample()
    sign = -1 * (bern * 2 - 1)
    random = torch.rand(n_lists, n_numbers)     # 从[0,1]中采样均匀分布
    random_with_sign = random * sign
    ordered, permutes = torch.topk(random_with_sign, k=n_numbers)
    ordered = ordered * sign
    assert 0 not in (random[0][permutes[0]] == ordered[0])
    return ordered, random, permutes

# noinspection PyUnresolvedReferences
def sample_gumbel(shape,
                  eps=1e-20):
    """采样任意维度的符合标准 gunmbel 分布的变量

    @param shape: 一个整数列表
    @param eps: float， 为了数值稳定
    @return:
        @samples: 形状为shape的对应的gumbel分布的Tensor
    """
    u = torch.rand(shape)
    samples = -torch.log(-torch.log(u + eps) + eps)
    return samples


# noinspection PyUnresolvedReferences
def sinkhorn(log_alpha,
             n_iters=20):
    """对log_alpha实现不完善的Sinkhorn normalization

    实现过程见原文公式(1), 原本输入必须是exp(log_alpha)
    但是为了实现数值稳定，我们在log空间实现该操作，并且只
    在返回时调用exp函数

    @param log_alpha: 一个(N, N)或者是(batch_size, N, N)的Tensor
    @param n_iters: int, 表示迭代数
    @return: 一个(batch_size, N, N)的Tensor，即：
        batch_size个双线性矩阵, 如果温度τ够小，则返回的是置换矩阵
    """

    n = log_alpha.size(1)
    log_alpha = log_alpha.view(-1, n, n)

    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, 2).view(-1, n, 1)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, 1).view(-1, 1, n)
    return torch.exp(log_alpha)


def gumbel_sinkhorn(log_alpha,
                    temp=1.0,
                    n_samples=1,
                    noise_factor=1.0,
                    n_iters=20,
                    squeeze=True):

    """在log空间内实现 【Gumbel Sinkhorn】 算子，即原文中的【S(g(X^, θ) / τ)】.

    源代码: https://github.com/google/gumbel_sinkhorn/blob/312c73f61a8731960fc1addecef157cd4274a993/sinkhorn_ops.py#L141

    @param log_alpha: 一个(batch_size, N, N)的Tensor， 对应着【g(X^, θ)】。
    @param temp: 一个float值，原文中对应的温度τ。
    @param n_samples: 一个int值，采样的个数。
    @param noise_factor:  一个float值，代表着加入的噪声强度。
    @param n_iters: 一个int值，实现Sinkhorn operator的循环次数
    @param squeeze: 一个bool值，如果是True并且只有一个sample，则返回一个3D的Tensor
    @return:
        【1】 sink:  一个(batch_size, n_samples, N, N)的tensor,
                经过可微分操作的sinkhorn算子得到
                即batch_size*n_sample个双随机矩阵，如果温度τ足够小，并且n_iters足够大,
                则可以近似置换矩阵， 如果n_sample=1，则输出为3D. ----
        【2】 log_alpha_w_noise: 一个(batch_size, n_samples, N, N)的tensor，
                对应着log_alpha的噪声采样，并且被τ除，如果n_samples=1, 则输出为3D.
                预测的时候，该Tensor应该能够大致和置换矩阵类似，因此在预测时采用匈牙利算法
                从该Tensor得到batch_size个置换矩阵
    """

    use_gpu = False
    if log_alpha.is_cuda:
        use_gpu = True

    n = log_alpha.size(1)
    log_alpha = log_alpha.view(-1, n, n)
    batch_Size = log_alpha.size(0)
    log_alpha_w_noise = log_alpha.repeat(n_samples, 1, 1)
    ##########################################################################################################
    # the function here: 加入gumbel噪声，原理是根据                                                           
    ##########################################################################################################
    
    noise = sample_gumbel([n_samples * batch_Size, n, n]) * noise_factor
    
    if use_gpu:
        noise = noise.cuda()
    ##########################################################################################################
    # function here:    根据原文描述增加噪声:“Importantly, we will always add noise
    #                   to the output layer g( ˜X, θ) as a regularization device”
    #                   除以温度τ以实现公式(4)
    ##########################################################################################################

    log_alpha_w_noise = log_alpha_w_noise + noise  # (n_sample * batch_size, n, n)
    log_alpha_w_noise = log_alpha_w_noise / temp
    sink = sinkhorn(log_alpha_w_noise, n_iters)
    if n_samples > 1 or squeeze is False:
        sink = sink.view(n_samples, batch_Size, n, n)
        sink = sink.permute(1, 0, 2, 3)
        log_alpha_w_noise = log_alpha_w_noise.view(n_samples, batch_Size, n, n)
        log_alpha_w_noise = log_alpha_w_noise.permute(1, 0, 2, 3)

    return sink, log_alpha_w_noise


def matching(matrix_batch):
    """
    @param matrix_batch: 一个(n_sample * batch_size, n, n)的tensor，
    传入的原始Tensor是【log_alpha_w_noise_flat】
    @return:
            listperms: 一个(n_sample * batch_size, n)的IntTesor，
            即： listperms[i,:]是matrix_batch[i,:,:]的最大指派任务解.

    """

    if matrix_batch.dim() == 2:
        matrix_batch = matrix_batch.view(-1, matrix_batch.size(1), matrix_batch.size(2))
        
    ##########################################################################################################
    # TODO: 弄明白为什么要使用detach，不然会报错：Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    ##########################################################################################################

    use_gpu = matrix_batch.is_cuda
    matrix_batch = matrix_batch.detach().cpu()
    sol = torch.zeros(matrix_batch.size(0), matrix_batch.size(1)).long()

    for i in range(matrix_batch.size(0)):
        sol[i, :] = torch.LongTensor(linear_sum_assignment(-matrix_batch[i, :])[1].astype(int))
    if use_gpu:
        sol = sol.cuda()
    return sol


def torch_batch_one_hot(batch_list, class_num=None):
    """实现batch形式的one_hot

    @param batch_list:  一个(batch_size, n)/(n)的Tensor.
    @param class_num:   指定的one_hot向量的长度，必须大于batch_list元素的最大值
                        如果不指定，默认为batch_list的最大值
    @return:
            batch_one_hot: 如果batch_size>1, 返回一个(batch_size, n, class_num)的Tensor，
                        对应batch_size个one_hot向量.
                        如果batch_size==1, 则返回一个(1, class_num)的Tensor.
    @example1:
            batch_list = tensor([[2, 3, 4],
                                  [1, 0, 2]])
            class_num = None

            batch_one_hot = tensor([[[0, 0, 1, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1]],

                                  [[0, 1, 0, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0]]], dtype=torch.int32)
   @eexample2:
            batch_list = tensor([[2, 3, 4],
                                  [1, 0, 2]])
            class_num = 6

            batch_one_hot = tensor([[[0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0]],

                                    [[0, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]]], dtype=torch.int32)
    """
    if batch_list.dim() == 1:
        batch_list = batch_list.view(1, -1)

    batch_size = batch_list.size(0)
    n = batch_list.size(1)

    max_item = torch.max(batch_list).item() + 1
    if class_num == None:
        class_num = max_item
    if class_num < max_item:
        raise Exception("class_num should lager than the max item of batch_list")

    zeros = torch.zeros(batch_size, n, class_num)
    if batch_list.is_cuda:
        zeros = zeros.cuda()
    batch_one_hot = zeros.scatter_(dim=2, index=batch_list.view(batch_size, n, 1), value=1)
    return batch_one_hot.long()


def torch_inv_batch_one_hot(batch_one_hot):
    """实现batch个one_hot向量到原始batch个list的转换，是torch_batch_one_hot的逆函数.
        主要是通过线性代数的乘法实现

    @param batch_one_hot:   一个(batch_size, n, class_num)或者(n, class_num)
                            的one_hot标签
    @return:
            batch_list: 一个(batch_size, n)的Tensor. 如果输入为(n, class_num)
                        则batch_size == 1
    """
    if batch_one_hot.dim() == 2:
        batch_one_hot = batch_one_hot.view(1, batch_one_hot.size(0), batch_one_hot.size(1))

    operator = torch.arange(batch_one_hot.size(2)).view(-1, 1)
    if batch_one_hot.is_cuda:
        operator = operator.cuda()
    ##########################################################################################################
    # TODO: 探究：【batch_list = torch.matmul(batch_one_hot, operator).long()】
    #               这里会报错:RuntimeError: addmm for CUDA tensors only supports floating-point types.
    #               Try converting the tensors with .float() at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:408
    #       原因:Cuda环境下只能进行浮点运算
    #       修正：【 batch_list = torch.matmul(batch_one_hot.float(), operator.float()).long()】
    ##########################################################################################################

    batch_list = torch.matmul(batch_one_hot.float(), operator.float()).long()
    return batch_list.squeeze(-1)


def invert_listperm(listperm):
    """ 求得排列的逆。

    @param listperm: 一个(batch_size, n)的IntTensor, 每一行一个排列。
    @return: 一个(batch_size, n)的IntTensor, 每一行也是一个排列，并且与对应输入成逆。
    """

    matperm = torch_batch_one_hot(listperm)
    inv_matperm = matperm.permute(0, 2, 1)
    inv_listperm = torch_inv_batch_one_hot(inv_matperm)
    return inv_listperm

def permute_batch_list(batch_split, permutes):
    """
    @param batch_split: 一个(batch_size, n_number)的Tensor
    @param permutes: 一个(batch_size, n_number)的IntTensor. 实现重排
    @return:
        perm_batch_split: 一个batch_size, n_number的Tensor. 即根据permute
        重拍后的Tensor
    """
    perm_matrix = torch_batch_one_hot(permutes) # (batch_size, n, n)
    batch_split = batch_split.unsqueeze(-1)         # (batch_size, n, 1)
    perm_batch_split = torch.matmul(perm_matrix.float(), batch_split.float()) # (batch_size, n, 1)
    return perm_batch_split.squeeze(-1)

def kendall_tau(x, y):
    """
    计算两个排列的kendall相关系数
    @param x: 一个(batch_size, n)的tensor
    @param y: 一个(batch_size, n)的tensor
    @return: (batch_size, 1)的tensor，表示batch_size个相关系数
    """
    if x.dim() == 1:
        x = x.view(1, -1)
    if y.dim() == 1:
        y = y.view(1, -1)
    kendall = torch.zeros(x.size(0), 1)
    for i in range(x.size(0)):
        kendall[i] = kendalltau(x[i,:].cpu(), y[i,:].cpu())[0]
    return kendall















































