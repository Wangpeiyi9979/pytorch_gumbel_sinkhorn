"""
@Time: 2019/12/17 9:40
@Author: Wang Peiyi
@Site : 
@File : config.py
"""



from texttable import Texttable


class DefaultConfig(object):
    """
    user can set default hyperparamter here
    hint: don't use 【parse】 as the name
    """

    model = 'Sorting_model'

    seed = 10
    use_gpu = 1
    gpu_id = 1

    batch_size = 10
    n_numbers = 100
    temperature = 1.0
    prob_inc = 1
    lr = 0.1
    samples_per_num = 5
    n_iter_sinkhorn = 10
    n_units = 32
    noise_factor = 1.
    optimizer = 'adam'
    drop_prob = 0

    # 优化器:
    clip_grad = 2

    # 模型区分参数
    model_opt = 'DEF'

    epoches = 500

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter；；；：；：jkkj:;;jjkkwwwwwwwbbb;::；:q
        :：；：：
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        """
        一些依赖于其他超参数的超参数设置, 比如，在调参过程中，需要设置模型区分参数【model_opt】随着【lr】改变，就需要增加下句：
         setattr(self, 'print_opt', "model_{}_lr_{}".format(self.model, self.lr))
        """
        setattr(self, 'print_opt', "model_{}_lr_{}".format(self.model, self.lr))


        """
        print the information of hyperparater
        """
        t = Texttable()
        t.add_row(["Parameter", "Value"])
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not (k.startswith('__') or k == 'parse'):
                t.add_row(["P:" + str(k), "V:" + str(getattr(self, k))])
        print(t.draw())


opt = DefaultConfig()



