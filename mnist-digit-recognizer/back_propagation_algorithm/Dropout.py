import numpy as np
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio #这里的*号是为了把x.shape的元组形式解包成为单独的两个或者多个数
            return x * self.mask #被丢弃的神经元的值变为 0，其他神经元保留
        else:
            return x * (1.0- self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask #仅对前向传播中未被丢弃的神经元传递梯度，其他梯度为 0