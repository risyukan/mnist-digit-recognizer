import numpy as np
class AdaGrad: #定制每一个参数的学习率，该参数的梯度变化越大，学习率就会慢慢减小
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self,params,grads): 
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val) #生成一个与w相同的空字典

        for key in params.keys():
            self.h += grads[key] * grads[key] #两个相同大小的矩阵各元素相乘，直接用*
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) #1e-7是为了防止分母为0。这行代码中矩阵里各个元素进行单独运算
        