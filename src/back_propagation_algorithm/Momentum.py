import numpy as np
class Momentum: #模拟物理中实际存在的惯性,改善SGD算法
    def __init__(self, lr=0.01,momentum=0.9): 
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None: #初始化v为0矩阵
            self.v = {}
            for key,val in params.items(): #循环这个字典的键和值分别放在key和val中
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys(): #进行参数更新一次
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
  