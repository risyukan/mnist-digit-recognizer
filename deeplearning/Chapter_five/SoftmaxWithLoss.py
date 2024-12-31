import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmax の出力
        self.t = None # 教師データ（one-hot vector）

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size #除以batch_size是因为求导推理公式时，是对单个数据的推理，多个数据时公式里会多1/N.
        return dx