import os
import sys
sys.path.append(os.pardir)
from function.util import im2col
import numpy as np
class Convolution:
    def __init__(self,W, b, stride=1,pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self,x): 
        FN, C, FH, FW = self.W.shape #这是filter
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_W = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride,self.pad) #对输入数据进行处理，转换为2维数据
        col_W = self.W.reshape(FN,-1).T # フィルターの展開
        out = np.dot(col,col_W) + self.b

        out = out.reshape(N, out_h, out_W,-1).transpose(0, 3, 1, 2) #out 是二维矩阵，通过 np.dot 计算得到，形状为 (N * out_h * out_W, FN)。
        #我们之所以reshape(N, out_h, out_W,-1)，发挥空间想象力，原本的二维数据是N * out_h * out_W条FN,还原数据为四维度，也必须以FN为基础，搭建成平面，然后成正方体，然后再搭建成多个正方体。之后再进行transpose
        #直接reshape(N,FN, out_h, out_W)会导致数据混乱，因为基础数据条为out_W。
        return out
    