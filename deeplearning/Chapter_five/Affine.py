import numpy as np
class Affine:
    def __init__(self, w, b): #这个类需要导入w和b作为初始属性
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.w) + self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.w.T) #dout乘以w的转置
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0) #dout按照行进行求和
        return dx
    