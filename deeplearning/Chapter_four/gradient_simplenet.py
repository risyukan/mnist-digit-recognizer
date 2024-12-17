import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 生成符合标准正态分布（均值为0，标准差为1）的二维数组。作为神经网络的初始权重

    def predict(self, x): #进行一层的神经网络计算
        return np.dot(x, self.W) #self.来访问当前类中的属性

    def loss(self, x, t):   
        z = self.predict(x) #加上self.来访问当前类中的方法
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(np.argmax(p))
t = np.array([0, 0, 1])
print(net.loss(x, t))

def f(W): #W并没有被用到，因为net.w的特性。之所以要设置W是因为numerical_gradient代码中的f（x）这一结构，必须有一个引数
    return net.loss(x, t)

dW = numerical_gradient(f, net.W) #第一个参数需要是一个函数，而不能是一个值。为什么是f
print(dW) #这一步非常难，在f（新net.w）这一步，将会进入到net.loss（输入图像，标签）这一步，
#然后进入到net.predict这一步，输入将会和新的w.net矩阵进行运算，实现代码中f（x）的功能
#代码是计算梯度矩阵的
