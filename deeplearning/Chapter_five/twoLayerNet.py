import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01): #(初始化神经网络权重和偏置) weight_init_std是权重初始化标准差（默认值为 0.01）。用于控制随机初始化的权重值的范围，避免权重过大或过小。
        self.params = {} #定义了一个空字典
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 使权重值在较小范围内
        self.params['b1'] = np.zeros(hidden_size) #建立一个全零的向量，长度为 后层神经元个数
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)# 第二层的初始权重
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict() #建立一个有序字典
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1']) #类中建立了一个类实例，放在字典里
        self.layers['Relu1'] = Relu() #建立激活函数层
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss() #之所以不在字典中，是为了下一行的for循环做铺垫，因为这个类的forward函数需要两个参数，和其他的类不同

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x) #每循环一次，进行一层的正向推理
        
        return x #输出第二层的输出
    
    # x:入力データ, t:教師データ
    def loss(self,x,t): #在其他方法的基础上进行编写
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self, x, t): #神经网络的输入和标签（教师数据）
        y = self.predict(x)
        y = np.argmax(y, axis=1) #指定沿第 1 轴（即行）进行操作,返回最大值的索引
        t = np.argmax(t, axis=1) 

        accuracy = np.sum(y == t) / float(x.shape[0]) #会返回数组或矩阵 x 的第一个维度的大小，即行数
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) #建立一个函数，w是未知数，loss是公式
        grads = {} 
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self,x,t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values()) #将字典的值取出，组成一个数列
        layers.reverse() #将有序字典反转顺序，便于反向传播
        for layer in layers: #进行反向传播
            dout = layer.backward(dout)
        
        # 最重要的是w和b的梯度，需要存储下来
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db



        return grads