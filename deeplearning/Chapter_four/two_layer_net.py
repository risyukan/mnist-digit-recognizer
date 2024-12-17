import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01): #(初始化神经网络权重和偏置) weight_init_std是权重初始化标准差（默认值为 0.01）。用于控制随机初始化的权重值的范围，避免权重过大或过小。
        self.params = {} #定义了一个空字典
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 使权重值在较小范围内
        self.params['b1'] = np.zeros(hidden_size) #建立一个全零的向量，长度为 后层神经元个数
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)# 第二层的初始权重
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x): #x代表神经网络输入
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t): #神经网络的输入和标签（教师数据）
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t): #神经网络的输入和标签（教师数据）
        y = self.predict(x)
        y = np.argmax(y, axis=1) #指定沿第 1 轴（即行）进行操作
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