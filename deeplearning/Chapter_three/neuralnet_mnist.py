import sys,os 
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
from sigmoid_function import sigmoid
from softmax_function import softmax

def get_testdata():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True,one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f: #以rb格式来打开这个文件，将其赋给变量f
        network = pickle.load(f) #将存储的字节流还原为原始的 Python 对象.通常 f 是一个以二进制读取模式（'rb'）打开的文件
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] #总共有三层神经网络
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1  #dot是行列乘积
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_testdata()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)): #当x是二维矩阵时，len（x）输出的是矩阵的行数
    y = predict(network, x[i]) #每一行代表一个图像，用每个图像输入神经网络进行分类
    p = np.argmax(y) #返回最大值所在的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) #将accuracy_cnt转换为浮点数，确保计算结果也是浮点数，然后再转换成字符串输出
