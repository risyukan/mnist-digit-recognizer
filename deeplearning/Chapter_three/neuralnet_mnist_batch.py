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
    with open("sample_weight.pkl", 'rb') as f: #以rb格式来打开这个文件，将其赋给变量f。这个文件以字典的方式存储数据
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

batch_size = 100 # バッチの数
accuracy_cnt = 0
for i in range(0, len(x), batch_size): #这个函数从0循环到len（x）-100，每次的间隔为100，例如0，100，200
    x_batch = x[i:i+batch_size] #这个函数将行列切成i到 i+batch_size-1,共100份
    y_batch = predict(network, x_batch) #对这100个图像进行推理，得到100x10的矩阵
    p = np.argmax(y_batch, axis=1)  #对矩阵每一行进行单独分析，输出该行最大值在该行索引，得到100个元素的一维矩阵
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #对预测值 p 和目标值 t[i:i+batch_size] 进行逐元素比较，返回一个布尔数组。
    #例如，[1, 2, 3] == [1, 2, 4] 会返回 [True, True, False]。最后np.sum将布尔数组中的 True 视为 1，False 视为 0，然后求和。

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))