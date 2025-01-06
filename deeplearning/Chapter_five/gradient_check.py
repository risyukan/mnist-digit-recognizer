import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from twoLayerNet import TwoLayerNet
#用于验证误差反向传播法的实装有没有错误
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3] #取出前三个训练数据
t_batch = t_train[:3] #取出前三个标签数据

grad_numerical = network.numerical_gradient(x_batch, t_batch) #数值微分的方法,得到字典
grad_backprop = network.gradient(x_batch, t_batch) #反向传播的方法，得到字典

 # 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():  #在字典的键中进行循环
    diff = np.average(np.abs(grad_backprop[key] - grad_backprop[key])) #average计算总的平均值，abs计算差值的绝对值，去掉正负号，关注梯度的偏差量，而不考虑方向。
    print(key + ':' + str(diff))