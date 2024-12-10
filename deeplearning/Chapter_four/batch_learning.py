import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(normalize= True,one_hot_label=True) #在模型推理时，最大值索引和标签数字进行对比，所以不需要one_hot形式。
# 在模型学习时，要带入公式计算所以需要one_hot形式。
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0] #0是行数，1是列数，以此类推
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #从0到第一个参数-1的范围中，随机选出第二参数个数的数，组成数组
x_batch = x_train[batch_mask] #取出随机的十个图像（10x768）
t_batch = t_train[batch_mask] #取出随机的十个标签（one-hot形式）（10x10）

print(x_batch.shape) 
print(t_batch.shape)