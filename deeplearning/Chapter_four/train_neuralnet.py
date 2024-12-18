import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = [] #这不是空字典，这是一个空行列
train_acc_list = []
test_acc_list = []

 # ハイパーパラメータ
iters_num = 10000 #学习次数
train_size = x_train.shape[0] 
batch_size = 100
learning_rate = 0.1

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1) #当batch数量大于训练数据数量时，iter_per_epoch为1，意味着每次迭代都要记录识别精度

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #建立实例

for i in range(iters_num):  #range(开始，终止，间隔)
    batch_mask = np.random.choice(train_size, batch_size)  #从train_size中随机选出batch_size个数，组成数列
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch) # 高速版!

     # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key]-= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)   #把每一步算出来的损失函数，加入到空行列中进行记录

# 1エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train) #所有训练数据的精度
        test_acc = network.accuracy(x_test, t_test) #所有测试数据的精度
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list)) #使用 numpy 的 arange 函数生成一个等差整数序列
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0) #设置纵轴的显示范围为 [0, 1.0]，以适应准确率的范围
plt.legend(loc='lower right') #显示图例，图例的位置设为右下角（lower right）。
plt.show() #调用 plt.show() 显示绘制好的图像。
