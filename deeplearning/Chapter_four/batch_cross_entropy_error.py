import numpy as np
def cross_entropy_error_1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #交叉熵误差通常处理批量数据，因此需要将一维数据调整为二维形状调整为（1，t.size）形
        y = y.reshape(1, y.size) #这段代码的目的确保输入数据的维度一致性
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_2(y, t): #y是（batch_size,10)形状，是神经网络训练时的输出
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #np.arange生成从 0到 batch_size - 1 的整数数组。提取出y二维数组中，对应图像对应行，t中数字标签对应的列，来进行计算。
    #この例では、y[np.arange(batch_size), t] は[y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]] の NumPy 配列を生成します