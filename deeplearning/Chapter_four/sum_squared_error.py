import numpy as np
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(sum_squared_error(np.array(y), np.array(t))) #将变量转换为numpy数组，需要提前统一变量的类型，全是浮点数或者不是

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(sum_squared_error(np.array(y), np.array(t))) #神经网络分类错误时，损失函数的值较大