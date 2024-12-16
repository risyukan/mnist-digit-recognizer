import numpy as np
def numerical_gradient(f, x): #生成梯度向量的函数，x是行列
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # np.zeros_like(x) は、x と同じ形状の配列で、その要素がすべて0の配列を生成する
    for idx in range(x.size): #缺点：这个函数只能处理一维度的x（x是指神经网络权重）
        tmp_val = x[idx]
        # f(x+h) の計算
        x[idx] = tmp_val + h
        fxh1 = f(x) #求出对第一个未知数的f（x+h，x1，x2，）
        # f(x-h) の計算
        x[idx] = tmp_val- h
        fxh2 = f(x)  #求出对第一个未知数的f（x-h,x2,x3,x4）
        grad[idx] = (fxh1- fxh2) / (2*h)  #算出第一个梯度
        x[idx] = tmp_val # 値を元に戻す。 接下来下次循环，解决第二个梯度（对第二个未知数的偏导数）
    return grad

def function_2(x): #x是个行列
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))