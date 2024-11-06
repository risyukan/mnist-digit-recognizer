import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x)) #标量值会与numpy行列的各要素进行运算
if __name__ == "__main__": #使用这个if来确保，在引入sigmoid函数时，以下代码不会被执行
    x = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(x))