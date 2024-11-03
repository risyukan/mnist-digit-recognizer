import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x)) #标量值会与numpy行列的各要素进行运算
x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))