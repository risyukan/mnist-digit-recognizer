import numpy as np
def softmax(a):
    c = a - np.max(a) #减去行列中最大的数，为了减小exp（x）的大小，防止溢出，使计算正确运行
    exp_a = np.exp(c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
if __name__ == "__main__":
    a = np.array([1010, 1000, 990])
    print(softmax(a))