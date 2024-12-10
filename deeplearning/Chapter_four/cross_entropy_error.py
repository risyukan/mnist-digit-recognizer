import numpy as np

def cross_entropy_error(y,t):
    delta = 1e-7 #这是1乘10的负7次幂
    return  -np.sum(t * np.log(y + delta)) #这是为了防止y为0时，数值过大导致溢出
if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))