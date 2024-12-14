import numpy as np
def numerical_diff(f, x):
    h = 1e-4 # 0.0001  数值过小的话python会省略小数点，导致无法正确显示数值
    return (f(x+h)- f(x-h)) / (2*h) #用这个方法减小误差