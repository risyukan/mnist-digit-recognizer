import numpy as np
import matplotlib.pylab as plt
def step_function(x):
    y = x > 0   #将x转换成布尔型的行列
    return y.astype(np.int)  #astype（）可以将numpy的行列类型进行转换