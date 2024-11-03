import numpy as np
import matplotlib.pylab as plt
def step_function(x):
    y = x > 0   #将x转换成布尔型的行列
    return y.astype(int)  #astype（）可以将numpy的行列类型进行转换,将ture转换为1，false为0

x = np.arange(-5.0, 5.0, 0.1) #以0.1为间隔，从-5.0生成到4.9为止
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y 軸の範囲を指定
plt.show()