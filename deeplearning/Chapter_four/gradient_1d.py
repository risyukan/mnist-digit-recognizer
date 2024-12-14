import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x): #这是求某点切线方程的函数
    d = numerical_diff(f, x)  #计算出切线的斜率
    print(d)
    y = f(x) - d*x  #计算切线在y轴上的截距
    return lambda t: d*t + y  #点斜式：返回一个以t为未知数的匿名函数
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 10)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()