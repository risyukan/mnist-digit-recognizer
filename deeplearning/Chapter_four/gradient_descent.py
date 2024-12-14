from numerical_gradient import numerical_gradient
import numpy as np
def gradient_descent(f, init_x, lr=0.01, step_num=100): #init_x是初始参数,进行了100次参数调整
    x = init_x
    for i in range(step_num): #生成0到99的数列，循环100次
        grad = numerical_gradient(f, x)
        x-= lr * grad
    return x

def function_2(x): #x是个行列
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
y = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(y)