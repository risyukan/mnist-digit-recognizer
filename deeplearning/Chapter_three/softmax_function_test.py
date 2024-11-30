import numpy as np
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 形成一个指数构成的行列
print(exp_a)
sum_exp_a = np.sum(exp_a) # 对这个行列求和得到一个数
print(sum_exp_a)
y = exp_a / sum_exp_a # 行列除以一个数得到一个行列
print(y)