import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A)) #ndim可以获得行列的次元
print(A.shape) #可以获取这个实例行列的形状
print(A.shape[0]) #0是行数，1是列数。2是层数
B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B))
print(B.shape) 