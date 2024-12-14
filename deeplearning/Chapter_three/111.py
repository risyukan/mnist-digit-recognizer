import numpy as np
print(np.float32(1e-7)+1) #计算出现误差是因为，1是float64，当1e-7转换成float64时会携带微小误差
print(1e-7+1)
