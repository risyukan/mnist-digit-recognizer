import numpy as np
a =np.array([[1,2],[3,4]])
a[[[True,False],[True,False]]] = 0
print(a)
print(a[[[True,False],[True,False]]])