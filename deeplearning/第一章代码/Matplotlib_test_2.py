import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,8,0.01)
y1 = np.sin(x) #y1的值
y2 = np.cos(x) #y2的值
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label='cos') # 定制这条线的线形和标签名称
plt.xlabel("x") # x 軸のラベル
plt.ylabel("y") # y 軸のラベル
plt.title('sin&cos') # 整个图的タイトル
plt.legend() # 显示每条线对应的图例
plt.show()  # 显示图表