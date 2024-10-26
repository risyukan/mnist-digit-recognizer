import matplotlib.pyplot as plt
from matplotlib.image import imread
inori = imread(r"D:\stable diffusion\AI绘画成品\蝶祈\00017-3939819118.png") #imread是读取图像的函数
plt.imshow(inori) #将矩阵渲染成图像
plt.axis('off') #关闭坐标轴
plt.show() #弹窗显示所有图像
print(inori.shape)