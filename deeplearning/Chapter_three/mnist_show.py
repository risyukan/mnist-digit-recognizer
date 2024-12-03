import sys,os 
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #np.unit8将图像数据转换为无符号 8 位整数,然后将这个NumPy数组转换为 PIL 图像对象
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0] #这个变量里有784个数，是矩阵中一行的元素
label = t_train[0]
print(label) # 5
print(img.shape) # (784,)
img = img.reshape(28, 28) # 形状を元の画像サイズに変形
print(img.shape) # (28, 28)
print(x_train.shape) #六万张训练图像
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
img_show(img)