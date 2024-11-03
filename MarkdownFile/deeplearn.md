# 从零开始的深度学习笔记
## 1.multi-layered perceptron
![alt text](1730010147874.png)
看上去是3层构造，但是实际上只有两层有权重，所以是2层感知机。
## 2.sigmoid function
![alt text](78a0a59172e4ea4dd75e711a0d8239a.png)
## 3.sigmoid函数图像
![alt text](1730638215829.png)
線形関数の問題点は、どんなに層を深くしても、それと同じことを行う「隠れ層のないネットワーク」が必ず存在する、という事実に起因します。
## 4.ReLU函数
NumPyのmaximumという関数を使っています。このmaximumは、入力された値から大きいほうの値を選んで出力する関数です。