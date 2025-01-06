class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads): #输入初始参数和梯度进行一次更新
        for key in params.keys():
            params[key]-= self.lr * grads[key]