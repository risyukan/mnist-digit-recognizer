
class MulLayer: #定义了一个乘算的单位
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x #为了保存前向传播时的输入值
        self.y = y
        out = x * y
        return out
 
    def backward(self, dout): #dout是上流传递过来的微分值
        dx = dout * self.y # x とyをひっくり返す
        dy = dout * self.x
        return dx, dy   