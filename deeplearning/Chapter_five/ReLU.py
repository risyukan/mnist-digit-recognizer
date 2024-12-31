class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self,x): #x是个数组
        self.mask = (x <= 0)
        out = x.copy() #复制输入 x，以防止直接修改 x 的数据（保持输入数据的完整性）
        out[self.mask] = 0 #将 out 中对应于 x 小于等于 0 的位置的值设置为 0。

        return out
    
    def backward(self,dout): #根据正方向的输入x来判断，反向传播时的值
        dout[self.mask] = 0 #把对应为Ture的地方置为0
        dx = dout
        return dx