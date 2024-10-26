def AND (x1,x2):
    w1,w2,theta = 0.5,0.5,0.8
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

if __name__ == '__main__': #确保代码仅在作为主程序运行时执行。
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]: #定义元组列表
        y = AND(xs[0], xs[1])
        print(f'{xs} -> {y} ')  #str可以将元组转换为字符串
        print(str(xs) + " -> " + str(y)) #两种print都可以