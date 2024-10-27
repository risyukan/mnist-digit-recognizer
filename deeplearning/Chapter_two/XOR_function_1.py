import OR_function
import AND_function
import NAND_function
def XOR(x1,x2):
    s1 = NAND_function.NAND(x1,x2)
    s2 = OR_function.OR(x1,x2)
    y = AND_function.AND(s1,s2)
    return y
if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))