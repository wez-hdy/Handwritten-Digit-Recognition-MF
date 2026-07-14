import numpy as np

def or_gate(x1,x2): # 函数名称应该小写。
    # 或门逻辑实现
    x = np.array([x1,x2])
    w = np.array([0.2,0.2])
    b = -0.1
    if np.sum(x*w)+b <= 0:
        return 0
    else:
        return 1

def and_gate(x1,x2):
    # 与门逻辑实现
    x = np.array([x1,x2])
    w = np.array([0.2,0.2])
    b = -0.3
    if np.sum(x*w)+b <= 0:
        return 0
    else:
        return 1

def nand_ate(x1,x2):
    # 与非门逻辑实现
    x = np.array([x1,x2])
    w = np.array([-0.2,-0.2])
    b = 0.3
    if np.sum(x*w)+b <= 0:
        return 0
    else:
        return 1

def xor_gate(x1,x2):
    # 异或门逻辑实现。输入信号只包含一个1时，输出信号才会为1。
    y1 = or_gate(x1,x2)
    y2 = nand_ate(x1,x2)
    return and_gate(y1,y2)
    # 如果不写return，and_gate的返回值就会被丢弃。因为它没有被赋值给任何变量，也没有被return。
    # xor_gate函数执行完毕，如果没有return语句，默认返回None。

print(xor_gate(0,0)) # 0
print(xor_gate(0,1)) # 1
print(xor_gate(1,0)) # 1
print(xor_gate(1,1)) # 1