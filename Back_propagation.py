from symtable import Class

from ch07.apply_filter import out
import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self,dout):
        dx = dout * self.y # 翻转
        dy = dout * self.x
        return dx, dy


apple = 100
apple_num = 2
tax = 1.1
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = out

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return self.out

    def backward(self,dout):
        dx = dout * self.out * (1.0-self.out)
        return dx


class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx