import numpy as np
import matplotlib.pylab as plt

def function(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

def num_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f,x):
    h = 1e-4
    g = np.zeros_like(x)
    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + h
        f_1 = f(x)
        x[idx] = tmp - h
        f_2 = f(x)
        x[idx] = tmp
        g[idx] = (f_1 - f_2) / (2*h)
    return g

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x

x = np.arange(0.0,20.0,0.1)
y = function(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

print(num_diff(function,5))

print(function_2(np.array([1,2])))

print(numerical_gradient(function_2,np.array([1.0,2.0])))
print(gradient_descent(function_2,np.array([-3.0,4.0]),lr=0.01,step_num=1000))
