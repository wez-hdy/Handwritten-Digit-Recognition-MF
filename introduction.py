from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x = np.array([1.1,2.4,6.2]) # numpy数组
y = np.array([7.7,5.6,9.0])
print(x)
print(type(x))
print(x.shape) # 输出(3,),说明x是一个长度为3的一维数组
# 一维数组是[]定义，二维数组是[[]]定义。
print(x+y) # element-wise运算
print(x*2) # 广播运算

a = np.array([
    [1.2,3.8,11.6],
    [2.3,1.5,89.0]
])
# np.array() 的第一个参数应该是一个完整的列表，最好格式化表达。
print(a.shape) # 输出(2,3),2行3列
print(a.dtype) # 小数的数据类型默认是float64，双精度浮点数
# float64 精度更高，适合科学计算
# float32 精度较低，但速度更快，常用于深度学习
b = np.array([
    [14.3,7.2,1.6],
    [9.3,5.5,9.0]
])
print(a+b)
print(a*b) # *在NumPy中是逐元素乘法（element-wise multiplication），不是矩阵乘法
# 逐元素乘法的规则：
# 两个数组的形状必须相同（或可以广播）、对应位置的元素相乘、结果形状不变
print(a*10)
print(a*x) # 尽管两个数组的形状，但是numpy数组具有广播功能，可以把x展开成和a矩阵一样的形状，再进行逐元素计算
# 广播的规则：
# NumPy比较两个数组的形状时，是从最后一个维度（最右边）开始（type属性的输出，如（3，）/（2，2）），向前逐一比较的。
# 两个维度能兼容，必须满足以下任意一个条件：
# 1.相等：两个维度的大小相同；2.其中一个是 1：其中一个数组在该维度上的大小为 1；3.缺失维度：其中一个数组没有这个维度（比如一个是二维，一个是一维）。
# 如果所有维度都兼容，就可以广播；只要有一个维度不兼容，就会报错ValueError。

print(type(a[0]))
print(a[0,1])
for i in a: # 在NumPy中，对一个多维数组进行迭代（遍历）时，默认是沿着第一个轴（axis=0）进行的
    print(i) # 所以每次循环会把每一行作为一维数组，直接返回

bf = b.flatten()
for i in bf:
    print(i,end=" ")
print(bf>12) # [ True False False False False False]
print(bf[bf>12])

a = np.arange(0,6,0.1) # 创建一个从0到10的等差数列，步长为0.1。
b1 = np.cos(a)
b2 = np.sin(a)
plt.plot(a,b1,label='cosa') # 绘制a和b的曲线图。
plt.plot(a,b2,linestyle='--',label='sina')
plt.xlabel('a') # 设置x轴标签
plt.ylabel('b') # 设置y轴标签
plt.title("cosa&sina") # 设置标题
plt.legend() # 显示图例
plt.show() # 显示图像

img = imread(r"C:\Users\Wesley\Pictures\一志愿待录取通知.jpg")
# imread()功能：从文件中读取图片数据
# 返回值：将图片转换为 NumPy 数组（三维数组）
# 形状为 (高度, 宽度, 通道数)，例如 (1080, 1920, 3)
# 每个元素是像素的 RGB 值（0-255 或 0.0-1.0）
plt.imshow(img)
# plt.imshow()功能：在 matplotlib 的画布上绘制/渲染图像
# 注意：这一步只是把图像"画"到内存中的画布上，并没有真正显示出来
plt.show()
# 打开一个窗口，真正绘制所有的内容。