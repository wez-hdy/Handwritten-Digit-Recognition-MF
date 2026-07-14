import numpy as np
import matplotlib.pylab as plt
from fontTools.misc.cython import returns


def step_function(x):
    # 阶跃函数
    return np.array(x>0,dtype=int)

def sigmoid(x):
    # sigmoid函数
    return 1 / (1+np.exp(-x))

def relu(x):
    # ReLU函数
    return np.maximum(0,x)

def softmax(a):
    # softmax函数。soft函数一般用于分类问题。
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# x = np.arange(-5.0,5.0,0.1)
# y = relu(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

A = np.array([[12,43,1],[2,5,6]])
print(A.shape)
print(A.ndim) # 2，二维数组。
B = np.array([[14,2,0],[74,44,8],[1,5,54]])
print(B.shape)
print(B.ndim)
C = np.dot(A,B)
print(C)
print(C.shape)
print(C.ndim)

D = np.array([1,3])
E = np.array([[2,4,5],[5,7,0]])
print(np.dot(D,E))

F = softmax(np.array([1.1,1,4.9]))
print(F)
print(np.sum(F))
#
# from torchvision import datasets, transforms
#

# # 下载到当前目录的 data/ 文件夹下（第一次会自动下载）
# mnist_train = datasets.MNIST(
#     root='./data',      # 存放路径
#     train=True,         # 训练集
#     download=True,      # 自动下载
#     transform=transforms.ToTensor()  # 转成 [0,1] 范围的 torch.Tensor
# )
#
# mnist_test = datasets.MNIST(
#     root='./data',
#     train=False,        # 测试集
#     download=True,
#     transform=transforms.ToTensor()
# )


from torchvision import datasets
import numpy as np

# ========== 1. PyTorch 加载（自动解压 + 解析二进制）==========
# root='./data' 表示从当前项目目录的 data/ 文件夹里找
mnist_train = datasets.MNIST(root='./data', train=True, download=False)
mnist_test  = datasets.MNIST(root='./data', train=False, download=False)

# ========== 2. 转成 NumPy 数组 ==========
x_train = mnist_train.data.numpy()      # 训练图像: (60000, 28, 28)
t_train = mnist_train.targets.numpy()   # 训练标签: (60000,)

x_test = mnist_test.data.numpy()        # 测试图像: (10000, 28, 28)
t_test = mnist_test.targets.numpy()     # 测试标签: (10000,)

# ========== 3. 和鱼书第三章格式对齐 ==========
# 鱼书里是把 28x28 展平成 784 维向量，并把 0~255 归一化到 0.0~1.0
# x_train、x_test已经被归一化（除以 255）了，像素值变成了 0.0 ~ 1.0 的浮点数。
x_train = x_train.reshape(-1, 784) / 255.0   # 形状变成 (60000, 784)
x_test  = x_test.reshape(-1, 784) / 255.0    # 形状变成 (10000, 784)

# ========== 4. 验证一下 ==========
print(f"x_train 类型: {type(x_train)}, 形状: {x_train.shape}, 数据范围: {x_train.min():.2f} ~ {x_train.max():.2f}")
print(f"t_train 类型: {type(t_train)}, 形状: {t_train.shape}") # 所有numpy数组的类的类型都是numpy.ndarray。
print(f"x_test  类型: {type(x_test)}, 形状: {x_test.shape}")
print(f"t_test  类型: {type(t_test)}, 形状: {t_test.shape}")

# 看看第一张图是什么数字
print(f"\n第一张图的标签是: {t_train[0]}")

from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label) # 5
print(img.shape) # (784,)
img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
print(img.shape) # (28, 28)
img_show(img * 255)
