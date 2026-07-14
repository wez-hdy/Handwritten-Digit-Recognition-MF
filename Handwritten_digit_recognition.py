# 1.引入必要的库
import  torch
import torch.nn as nn # 提供构建网络的模块
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 2.定义超参数
batch_size = 128 # 每批处理的数据数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU/CPU训练
epochs = 20 # 训练数据集的轮数

# 3.构建pipeline，对图像进行处理
pipeline = transforms.Compose([
    transforms.ToTensor(), # 将图片转换成Tensor格式
    transforms.Normalize((0.1307,), (0.3081,)) # 正则化：降低模型复杂度
])

# 4.下载、加载数据
# 下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)
# 加载数据
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=True)

"""
# 插入代码，显示MNIST的图片
import matplotlib.pyplot as plt
image, label = test_set[0] # test_set[0]是一个元组，(图像数据，标签)，其中图像数据是一个pytorch张量，(1,28,28)
plt.imshow(image.squeeze(), cmap='gray') # imshow()，只能传入二维灰度图或通道为3的RGB彩图。squeeze会去掉数据中的通道维
plt.show()
"""

# 5.构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1:灰度图片的通道;10:输出通道;5:卷积核
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3) # 10:输入通道;20:输出通道;3:卷积核
        self.fc1 = nn.Linear(20*10*10,500) # 20*10*10输入通道，500输出通道
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0) # x的形状是 batch_size * 1 * 28 * 28
        x = self.conv1(x) # 输入：batch_size*1*28*28，输出：batch_size*10*24*24,(28-5+1=24)，步长默认是1
        x = F.relu(x) # 激活函数，输出非线性函数，保持形状不变，增加神经网路的表达能力
        x = F.max_pool2d(x, 2,2) # 输入：batch_size*10*24*24，输出：batch_size*10*12*12

        x = self.conv2(x) # 输入：batch_size*10*12*12，输出：batch_size*20*10*10 (12-3+1=10)
        x = F.relu(x)

        x = x.view(input_size, -1) # 拉平，-1自动计算维度——> 20*10*10=2000

        x = self.fc1(x) # 输入：batch*2000，输出：batch*500
        x = F.relu(x) # batch*500

        x = self.fc2(x) # 输入：batch*500，输出：batch*10

        return x

# 6.定义优化器
model = Digit().to(device) # 将模型部署到设备

optimizer = optim.Adam(model.parameters()) # Adam中每个训练参数的学习率可以自适应变化

# 7.定义训练方法
def train_model(model,device,train_loader,optimizer,epoch):
    # 模型训练
    model.train()
    for batch_index , (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 根据梯度参数优化
        optimizer.step()
        if batch_index % 100 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))

# 8.定义测试方法
def test_model(model,device,test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    # 测试损失
    with torch.no_grad(): # 测试不计算梯度，不进行反向传播
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.argmax(dim=1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test_Average Loss : {:.4f}, Accuracy : {:.3f}\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)))

# 9.调用方法7/8
for epoch in range(1, epochs + 1):
    train_model(model,device,train_loader,optimizer,epoch)
    test_model(model,device,test_loader)