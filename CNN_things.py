import  numpy as np

x = np.random.rand(10,1,28,28)
print(x.shape)
print(x[0].shape)
print(x[0][0]) # 28*28

import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters:
        input_data : 4维数组 (数据量, 通道, 高, 长)
        filter_h   : 滤波器的高
        filter_w   : 滤波器的长
        stride     : 步幅
        pad        : 填充
    Returns:
        col : 2维数组 (滑动次数, 滤波器展开后的长度)
    """
    # 1. 获取输入数据的形状
    N, C, H, W = input_data.shape

    # 2. 计算输出特征图的高和宽（卷积后会有几个滑动位置）
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 3. 填充（零填充）
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 4. 开辟一个巨大的空容器（这就是消耗内存的地方）
    #    形状: (N, C, filter_h, filter_w, out_h, out_w)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 5. 核心魔法：利用切片和步长，一次性把所有的滑动窗口填进去
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 6. 变形：把 (N, C, FH, FW, OH, OW) 变成 (N*OH*OW, C*FH*FW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col