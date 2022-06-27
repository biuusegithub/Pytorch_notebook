import torch
from torch import nn
from d2l import torch as d2l


'''
缓解全连接层所导致的网络中参数量过大的问题

Nin的架构: 
        无全连接层
        交替使用NiN块和步幅为2的最大池化层，逐步减少高宽和增大通道数
        最后使用全局平均池化层（即池化层大小 = 输入大小）得到输出，其输入通道数是类别数
        [全局池化层是一个很强的操作去降低模型的复杂度，但其缺点是收敛变慢]
'''


'''NiN块: 一个卷积层之后用2个1X1卷积层去替代全连接层的作用'''
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


'''NiN模型'''
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())


'''训练'''
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())