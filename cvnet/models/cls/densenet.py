# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DenseNet

在跨层连接上，不同于ResNet中将输入与输出相加，DenseNet在通道维上连结输入与输出。
DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）
"""

import sys

import torch
from torch import nn

sys.path.append("../../..")
from cvnet.dataset import fashion_mnist
from cvnet.engine import trainer
from cvnet.models.cls.custom_layer import GlobalAvgPool2d, FlattenLayer


# DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk


# 稠密块由多个conv_block组成，每块使用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维度上把输入和输出连结
        return X


# 由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层(transition)用来控制模型复杂度。
# 它通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk


# 构造DenseNet模型。DenseNet首先使用同ResNet一样的单卷积层和最大池化层。
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# DenseNet使用的是4个稠密块
num_channels, growth_rate = 64, 32  # num_channels为当前通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 同ResNet一样，我们可以设置每个稠密块使用多少个卷积层。这里我们设成4

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module("DenseBlock_%d" % i, DB)
    # 上一个稠密快的输出通道数
    num_channels = DB.out_channels
    # 稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 同ResNet一样，最后接上全局池化层和全连接层来输出。
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("rele", nn.ReLU())
net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出（batch, num_channels, 1, 1）
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))  # 类别数10

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(net)

    X = torch.rand((1, 1, 96, 96))
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    batch_size = 256
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size=batch_size, resize=96)

    lr = 0.001
    num_epochs = 5
    model_path = "densenet_mnist.pt"
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, model_path)
