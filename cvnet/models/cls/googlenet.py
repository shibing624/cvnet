# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: GoogLeNet model

Inception块相当于一个有4条线路的子网络。它通过不同窗口形状的卷积层和最大池化层来并行抽取信息，并使用1×11×1卷积层减少通道数从而降低模型复杂度。
GoogLeNet将多个设计精细的Inception块和其他层串联起来。其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。
GoogLeNet和它的后继者们一度是ImageNet上最高效的模型之一：在类似的测试精度下，它们的计算复杂度往往更低。
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("../../..")
from cvnet.dataset import fashion_mnist
from cvnet.engine import trainer
from cvnet.models.cls.custom_layer import FlattenLayer, GlobalAvgPool2d


class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        # c1 - c4为每条线路里的层的输出通道数
        super().__init__()
        # route1, 1x1
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # route2, 1x1后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # route3, 1x1后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # route4, 3x3最大池化后接1x1
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        # 在通道维上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），每个模块之间使用步幅为2的3×3最大池化层来减小输出高宽。
# 第一模块使用一个64通道的7×7卷积层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第二模块使用2个卷积层：首先是64通道的1×11×1卷积层，然后是将通道增大3倍的3×33×3卷积层。它对应Inception块中的第二条线路。
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第三模块串联2个完整的Inception块。
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第四模块更加复杂。它串联了5个Inception块，
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第五模块有输出通道数为256+320+128+128=832256+320+128+128=832和384+384+128+128=1024384+384+128+128=1024的两个Inception块。
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

# 最后我们将输出变成二维数组后接上一个输出个数为标签类别数的全连接层。
net = nn.Sequential(
    b1, b2, b3, b4, b5,
    FlattenLayer(),
    # 标签类别数是10
    nn.Linear(1024, 10)
)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(net)
    print("-" * 42)
    net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))
    print(net)
    X = torch.rand(1, 1, 96, 96)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
