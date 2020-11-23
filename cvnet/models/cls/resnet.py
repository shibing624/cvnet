# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ResNet

残差块通过跨层的数据通道从而能够训练出有效的深度神经网络。
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("../../..")
from cvnet.dataset import fashion_mnist
from cvnet.engine import trainer
from cvnet.models.cls.custom_layer import GlobalAvgPool2d, FlattenLayer


class Residual(nn.Module):
    # 它可以设定输出通道数、是否使用额外的1×1卷积层来修改通道数以及卷积层的步幅。
    # 我们也可以在增加输出通道数的同时减半输出的高和宽。
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1)  # in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


# ResNet则使用4个由残差块组成的模块
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 接着我们为ResNet加入所有残差块。这里每个模块使用两个残差块。
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
# 最后，与GoogLeNet一样，加入全局平均池化层后接上全连接层输出。
net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))  # 分类：10类

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(net)

    X = torch.rand((1, 1, 224, 224))
    print(X, X.shape)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    batch_size = 256
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size=batch_size, resize=96)

    lr = 0.001
    num_epochs = 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, model_path='')
