# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LeNet

卷积神经网络就是含卷积层的网络。
LeNet交替使用卷积层和最大池化层后接全连接层来进行图像分类。
"""

import sys

import torch
from torch import nn

sys.path.append("../../..")
from cvnet.dataset import fashion_mnist
from cvnet.engine import trainer
from cvnet.models.cls.custom_layer import FlattenLayer


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


lenet_bn = nn.Sequential(
    nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),  # kernel_size, stride
    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    FlattenLayer(),
    nn.Linear(16 * 4 * 4, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet()
    print(net)
    batch_size = 256
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size=batch_size)

    lr = 0.001
    num_epochs = 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    print("-" * 42)
    print(lenet_bn)
    optimizer = torch.optim.Adam(lenet_bn.parameters(), lr=lr)
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size=batch_size)
    trainer.train(lenet_bn, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
