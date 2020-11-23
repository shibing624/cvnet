# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: NiN model

NiN重复使用由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络。
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。
NiN的以上设计思想影响了后面一系列卷积神经网络的设计。
"""

import sys

import torch
from torch import nn

sys.path.append("../../..")
from cvnet.dataset import fashion_mnist
from cvnet.engine import trainer
from cvnet.models.cls.custom_layer import FlattenLayer, GlobalAvgPool2d


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, 3, 1, 1),
    nn.MaxPool2d(3, 2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, 3, 1, 1),
    GlobalAvgPool2d(),
    FlattenLayer()
)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(net)
    X = torch.rand(1, 1, 224, 224)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
