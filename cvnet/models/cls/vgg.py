# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: VGG

VGG-11通过5个可以重复使用的卷积块来构造网络。根据每块里卷积层个数和输出通道数的不同可以定义出不同的VGG模型。
"""

import sys

import torch
from torch import nn

sys.path.append("../../..")
from cvnet.dataset import fashion_mnist
from cvnet.engine import trainer

from cvnet.models.cls.custom_layer import FlattenLayer


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7  # c * w * h
fc_hidden_units = 4096


# VGG-11
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # conv
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # fc
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net


net = vgg(conv_arch, fc_features, fc_hidden_units)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(net)
    X = torch.rand(1, 1, 224, 224)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    print("-" * 42)
    ratio = 8
    small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
    net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    print(net)

    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
