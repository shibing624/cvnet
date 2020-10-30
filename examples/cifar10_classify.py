# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: resnet cifar10 classify
"""
import os
import sys

import torch

sys.path.append("..")

from cvnet.engine import trainer
from cvnet.models.resnet import net
from cvnet.dataset import cifar


def main():
    cwd = os.getcwd()
    print(cwd)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(net)

    X = torch.rand((1, 1, 224, 224))
    print(X, X.shape)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    batch_size = 256
    lr = 0.001
    num_epochs = 5
    # 在cifar10数据集上测试
    train_iter, test_iter = cifar.load_data_cifar10(batch_size=batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()
