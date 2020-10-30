# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 迁移学习

迁移学习将从源数据集学到的知识迁移到目标数据集上。微调是迁移学习的一种常用技术。
目标模型复制了源模型上除了输出层外的所有模型设计及其参数，并基于目标数据集微调这些参数。而目标模型的输出层需要从头训练。
一般来说，微调参数会使用较小的学习率，而从头训练输出层可以使用较大的学习率。
"""
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.append("..")

from cvnet.engine import trainer


def load_data_hotdog(batch_size=128, root=os.path.join(
    '~', '.pytorch', 'datasets', 'hotdog')):
    # download hotdog url: https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip
    root = os.path.expanduser(root)
    print(os.listdir(root))  # ['train', 'test']

    # 指定RGB三个通道的均值和方差来将图像通道归一化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

    train_imgs = ImageFolder(os.path.join(root, 'train'), transform=train_augs)
    test_imgs = ImageFolder(os.path.join(root, 'test'), transform=test_augs)
    train_iter = DataLoader(train_imgs, batch_size, shuffle=True)
    test_iter = DataLoader(test_imgs, batch_size)
    return train_iter, test_iter


def main():
    cwd = os.getcwd()
    print(cwd)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_net = models.resnet18(pretrained=True)
    print(pretrained_net.fc)
    pretrained_net.fc = nn.Linear(512, 2)
    # 输出分类由imagenet的1000类，改为热狗数据集的2类
    print(pretrained_net.fc)

    # 分开设置lr，预训练部分模型的学习率0.01，fc部分的学习率0.1
    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

    lr = 0.001
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                          lr=lr, weight_decay=0.001)
    batch_size = 128
    num_epochs = 5
    train_iter, test_iter = load_data_hotdog(batch_size)
    trainer.train(pretrained_net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


def main_no_pretrained():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scratch_net = models.resnet18(pretrained=False, num_classes=2)

    lr = 0.1
    optimizer = optim.SGD(scratch_net.parameters(),
                          lr=lr, weight_decay=0.001)
    batch_size = 128
    num_epochs = 5
    train_iter, test_iter = load_data_hotdog(batch_size)
    trainer.train(scratch_net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()
