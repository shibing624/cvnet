# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: fashion mnist data

"""

import os

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .transform import build_transforms


def load_data_mnist(cfg, is_train=True, root=os.path.join(
    '~', '.pytorch', 'datasets', 'mnist')):
    """Download the MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)

    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    # get data
    datasets = MNIST(root=root, train=is_train, transform=transforms, download=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


def load_data_mnist_without_cfg(batch_size, resize=None, root=os.path.join(
    '~', '.pytorch', 'datasets', 'mnist'), use_normalize=True):
    """Download the MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    if use_normalize:
        transformer += [transforms.Normalize(mean=[0.5], std=[0.5])]
    transformer = transforms.Compose(transformer)

    mnist_train = torchvision.datasets.MNIST(root=root, train=True, transform=transformer, download=True)
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, transform=transformer, download=False)
    num_workers = 4

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter
