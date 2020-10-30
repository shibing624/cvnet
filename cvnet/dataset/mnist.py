# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: fashion mnist data

"""

import os

from torch.utils.data import DataLoader
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
