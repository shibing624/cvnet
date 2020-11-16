# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import  torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl


# Define dataset
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=os.path.join('~', '.pytorch', 'datasets', 'mnist'), batch_size=128, num_workers = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,),(0.3081,))
        ])

        self.dims = (1,28,28)
        self.num_classes = 10

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            print('mnist_full len:', len(mnist_full))
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

# Generator
class Generator(nn.Module):

    def __init__(self, latent_size, img_shape,hidden_size = 128):
        super().__init__()
        self.img_shape = img_shape

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_size, hidden_size, normalize=False),
            *block(hidden_size,hidden_size*2),
            *block(hidden_size*2,hidden_size*4),
            *block(hidden_size*4, hidden_size*8),
            nn.Linear(hidden_size*8, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self,x):
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img

# D
class Discriminator(nn.Module):
    def __init(self, img_shape,hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x_flat = x.view(x.size(0), -1)
        out = self.model(x_flat)
        return out



