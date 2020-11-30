# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class BagDataset(Dataset):
    def __init__(self, transform=None, data_dir=''):
        self.transform = transform
        self.data_dir = data_dir
        self.bag_data_dir = '{}/bag_data'.format(data_dir)
        self.mask_data_dir = '{}/bag_data_msk'.format(data_dir)

    def __len__(self):
        return len(os.listdir(self.bag_data_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.bag_data_dir)[idx]
        imgA = cv2.imread(os.path.join(self.bag_data_dir, img_name))
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread(os.path.join(self.mask_data_dir, img_name), 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
        return imgA, imgB


def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk - 1] = 1
    return buf


def load_data(data_dir, batch_size=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # imagenet params

    bag = BagDataset(transform, data_dir)

    train_size = int(0.9 * len(bag))
    test_size = len(bag) - train_size
    train_dataset, test_dataset = random_split(bag, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders = {
        'train': train_dataloader,
        'val': test_dataloader
    }
    return dataloaders
