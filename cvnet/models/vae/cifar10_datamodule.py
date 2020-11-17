# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
from warnings import warn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import CIFAR10
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


def imagenet_normalization():
    normalize = transform_lib.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


def cifar10_normalization():
    normalize = transform_lib.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    return normalize


def stl10_normalization():
    normalize = transform_lib.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
    return normalize


class CIFAR10DataModule(LightningDataModule):
    name = 'cifar10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs
    ):
        """
        .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
            Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
            :width: 400
            :alt: CIFAR-10

        Specs:
            - 10 classes (1 per class)
            - Each image is (3 x 32 x 32)

        Standard CIFAR10, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])

        Example::

            from pl_bolts.datamodules import CIFAR10DataModule

            dm = CIFAR10DataModule(PATH)
            model = LitModel()

            Trainer().fit(model, dm)

        Or you can set your own transforms

        Example::

            dm.train_transforms = ...
            dm.test_transforms = ...
            dm.val_transforms  = ...

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ImportError('You want to use CIFAR10 dataset loaded from `torchvision` which is not installed yet.')

        self.dims = (3, 32, 32)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self.DATASET(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args)

    def train_dataloader(self):
        """
        CIFAR train set removes a subset to use for validation
        """
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split]
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        CIFAR10 val set uses a subset of the training set for validation
        """
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split]
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def test_dataloader(self):
        """
        CIFAR10 test set uses the test split
        """
        transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms, **self.extra_args)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return cf10_transforms
