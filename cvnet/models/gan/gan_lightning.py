# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

reference: https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb#scrollTo=DOY_nHu328g7

"""

import os
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


# Define dataset
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=os.path.join('~', '.pytorch', 'datasets', 'mnist'), batch_size=100, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])

        self.img_shape = (1, 28, 28)
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
    def __init__(self, latent_size, img_shape, hidden_size=128):
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
            *block(hidden_size, hidden_size * 2),
            *block(hidden_size * 2, hidden_size * 4),
            *block(hidden_size * 4, hidden_size * 8),
            nn.Linear(hidden_size * 8, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        out = self.model(x_flat)
        return out


class GAN(pl.LightningModule):
    def __init__(self, channels, width, height, latent_size=100, lr=0.0002, **kwargs):
        super().__init__()
        self.latent_size = latent_size
        self.lr = lr
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_size=self.latent_size, img_shape=data_shape)
        self.discriminator = Discriminator(data_shape)

        self.validataion_z = torch.randn(8, self.latent_size)
        self.example_input_array = torch.zeros(2, self.latent_size)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_size)
        z = z.type_as(imgs)

        # train G
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z)

            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # BCEloss
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                "loss": g_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            return output

        # train D
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # D loss
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                "loss": d_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.lr  # self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validataion_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    dm = MNISTDataModule()
    print(dm.img_shape)
    model = GAN(*dm.img_shape)
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=4, max_epochs=100, progress_bar_refresh_rate=20)
    else:
        trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, dm)
