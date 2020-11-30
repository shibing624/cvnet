# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
03-advanced/generative_adversarial_network/main.py
"""

import os
import sys

import torch
import torch.nn as nn
from torchvision.utils import save_image

sys.path.append("../../..")
from cvnet.dataset import mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

latent_size = 64
hidden_size = 128
image_size = 28 * 28
num_epochs = 100
batch_size = 100
lr = 0.0002
sample_dir = "samples"

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

data_loader, _ = mnist.load_data_mnist_without_cfg(batch_size=batch_size, use_normalize=False)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# train
total_step = len(data_loader)
for epoch in range(num_epochs):
    images = None
    fake_images = None
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        # labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # train discriminator
        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs

        # compute loss with fake image
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train generator
        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_fn(outputs, real_labels)
        # back prop and optimizer
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}'.format(
                epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(),
                fake_score.mean().item()
            ))

    # Save real image
    if (epoch + 1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images_{}.png'.format(epoch + 1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.pt')
torch.save(D.state_dict(), 'D.pt')
