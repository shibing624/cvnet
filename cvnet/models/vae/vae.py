# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: VAE

原理：https://zhuanlan.zhihu.com/p/34998569
reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py#L38-L65
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.append("../../..")
from cvnet.dataset import mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

hidden_size = 400
z_size = 20
image_size = 28 * 28
num_epochs = 100
batch_size = 100
lr = 0.0001
sample_dir = "samples"

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

data_loader, _ = mnist.load_data_mnist_without_cfg(batch_size=batch_size, use_normalize=False)


class VAE(nn.Module):
    def __init__(self, image_size=28 * 28, hidden_size=400, z_size=20):
        super().__init__()
        self.fc1 = nn.Linear(image_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, z_size)
        self.fc3 = nn.Linear(hidden_size, z_size)
        self.fc4 = nn.Linear(z_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_val = self.encode(x)
        z = self.reparameterize(mu, log_val)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_val


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

m = iter(data_loader)
n = next(m)
print(n)

# start training
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):
        # Forward
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Backprop
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()
            ))

    with torch.no_grad():
        # Save image
        z = torch.randn(batch_size, z_size).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled_{}.png'.format(epoch + 1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst_{}.png'.format(epoch + 1)))
