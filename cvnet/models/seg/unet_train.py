# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time
# sys.path.append('../../..')
from cvnet.models.seg.bag_data import load_data
from cvnet.models.seg.unet import ResNetUNet, calc_loss, print_metrics
from cvnet.models.seg import helper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def train_model(model, dataloaders, optimizer, scheduler,num_epochs=50, model_path=''):

    all_train_iter_loss = []
    all_test_iter_loss = []
    best_loss = 1e10
    # start train
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        since = time.time()
        train_loss = 0.
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        pred = model(inputs)
                        pred = torch.sigmoid(pred)
                        pred = pred.data.cpu().numpy()
                        print(pred.shape)

                        # Change channel-order and make 3 channels for matplot
                        input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

                        # Map each channel (i.e. class) to each color
                        target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
                        pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

                        helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), '{}/model_{}.pt'.format(model_save_dir, epoch))
                print('save to {}/model_{}.pt'.format(model_save_dir, epoch))


        time_elapsed = time.time() - since
        print('spend: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/bag/', help='path for load data')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth', help='path for save model')
    parser.add_argument('--num_epochs', type=int, default=5)
    args = parser.parse_args()
    print(args)

    dataloaders = load_data(args.data_dir)
    # Get a batch of training data
    inputs, masks = next(iter(dataloaders['train']))

    print(inputs.shape, masks.shape)
    for x in [inputs.numpy(), masks.numpy()]:
        print(x.min(), x.max(), x.mean(), x.std())

    plt.imshow(reverse_transform(inputs[0]))


    n_class = 2
    model = ResNetUNet(n_class=n_class)
    model_save_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)
    train_model(model,dataloaders, optimizer_ft, exp_lr_scheduler,  args.num_epochs, args.model_path)
