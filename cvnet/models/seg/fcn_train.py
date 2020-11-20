# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


sys.path.append('../../..')
from cvnet.models.seg.bag_data import load_data
from cvnet.models.seg.fcn import FCNs, VGGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(num_epochs=50, show_vgg_params=False, data_dir='', model_save_dir='./checkpoints/'):
    train_dataloader, test_dataloader = load_data(data_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch: {}, iter: {}/{}, train loss: {}'.format(epoch, index, len(train_dataloader), iter_loss))

                plt.subplot(1, 2, 1)
                plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
                plt.subplot(1, 2, 2)
                plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                plt.pause(0.5)
                plt.savefig('train_{}_{}.png'.format(epoch, index))

        test_loss = 0.
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = np.argmin(bag_msk_np, axis=1)

                if np.mod(index, 5) == 0:

                    plt.subplot(1, 2, 1)
                    plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                    plt.pause(0.5)
                    plt.savefig('test_{}_{}.png'.format(epoch,index))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss: %.4f, epoch test loss: %.4f, %s'
              % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))

        if np.mod(epoch, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epoch))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/bag/', help='path for load data')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints/', help='path for save model')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--show_vgg_params', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    train(args.num_epochs, args.show_vgg_params, args.data_dir, args.model_save_dir)
