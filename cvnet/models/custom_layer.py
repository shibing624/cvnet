# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import torch
import torch.nn.functional as F
from torch import nn


class FlattenLayer(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
