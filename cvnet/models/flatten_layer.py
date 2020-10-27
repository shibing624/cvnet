# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import torch
from torch import nn

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

