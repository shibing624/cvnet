# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

print(1e3)


import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

sys.path.append("..")

content_img = Image.open('../docs/rainier.jpg')
content_img.show()
