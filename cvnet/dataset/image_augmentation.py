# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

图像增广基于现有训练数据生成随机图像从而应对过拟合。
为了在预测时得到确定的结果，通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广。
可以从torchvision的transforms模块中获取有关图片增广的类。
"""

import os
import sys

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
sys.path.append("../..")
from cvnet.figure import plt, use_svg_display, set_figsize,show_images

if __name__ == '__main__':

    img = Image.open('../../docs/7.jpg')
    img.show()

    def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
        Y = [aug(img) for i in range(num_rows * num_cols)]
        show_images(Y, num_rows, num_cols, scale)

    apply(img, torchvision.transforms.RandomHorizontalFlip())
    apply(img, torchvision.transforms.RandomVerticalFlip())