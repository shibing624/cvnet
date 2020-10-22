# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import cv2
import matplotlib.pyplot as plt
cwd = os.getcwd()
image_path = os.path.join(cwd, '../../data/7.jpg')
img = cv2.imread(image_path, 0)
ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(ret, th)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imges = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
titles = ['img', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'thresh5']

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imges[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()

# ---------------------------------自适应阈值分割---------------------------------
# img = cv2.imread('./image/paper2.png', 0)

# 固定阈值
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
thresh3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)

# 全局阈值，均值自适应，高斯加权自适应对比
titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, thresh1, thresh2, thresh3]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
