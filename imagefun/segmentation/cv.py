# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import cv2
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
import matplotlib.pyplot as plt
imges = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
titles = ['img', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'thresh5']

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imges[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()