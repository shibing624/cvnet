# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

样式迁移常用的损失函数由3部分组成：内容损失使合成图像与内容图像在内容特征上接近，样式损失令合成图像与样式图像在样式特征上接近，而总变差损失则有助于减少合成图像中的噪点。
可以通过预训练的卷积神经网络来抽取图像的特征，并通过最小化损失函数来不断更新合成图像。
用格拉姆矩阵表达样式层输出的样式。
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

print(os.getcwd())
content_img = Image.open('../../../docs/rainier.jpg')
content_img.show()

style_img = Image.open('../../../docs/autumn_oak.jpg')
style_img.show()

# 预处理和后处理图像
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])


def preprocess(PIL_img, image_shape):
    process = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)


def postprocess(img_tensor):
    inv_normalize = torchvision.transforms.Normalize(
        mean=-rgb_mean / rgb_std,
        std=1 / rgb_std)
    to_PIL_image = torchvision.transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))


pretrained_net = torchvision.models.vgg19(pretrained=True)
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 在抽取特征时，我们只需要用到VGG从输入层到最靠近输出层的内容层或样式层之间的所有层。
net_list = []
for i in range(max(content_layers + style_layers) + 1):
    net_list.append(pretrained_net.features[i])
net = torch.nn.Sequential(*net_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
net = net.to(device)


# 需要中间层的输出，因此这里我们逐层计算，并保留内容层和样式层的输出。
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    """
    对内容图像抽取内容特征
    :param image_shape:
    :param device:
    :return:
    """
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, device):
    """
    对样式图像抽取样式特征
    :param image_shape:
    :param device:
    :return:
    """
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    """内容损失"""
    return F.mse_loss(Y_hat, Y)


def gram(X):
    """用这样的格拉姆矩阵表达样式层输出的样式"""
    num_channels, n = X.shape[1], X.shape[2] * X.shape[3]
    X = X.view(num_channels, n)
    return torch.matmul(X, X.t()) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    """样式损失"""
    return F.mse_loss(gram(Y_hat), gram_Y)


def tv_loss(Y_hat):
    """
    合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。
    一种常用的降噪方法是总变差降噪（total variation denoising）。
    降低总变差损失能够尽可能使邻近的像素值相似。
    :param Y_hat:
    :return:
    """
    return 0.5 * (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))


# 样式迁移的损失函数即内容损失、样式损失和总变差损失的加权和。
content_weight, style_weight, tv_weight = 1, 1000, 10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l) + sum(contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


# 在样式迁移中，合成图像是唯一需要更新的变量。
class GeneratedImage(torch.nn.Module):
    def __init__(self, img_shape):
        super(GeneratedImage, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


# 样式图像在各个样式层的格拉姆矩阵styles_Y_gram将在训练前预先计算好。
def get_inits(X, device, lr, styles_Y):
    gen_img = GeneratedImage(X.shape).to(device)
    gen_img.weight.data = X.data
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer


# 训练
def train(X, contents_Y, styles_Y, device, lr, max_epochs, lr_decay_epoch):
    print("training on ", device)
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, gamma=0.1)
    for i in range(max_epochs):
        start = time.time()

        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)

        optimizer.zero_grad()
        l.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, sum(contents_l).item(), sum(styles_l).item(), tv_l.item(),
                     time.time() - start))
    return X.detach()


def main():
    image_shape = (150, 225)
    content_X, contents_Y = get_contents(image_shape, device)
    style_X, styles_Y = get_styles(image_shape, device)
    output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)
    out_img = postprocess(output)
    out_img.show()
    out_img.save("style_small.png")

    # 为了得到更加清晰的合成图像，下面我们在更大的300×450尺寸上训练
    image_shape = (300, 450)
    _, content_Y = get_contents(image_shape, device)
    _, style_Y = get_styles(image_shape, device)
    X = preprocess(postprocess(output), image_shape).to(device)
    big_output = train(X, content_Y, style_Y, device, 0.01, 500, 200)
    out_img = postprocess(big_output)
    out_img.save("style_big.png")


if __name__ == '__main__':
    main()
