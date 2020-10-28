import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def load_data_cifar10(batch_size, resize=None, root=os.path.join(
    '~', '.pytorch', 'datasets', 'CIFAR10')):
    """Download the CIFAR10 dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    train_transformer = transformer + [transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()]
    test_transformer = transformer + [transforms.ToTensor()]

    train_transformer = transforms.Compose(train_transformer)
    test_transformer = transforms.Compose(test_transformer)

    data_train = torchvision.datasets.CIFAR10(root=root, train=True, transform=train_transformer, download=True)
    data_test = torchvision.datasets.CIFAR10(root=root, train=False, transform=test_transformer, download=True)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(data_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(data_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def get_cifar10_labels(labels):
    """Get text labels for cifar10."""
    text_labels = ['airplane', 'car', 'bird' 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_labels[int(i)] for i in labels]


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # plt.savefig("img.png")


def show_cifar10():
    _, test_loader = load_data_cifar10(4)
    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    print(labels)
    print(get_cifar10_labels(labels))


if __name__ == '__main__':
    show_cifar10()
