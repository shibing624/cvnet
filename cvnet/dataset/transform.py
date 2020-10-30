# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import collections
import math
import random
from typing import Any, Callable, Dict, Optional, List

import torch
# import torchvision.transforms._transforms_video as transforms_video
from torch.distributions.beta import Beta
from torchvision import transforms

TRANSFORM_REGISTRY = {}


def register_transform(name: str):
    """Registers a :class:`ClassyTransform` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    :class:`ClassyTransform` from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyTransform subclass like this:

    .. code-block:: python

      @register_transform("my_transform")
      class MyTransform(ClassyTransform):
          ...

    To instantiate a transform from a configuration file, see
    :func:`build_transform`."""

    def register_transform_cls(cls: Callable[..., Callable]):
        if name in TRANSFORM_REGISTRY:
            raise ValueError("Cannot register duplicate transform ({})".format(name))
        if hasattr(transforms, name):
            raise ValueError(
                "{} has existed in torchvision.transforms, Please change the name!".format(
                    name
                )
            )
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return register_transform_cls


_IMAGENET_EIGEN_VAL = [0.2175, 0.0188, 0.0045]
_IMAGENET_EIGEN_VEC = [
    [-144.7125, 183.396, 102.2295],
    [-148.104, -1.1475, -207.57],
    [-148.818, -177.174, 107.1765],
]

_DEFAULT_COLOR_LIGHTING_STD = 0.1


@register_transform("lighting")
class LightingTransform(object):
    """
    Lighting noise(AlexNet - style PCA - based noise).
    This trick was originally used in `AlexNet paper
    <https://papers.nips.cc/paper/4824-imagenet-classification
    -with-deep-convolutional-neural-networks.pdf>`_

    The eigen values and eigen vectors, are taken from caffe2 `ImageInputOp.h
    <https://github.com/pytorch/pytorch/blob/master/caffe2/image/
    image_input_op.h#L265>`_.
    """

    def __init__(
            self,
            alphastd=_DEFAULT_COLOR_LIGHTING_STD,
            eigval=_IMAGENET_EIGEN_VAL,
            eigvec=_IMAGENET_EIGEN_VEC,
    ):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval)
        # Divide by 255 as the Lighting operation is expected to be applied
        # on `img` pixels ranging between [0.0, 1.0]
        self.eigvec = torch.tensor(eigvec) / 255.0

    def __call__(self, img):
        """
        img: (C x H x W) Tensor with values in range [0.0, 1.0]
        """
        assert (
            img.min() >= 0.0 and img.max() <= 1.0
        ), "Image should be normalized by 255 and be in range [0.0, 1.0]"
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
                .clone()
                .mul(alpha.view(1, 3).expand(3, 3))
                .mul(self.eigval.view(1, 3).expand(3, 3))
                .sum(1)
                .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def build_transforms(cfg, is_train=True):
    normalize_transform = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN,
                                         scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)),
            transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            transforms.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            normalize_transform
        ])

    return transform


def convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    """
    assert (
        torch.max(targets).item() < classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = torch.zeros(
        (targets.shape[0], classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


class MixupTransform(object):
    """
    This implements the mixup data augmentation in the paper
    "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float, num_classes: Optional[int] = None):
        """
        Args:
            alpha: the hyperparameter of Beta distribution used to sample mixup
            coefficient.
            num_classes: number of classes in the dataset.
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sample: the batch data.
        """
        if sample["target"].ndim == 1:
            assert self.num_classes is not None, "num_classes is expected for 1D target"
            sample["target"] = convert_to_one_hot(
                sample["target"].view(-1, 1), self.num_classes
            )
        else:
            assert sample["target"].ndim == 2, "target tensor shape must be 1D or 2D"

        c = Beta(self.alpha, self.alpha).sample().to(device=sample["target"].device)
        permuted_indices = torch.randperm(sample["target"].shape[0])
        for key in ["input", "target"]:
            sample[key] = c * sample[key] + (1.0 - c) * sample[key][permuted_indices, :]

        return sample


@register_transform("tuple_to_map")
class TupleToMapTransform(object):
    """A transform which maps image data from tuple to dict.

    This transform has a list of keys (key1, key2, ...),
    takes a sample of the form (data1, data2, ...) and
    returns a sample of the form {key1: data1, key2: data2, ...}
    If duplicate keys are used, the corresponding values are merged into a list.

    It is useful for mapping output from datasets like the `PyTorch
    ImageFolder <https://github.com/pytorch/vision/blob/master/torchvision/
    datasets/folder.py#L177>`_ dataset (tuple) to dict with named data fields.

    If sample is already a dict with the required keys, pass sample through.

    """

    def __init__(self, list_of_map_keys: List[str]):
        """The constructor method of TupleToMapTransform class.

        Args:
            list_of_map_keys: a list of dict keys that in order will be mapped
                to items in the input data sample list

        """
        self._map_keys = list_of_map_keys

    def __call__(self, sample):
        """Transform sample from type tuple to type dict.

        Args:
            sample: input sample which will be transformed

        """
        # If already a dict/map with appropriate keys, exit early
        if isinstance(sample, dict):
            for key in self._map_keys:
                assert (
                    key in sample
                ), "Sample {sample} must be a tuple or a dict with keys {keys}".format(
                    sample=str(sample), keys=str(self._map_keys)
                )
            return sample

        assert len(sample) == len(self._map_keys), (
            "Provided sample tuple must have same number of keys "
            "as provided to transform"
        )
        output_sample = collections.defaultdict(list)
        for idx, s in enumerate(sample):
            output_sample[self._map_keys[idx]].append(s)

        # Unwrap list if only one item in dict.
        for k, v in output_sample.items():
            if len(v) == 1:
                output_sample[k] = v[0]

        return output_sample
