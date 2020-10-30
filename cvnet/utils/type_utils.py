# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn

# constants:
CHECKPOINT_FILE = "checkpoint.torch"
CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = torch.device("cuda")


def is_pos_int(number: int) -> bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


def is_pos_float(number: float) -> bool:
    """
    Returns True if a number is a positive float.
    """
    return type(number) == float and number >= 0.0


def is_pos_int_list(l: List) -> bool:
    """
    Returns True if a list contains positive integers
    """
    return type(l) == list and all(is_pos_int(n) for n in l)


def is_pos_int_tuple(t: Tuple) -> bool:
    """
    Returns True if a tuple contains positive integers
    """
    return type(t) == tuple and all(is_pos_int(n) for n in t)


def is_long_tensor(tensor: torch.Tensor) -> bool:
    """
    Returns True if a tensor is a long tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("LongTensor")
    else:
        return False


def is_float_tensor(tensor: torch.Tensor) -> bool:
    """
    Returns True if a tensor is a float tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("FloatTensor")
    else:
        return False


def is_double_tensor(tensor: torch.Tensor) -> bool:
    """
    Returns True if a tensor is a double tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("DoubleTensor")
    else:
        return False


def is_leaf(module: nn.Module) -> bool:
    """
    Returns True if module is leaf in the graph.
    """
    assert isinstance(module, nn.Module), "module should be nn.Module"
    return len([c for c in module.children()]) == 0 or hasattr(module, "_mask")


def is_on_gpu(model: torch.nn.Module) -> bool:
    """
    Returns True if all parameters of a model live on the GPU.
    """
    assert isinstance(model, torch.nn.Module)
    on_gpu = True
    has_params = False
    for param in model.parameters():
        has_params = True
        if not param.data.is_cuda:
            on_gpu = False
    return has_params and on_gpu


def is_not_none(sample: Any) -> bool:
    """
    Returns True if sample is not None and constituents are not none.
    """
    if sample is None:
        return False

    if isinstance(sample, (list, tuple)):
        if any(s is None for s in sample):
            return False

    if isinstance(sample, dict):
        if any(s is None for s in sample.values()):
            return False
    return True
