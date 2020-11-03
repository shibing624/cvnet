# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import copy
import torchvision.models as models

from cvnet.models.fcn import Fcn8s, Fcn16s, Fcn32s
from cvnet.models.segnet import Segnet
from cvnet.models.unet import Unet
# from ptsemseg.models.pspnet import pspnet
# from ptsemseg.models.icnet import icnet
# from ptsemseg.models.linknet import linknet
from cvnet.models.frrn import FRRN


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model_cls = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    name = name.lower()
    if name in ["frrnA", "frrnB"]:
        model = model_cls(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model_cls(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model_cls(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model_cls(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model_cls(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model_cls(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model_cls(n_classes=n_classes, **param_dict)

    else:
        model = model_cls(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": Fcn32s,
            "fcn8s": Fcn8s,
            "fcn16s": Fcn16s,
            "unet": Unet,
            "segnet": Segnet,
            # "pspnet": pspnet,
            # "icnet": icnet,
            # "icnetBN": icnet,
            # "linknet": linknet,
            "frrnA": FRRN,
            "frrnB": FRRN,
        }[name]
    except:
        raise ("Model {} not available".format(name))
