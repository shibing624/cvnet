# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

sys.path.append("../../..")
from cvnet.models.vae.basic_vae_module import VAE
from cvnet.models.vae.cifar10_datamodule import CIFAR10DataModule

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    pl.seed_everything()

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10"])
    script_args, _ = parser.parse_known_args()

    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    else:
        raise ValueError("undefined dataset")

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(os.path.join("~", ".pytorch/datasets/CIFAR10"))
    args.max_epochs = 10
    if torch.cuda.is_available():
        args.gpus = 4
    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    model = VAE(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
