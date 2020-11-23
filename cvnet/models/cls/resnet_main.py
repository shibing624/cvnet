# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append('../..')
from cvnet.configs import cfg
from cvnet.dataset.mnist import load_data_mnist
from cvnet.engine.trainer import do_train, make_optimizer
from cvnet.models.cls.resnet_model import build_model

from cvnet.engine.inference import inference

from cvnet.utils.logger import logger


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Train and Infer")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--phase", default="train", help="train/test", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    print(args)
    return args


def train(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if not output_dir:
        output_dir = "./checkpoints/"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg)
    optimizer = make_optimizer(cfg, model)

    train_loader = load_data_mnist(cfg, is_train=True)
    val_loader = load_data_mnist(cfg, is_train=False)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        F.cross_entropy,
    )


def test(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg)
    model_path = cfg.TEST.WEIGHT
    if not model_path:
        model_path = "./checkpoints/mnist_model_938.pt"
    model.load_state_dict(torch.load(model_path))
    val_loader = load_data_mnist(cfg, is_train=False)

    inference(cfg, model, val_loader)


if __name__ == '__main__':
    args = get_args()
    if args.phase == "train":
        train(args)
    elif args.phase == "test":
        test(args)
