# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy

sys.path.append('../..')

from cvnet.utils.logger import logger


def inference(
        cfg,
        model,
        val_loader
):
    logger.info("Start infer")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg.MODEL.DEVICE == 'cuda' else 'cpu'

    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()},
                                            device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_acc = metrics['accuracy']
        logger.info("Validation Results - Accuracy: {:.3f}".format(avg_acc))

    evaluator.run(val_loader)


if __name__ == '__main__':
    logger.info('infer')
