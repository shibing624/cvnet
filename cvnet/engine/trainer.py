# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

这里介绍下，利用Ignite进行训练的基本思路：

1.构建class Net(nn.Module)网络结构
2.定义函数get_data_loaders(train_batch_size, val_batch_size), 返回train_loader，val_loader
3.创建函数create_summary_writer(model，data_loader, dog_dir) ，返回的是一个SummaryWriter对象
4.定义run(train_batch_size，val_batch_size, epochs, lr, momentum, log_interval, log_dir)
    这个函数要定义 train_loader、val_batch_size 、模型、SummaryWriter对象、优化器、trainer、evaluator、
    定义handles，通过事件触发程序
5.运行函数

"""
import sys
import time

import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage

sys.path.append('../..')

from cvnet.utils.logger import logger


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg.MODEL.DEVICE == 'cuda' else 'cpu'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info("Start training")
    logger.info("use {}".format(device))
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'ce_loss': Loss(loss_fn)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, 'mnist', n_saved=10, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer, {'model': model})

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['ce_loss']
        logger.info("Training Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['ce_loss']
            logger.info("Validation Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                        .format(engine.state.epoch, avg_accuracy, avg_loss)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式，自动关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, model_path=''):
    net = net.to(device)
    print("device:", device)
    loss = torch.nn.CrossEntropyLoss()
    test_accs = []
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if len(test_accs) == 0 or test_acc > max(test_accs):
            if model_path:
                print('best model, saved.', model_path)
                torch.save(net.state_dict(), model_path)
        test_accs.append(test_acc)


if __name__ == '__main__':
    logger.info('kkj')
