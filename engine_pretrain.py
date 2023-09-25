# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.transforms import Resize

import random

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    print("dataloader size:", len(data_loader), "dataset size:", len(data_loader.dataset))

    for data_iter_step, samples in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # because for fused 2d, finetuned on 2d images with 3 channels
        samples = samples["image"].repeat(1,3,1,1,1)

        print("shape:", samples.shape)
        print("data_iter_step:", data_iter_step)

        if data_iter_step == 1:
            print("samples shape:",samples.shape)

        samples.to(device, non_blocking=True)
        print("device:", device)

        with torch.cuda.amp.autocast():
            loss, _, mask = model(samples, mask_ratio=args.mask_ratio)

        loss = (loss * mask).sum() / mask.sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_value /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # print("okay here")
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # if torch.cuda.is_available():
        #     # Get the current GPU device
        #     device = torch.cuda.current_device()

        #     # Get GPU memory usage in bytes
        #     gpu_memory_bytes = torch.cuda.memory_allocated(device)

        #     # Convert bytes to MB or GB for better readability
        #     gpu_memory_mb = gpu_memory_bytes / (1024 ** 2)
        #     gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)

        #     print(f"GPU memory used: {gpu_memory_mb:.2f} MB or {gpu_memory_gb:.2f} GB")
        # else:
        #     print("CUDA is not available.")
