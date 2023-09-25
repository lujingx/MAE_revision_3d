# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import Resize

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    SpatialPadd,
    Resized,
)
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, ThreadDataLoader
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    train_transforms = Compose(
        [   
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            # Resized(keys=["image"], spatial_size=(224,224,64), mode=("bilinear")),
            Resized(keys=["image"], spatial_size=224, mode=("bilinear"), size_mode="longest"),
            SpatialPadd(keys=["image"], spatial_size=(224,224,224), method="end"),
            # Resized(keys=["image"], spatial_size=(224,128,224), mode=("bilinear")),
            EnsureTyped(keys=["image"], device=device, track_meta=False, dtype=torch.float16),
        ]
    )
    val_transforms = Compose(
        [   
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            # Resized(keys=["image"], spatial_size=(224,224,64), mode=("bilinear")),
            Resized(keys=["image"], spatial_size=224, mode=("bilinear"), size_mode="longest"),
            SpatialPadd(keys=["image"], spatial_size=(224,224,224), method="end"),
            # Resized(keys=["image"], spatial_size=(224,128,224), mode=("bilinear")),
            EnsureTyped(keys=["image"], device=device, track_meta=False),
        ]
    )

    train_images = sorted(glob.glob(os.path.join(args.data_path, "*/imagesTr", "*.nii.gz")))
    # train_labels = sorted(glob.glob(os.path.join(args.data_path, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name} for image_name in train_images]
    index = [2,12,32,42,52,62,72,82,92,102,112,122,132,142,152,162,172,182,192,202,212,222,232,242,252,262,272,282,292,302,312,322,332,342,352,362,372,382,392,402.412,422,432,442,452,462,472,482,492]
    j=0
    k=0
    train_files = dict()
    val_files = dict()
    for i in range(len(data_dicts)):
        if i not in index:
            train_files[j] = data_dicts[i]
            j += 1
        else:
            val_files[k] = data_dicts[i]
            k += 1

    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.8, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.8, num_workers=4)

    print(first(train_ds)["image"].shape, first(train_ds)["image"].dtype)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    data_loader_train = ThreadDataLoader(train_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)
    data_loader_val = ThreadDataLoader(val_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)

    check_data = first(data_loader_train)
    print(check_data.keys())

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    if torch.cuda.is_available():
        # Get the current GPU device
        device = torch.cuda.current_device()

        # Get GPU memory usage in bytes
        gpu_memory_bytes = torch.cuda.memory_allocated(device)

        # Convert bytes to MB or GB for better readability
        gpu_memory_mb = gpu_memory_bytes / (1024 ** 2)
        gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)

        print(f"GPU memory used: {gpu_memory_mb:.2f} MB or {gpu_memory_gb:.2f} GB")
    else:
        print("CUDA is not available.")

    model.to(device)

    gpus = torch.cuda.device_count()
    if gpus > 1:
        device_ids = [0,1,2,3]
        print(f"has {gpus} gpus, using devices: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if torch.cuda.is_available():
        # Get the current GPU device
        device = torch.cuda.current_device()

        # Get GPU memory usage in bytes
        gpu_memory_bytes = torch.cuda.memory_allocated(device)

        # Convert bytes to MB or GB for better readability
        gpu_memory_mb = gpu_memory_bytes / (1024 ** 2)
        gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)

        print(f"GPU memory used: {gpu_memory_mb:.2f} MB or {gpu_memory_gb:.2f} GB")
    else:
        print("CUDA is not available.")
        
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # if(args.freeze):
    freezed_num = 0
    pass_num = 0
    freeze_param = []
    if(True):
        # state_dict_xy = torch.load('./checkpoint/finetune2d/checkpoint-400.pth', map_location=torch.device('cpu'))
        # state_dict_xz = torch.load('./checkpoint/finetune2d_xz/checkpoint-100.pth', map_location=torch.device('cpu'))
        # state_dict_yz = torch.load('./checkpoint/finetune2d_yz/checkpoint-120.pth', map_location=torch.device('cpu'))
        mae_dict = torch.load('./mae_visualize_vit_base.pth')
        print(mae_dict["model"].keys())

        new_state_dict = OrderedDict()
        state_dict = model.state_dict()
        print(state_dict.keys())

        for k in mae_dict["model"]:
            # print(k)
            v = mae_dict["model"][k]
            name = "module."+k # add `module.`
            # if k == "module.patch_embed.proj.weight":
            #     value = (torch.mean(state_dict_xy["model"][k],dim=1).unsqueeze(-1).repeat(1,1,1,16)+torch.mean(state_dict_xz["model"][k],dim=1).unsqueeze(-2).repeat(1,1,16,1)+torch.mean(state_dict_xz["model"][k],dim=1).unsqueeze(-3).repeat(1,16,1,1)).unsqueeze(1)
            # else:
            #     value = state_dict_xy["model"][k] + state_dict_xz["model"][k] + state_dict_yz["model"][k]
            if name == 'module.patch_embed.proj.weight' or name == 'module.patch_embed.proj.bias':
                continue
            if v.shape == state_dict[name].shape:
                new_state_dict[name] = v
                # if k != "module.decoder_pos_embed" and k != "module.pos_embed":
                #     freeze_param.append(k)
            else:
                print("del:",k)
                print(v.shape,state_dict[name].shape)

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        for (name, param) in model.named_parameters():
            # print(name)
            if name in freeze_param:
                param.requires_grad = False
            else:
                pass

        for (name, param) in model.named_parameters():
            if param.requires_grad == False:
                print("no grad:", name)
                freezed_num += 1
            else:
                pass_num += 1
        print('\n Total {} params, miss {} \n'.format(freezed_num + pass_num, pass_num))

    # print("model.named_parameters")
    # for name, p in model.named_parameters():
    #     print(name)
    #     print(p)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # print("model.named_parameters")
    # for name, p in model.named_parameters():
    #     print(name)
    #     print(p)
    
    val_loss = []
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 40:
            for (name, param) in model.named_parameters():
            # print(name)
                if name in freeze_param:
                    param.requires_grad = True
                else:
                    pass
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        print("begin validation:")
        if epoch % 10 == 0:
            model.eval()
            is_2d=False
            with torch.no_grad():
                count = 0
                total = 0
                if(is_2d):
                    for val_data in data_loader_val:
                        data = val_data["image"]
                        N,C,H,W,Z = data.shape
                        image = torch.zeros(N*4,C,H,W)
                        index = random.sample(range(Z), 16)
                        print("index:", index)
                        data.to(dtype=torch.float32)
                        print(type(data))

                        for turn in range(4):
                            k=0
                            for i in range(N):
                                for j in range(4*turn,4*turn+4):
                                    print(index[j])
                                    image[k,:,:,:] = data[i,:,:,:,index[j]]
                                    k+=1
                            samples = image.repeat(1,3,1,1)
                            torch_resize = Resize([1024,1024]) # 定义Resize类对象
                            samples = torch_resize(samples)

                            print("shape:", samples.shape)

                            samples.to(device, non_blocking=True)
                            print("device:", device)

                            loss, _, mask = model(samples, mask_ratio=args.mask_ratio)
                            loss = (loss * mask).sum() / mask.sum()
                            loss_value = loss.item()
                            print("loss:",loss,"loss item:", loss_value)
                            total += loss_value
                            count += 1
                else:
                    for val_data in data_loader_val:
                        # because for fused 2d, need to be 3 channels
                        samples = val_data["image"].repeat(1,3,1,1,1)
                        samples.to(device, non_blocking=True)
                        print("device:", device)

                        loss, _, mask = model(samples, mask_ratio=args.mask_ratio)
                        loss = (loss * mask).sum() / mask.sum()
                        loss_value = loss.item()
                        print("loss:",loss,"loss item:", loss_value)
                        total += loss_value
                        count += 1

                log_stats = {**{f'val_loss': total/count},
                        'epoch': epoch,}
                val_loss.append(total/count)
                plt.plot(val_loss)
                plt.savefig('val_loss_interpolate.png')

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
