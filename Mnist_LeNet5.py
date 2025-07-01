# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# %%
"""aihwkit example 3: MNIST training.

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

"""
# pylint: disable=invalid-name

import os
from time import time
from dataclasses import dataclass, field
# Imports from PyTorch.
import torch
torch.autograd.set_detect_anomaly(True)
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import numpy as np
import sys
from torch.optim.lr_scheduler import LambdaLR
from utils.logger import Logger
from collections import deque
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AIHWKIT_SRC = os.path.join(CURRENT_DIR, 'aihwkit')
sys.path.insert(0, AIHWKIT_SRC)
# For warm start
import aihwkit
print('aihwkit path: ', aihwkit.__file__)
from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda
import aihwkit.simulator.rpu_base.devices as dev
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.nn.conversion import convert_to_analog, convert_to_digital
from aihwkit.simulator.configs import (
    build_config,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
    FloatingPointRPUConfig,
    SingleRPUConfig,
    UpdateParameters,
)
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice,
    ConstantStepDevice,
    VectorUnitCell,
    LinearStepDevice,
    SoftBoundsDevice,
    SoftBoundsReferenceDevice,
    TransferCompound,
    MixedPrecisionCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
)
import argparse
from enum import Enum

parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Add command line arguments
parser.add_argument('-SETTING', '--SETTING', type=str, help="", default='FP SGD')
parser.add_argument('-BATCH_SIZE', '--BATCH_SIZE', type=int, help="", default='8')
parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
parser.add_argument('-tau', '--tau', type=float, help="", default=1)
parser.add_argument('-TTAWDC', '--TTv1-active-weight-decay-count', type=int, help="", default=0)
parser.add_argument('-TTAWDP', '--TTv1-active_weight_decay_probability', type=float, help="", default=0)
parser.add_argument('-save', '--save-checkpoint', action='store_true')
parser.add_argument('-Tcolumn', '--Tcolumn', type=int, help="", default='1')
parser.add_argument('-ns', '--ns', type=float, help="", default='1')
parser.add_argument('-sigma', '--sigma', type=float, help="", default='0.3')
parser.add_argument('-gamma', '--gamma', type=float, help="", default='0')
parser.add_argument('-Wmax', '--Wmax', type=float, help="", default='1')
parser.add_argument('-dwmin', '--dwmin', type=float, help="", default='0.1')
# IO precision and noise parameters
parser.add_argument('--io_inp_res_bit', type=float, default='7')
parser.add_argument('--io_out_res_bit', type=float, default='9')
parser.add_argument('--io_inp_noise', type=float, default='0.0')
parser.add_argument('--io_out_noise', type=float, default='0.0')
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--io_perfect_forward', type=str2bool, default=True)
parser.add_argument('--io_perfect_backward', type=str2bool, default=True)
checkpoint_path = "/home/jindan/Desktop/analog/checkpoints/MNIST-CNN/Softbounds/TT-v1-tile=6-alg2--6-state4-dataset-tau0.5.pth"
import os

print("Checking path:", checkpoint_path)
print("Exists?", os.path.exists(checkpoint_path))

args = parser.parse_args()
setting = args.SETTING

# Check device
USE_CUDA = 0
if cuda.is_compiled() and args.CUDA >= 0:
    USE_CUDA = 1
DEVICE = torch.device(f"cuda:{args.CUDA}" if USE_CUDA else "cpu")
print('Using Device: ', DEVICE)


# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data")

# Training parameters.
EPOCHS = 200
N_CLASSES = 10

tau = args.tau
# DEVICE_NAME = 'PCM'
# DEVICE_NAME = 'HfO2'
# DEVICE_NAME = 'OM'
DEVICE_NAME = 'Softbounds'
# DEVICE_NAME = 'RRAM-offset'

lr = 0.05


def get_model_size(model, input):
    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    raise SystemExit
    return macs, params
    
def get_RRAM():
    from pandas import read_csv
    from numpy import ones, absolute, concatenate, tile, average, sqrt
    df = read_csv('data/IEDM_2022.csv', header=None)
    response = df[0].values
    response -= response.mean()
    response /= absolute(average(response[180:200]))
    
    device_config = SoftBoundsReferenceDevice()
    params = {
        'dw_min': (0.1, 0.001, 1.0),
        'up_down': (0.0, -0.99, 0.99),
        'w_max': (1.0, 0.1, 2.0),
        'w_min': (-1.0, -2.0, -0.1),
    }
    from aihwkit.utils.fitting import fit_measurements
    up_down = [1, -1]
    up_down = tile(up_down, 100)
    pulses = concatenate([ones(200), -ones(200), up_down])

    result, device_config_fit, best_model_fit = fit_measurements(
        params,
        pulses,
        response,
        device_config)
    # print(device_config_fit)
    std = (response[500:] - best_model_fit[500:]).std() / device_config_fit.dw_min
    device_config_fit.dw_min_dtod = 0.3
    device_config_fit.dw_min_std = 5.0
    device_config_fit.w_min_dtod = 0.1
    device_config_fit.w_max_dtod = 0.1
    device_config_fit.up_down_dtod = 0.05
    device_config_fit.write_noise_std = sqrt(std ** 2 - device_config_fit.dw_min_std ** 2)/2
    device_config_fit.subtract_symmetry_point = True
    device_config_fit.reference_std = 0.05  # assumed programming error of the reference device
    device_config_fit.enforce_consistency=True  # do not allow dead devices
    device_config_fit.dw_min_dtod_log_normal=True # more realistic to use log-normal
    device_config_fit.mult_noise=False # additive noise
    device_config_fit.construction_seed = 123
    return device_config_fit
   
def get_device(device_name='CS'):
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        return SoftBoundsReferenceDevice(
           dw_min = 0.2,
            w_max=1, w_min=-1, construction_seed=10)
        # return SoftBoundsDevice(construction_seed=10)
    elif device_name == 'LS':
        return LinearStepDevice(w_max_dtod=0.4)
    elif device_name == 'RRAM':
        return get_RRAM()
    elif device_name == 'HfO2':
        from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
        return ReRamArrayHfO2PresetDevice()
    elif device_name == 'OM':
        from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice
        return ReRamArrayOMPresetDevice()
    elif device_name == 'PCM':
        from aihwkit.simulator.presets.devices import PCMPresetDevice
        return PCMPresetDevice()
    else:
        raise NotImplemented

def load_images(fraction=0.1, seed=None):
    """Load a random fraction of MNIST images for training and validation."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load full datasets
    train_set_full = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set_full = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)

    # Compute subset sizes
    num_train = int(len(train_set_full) * fraction)
    num_val = int(len(val_set_full) * fraction)

    # Optionally fix random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Randomly sample subset indices
    train_indices = random.sample(range(len(train_set_full)), num_train)
    val_indices = random.sample(range(len(val_set_full)), num_val)

    # Create subsets
    train_subset = Subset(train_set_full, train_indices)
    val_subset = Subset(val_set_full, val_indices)

    # Wrap into DataLoaders
    train_data = DataLoader(train_subset, batch_size=args.BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(val_subset, batch_size=args.BATCH_SIZE, shuffle=True)

    return train_data, validation_data

def create_analog_network(rpu_config):
    """Return a LeNet5 inspired analog model."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=rpu_config,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda(DEVICE)
    return model
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)

triggered_count = 0
loss_history = []

def train(model, train_set, config, logger, checkpoint_path):
    """Train the network."""
    optimizer_cls = config['optimizer_cls']
    classifier = nn.NLLLoss()
    optimizer = optimizer_cls(model.parameters())

    def lr_lambda(epoch):
        return 0.1 ** (epoch // 50)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

   
    # aggresive
    def aggressive_plateau(history, threshold=0.0001):
        if len(history) < 2:
            return False
        delta = history[-2] - history[-1]
        plateau = -delta > threshold
        print(f"[Aggressive] Δ={delta:.6f}, plateau={plateau}")
        return plateau

    # smooth
    def smooth_plateau(history, threshold=-0.01, window=5, max_violations=2):
        if len(history) < window + 1:
            return False
        recent = history[-(window + 1):]
        violations = 0

        for i in range(window):  
            delta = recent[i+1] - recent[i]
            if delta >= threshold:
                violations += 1
            print(f"Δ = {delta:.6f}, upward violation = {delta > threshold}")
        return violations >= max_violations


    def trigger_tile_switch_by_plateau(model, loss_history, aggressive_tile_count=4): 
        print("\n[Tile switch check]")
        global triggered_count
        any_triggered = False  # track if any tile was actually triggered this round

        for i, (name, module) in enumerate(model.named_modules()):
            if hasattr(module, "analog_module"):
                tile = module.analog_module.tile
                if triggered_count < aggressive_tile_count:
                    plateau = aggressive_plateau(loss_history)
                else:
                    plateau = smooth_plateau(loss_history)
                tile.set_flags(plateau)
                print(f"[Tile {i}] {name}: trigger_tile_switch_flag = {plateau}")
                if plateau:
                    any_triggered = True

        if any_triggered:
            triggered_count += 1
            loss_history.clear()  # Reset history to wait for new tile to accumulate fresh stats
            print("Tile switch triggered → Resetting loss history.")
            print("Current loss history:", loss_history)  
        print(f"\n=> Total triggered tiles = {triggered_count}")
        return triggered_count

    # --- Load checkpoint if exists ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1  # resume from next epoch
        # logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Training from scratch.")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_set)
        loss_history.append(train_loss)
        trigger_tile_switch_by_plateau(model, loss_history)
        # For warm start
        test_loss, test_accuracy = test_evaluation(model, validation_dataset)

        log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        

        logger.write(epoch, log_str, {
            "Loss/train": train_loss,
            "Loss/test": test_loss,
            "Accuracy/test": test_accuracy,
            "State/lr": scheduler.get_last_lr()[0],
            # "Tile/switch": float(plateau_triggered),  # optional: also track in tensorboard
        })

        if args.save_checkpoint:
            alg_name = config['name']
            alg_name += f'-tile={num_tile}'
            # alg_name += f'-scale_lr={True}'
            alg_name += f'-alg2--6-state10-dataset-tau0.3'
            path_name = f'{dataset_name}/{DEVICE_NAME}'
            check_point_folder = f'checkpoints/{path_name}'
            check_point_path = f'{check_point_folder}/{alg_name}.pth'
            if not os.path.isdir(check_point_folder):
                os.makedirs(check_point_folder)
            print(f'Saving model checkpoint to {check_point_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, check_point_path)

    print("\nTraining Time (s) = {}".format(time() - time_init))


@torch.no_grad()
def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()
    classifier = nn.NLLLoss()

    total_loss = 0
    for images, labels in val_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        loss = classifier(pred, labels)
        total_loss += loss.item()

    # print("\nNumber Of Images Tested = {}".format(total_images))
    # print("Model Accuracy = {}".format(predicted_ok / total_images))
    loss = total_loss / total_images
    accuracy = predicted_ok / total_images
    return loss, accuracy
def get_AnalogSGD_optimizer_generator(lr=lr, *args, **kargs):
    def _generator(params):
        return AnalogSGD(params, lr=lr, *args, **kargs)
    return _generator

construction_seed = 23
num_tile = 7
def config_IO(io_param: IOParameters, config): 
    if config["io_perfect"]:
        io_param.is_perfect = True
    else:
        if config["io_inp_res_bit"] != -1:
            io_param.inp_res = config["io_inp_res_bit"]
        if config["io_out_res_bit"] != -1:
            io_param.out_res = config["io_out_res_bit"]
        if config["io_inp_noise"] != -1:
            io_param.inp_noise = config["io_inp_noise"]
        if config["io_out_noise"] != -1:
            io_param.out_noise = config["io_out_noise"]
    print(io_param.out_res)
    print(args.io_perfect_forward)

def get_config(config_name):
    if config_name == 'FP SGD':
        # FPSGD
        config = {
            'name': 'FP SGD',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }
    elif config_name == 'FP SGDM':
        # FP GDM
        # Set the `batch_size` as full batch
        config = {
            'name': 'FPSGDM',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(momentum=0.99),
            'grad_per_iter': 1,
            # 'batch_size': DATASET_SIZE,
            'linestyle': '--',
        }
    elif config_name == 'Analog SGD':
        # Analog SGD
        rpu_config = SingleRPUConfig(
            # device=get_device('SB')
            device=get_device(DEVICE_NAME)
        )
        config = {
            'name': 'Analog SGD',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }
    elif config_name == 'ResL': 
        rpu_config = UnitCellRPUConfig(
            device=TransferCompound(
                unit_cell_devices=[
                    SoftBoundsDevice(w_min=-1, w_max=1, dw_min=0.5)
                    for _ in range(num_tile)
                ],
                units_in_mbatch=True,
                # transfer_every_vec=[5*2**(num_tile) if i == 0 else 5*2**(num_tile)*(2**i) for i in range(num_tile)],
                transfer_every_vec=[2 * (5**n) for n in range(num_tile)],
                gamma_vec=[0.5**(num_tile - 1 - i) for i in range(num_tile)],
                transfer_lr_vec=[0.1 if n == 0 else 0.1*1.2**n for n in range(num_tile - 1, -1, -1)],
                scale_transfer_lr=False,
                transfer_columns=True,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
        )

        rpu_config.mapping.learn_out_scaling = False
        rpu_config.mapping.weight_scaling_columnwise = True

        config_IO(rpu_config.forward, {
            "io_perfect":     args.io_perfect_forward,
            "io_inp_res_bit": 1/(2**args.io_inp_res_bit-2),
            "io_out_res_bit": 1/(2**args.io_out_res_bit-2),
            "io_inp_noise":   args.io_inp_noise,
            "io_out_noise":   args.io_out_noise,
        })

        config_IO(rpu_config.backward, {
            "io_perfect":     args.io_perfect_backward,
            "io_inp_res_bit": 1/(2**args.io_inp_res_bit-2),
            "io_out_res_bit": 1/(2**args.io_out_res_bit-2),
            "io_inp_noise":   args.io_inp_noise,
            "io_out_noise":   args.io_out_noise,
        })

        config_IO(rpu_config.device.transfer_forward, {
            "io_perfect":     args.io_perfect_backward,  
            "io_inp_res_bit": 1/(2**args.io_inp_res_bit-2),
            "io_out_res_bit": 1/(2**args.io_out_res_bit-2),
            "io_inp_noise":   args.io_inp_noise,
            "io_out_noise":   args.io_out_noise,
        })

        config = {
            'name': 'ResL',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }

    elif config_name == 'TT-v1':
        active_weight_decay_count = args.TTv1_active_weight_decay_count
        active_weight_decay_probability = args.TTv1_active_weight_decay_probability
        algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device("Softbounds")
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr =  0.1
        rpu_config.device.n_reads_per_transfer = 1
        # rpu_config.device.no_self_transfer = (not args.TTv1_self_transfer)
        if active_weight_decay_count != 0:
            rpu_config.device.active_weight_decay_count = active_weight_decay_count
        if active_weight_decay_probability != 0:
            rpu_config.device.active_weight_decay_probability = active_weight_decay_probability
        
        config = {
            'name': f'TT-v1',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
        if active_weight_decay_count != 0:
            config['name'] += f'-T={active_weight_decay_count}'
        elif active_weight_decay_probability > 0:
            config['name'] += f'-T={active_weight_decay_probability}'
            
        if rpu_config.device.n_reads_per_transfer > 1:
            config['name'] += f'-st={rpu_config.device.n_reads_per_transfer}'
        if not rpu_config.device.no_self_transfer:
            config['name'] += f'-stran'
    elif config_name == 'TT-v2':
        rpu_config = UnitCellRPUConfig(
            device=BufferedTransferCompound(
                # Devices that compose the Tiki-taka compound.
                unit_cell_devices=[
                    SoftBoundsDevice(w_min=-args.Wmax, w_max=args.Wmax,dw_min = args.dwmin),
                    SoftBoundsDevice(w_min=-args.Wmax, w_max=args.Wmax,dw_min = args.dwmin),
                ],
                # Make some adjustments of the way Tiki-Taka is performed.
                units_in_mbatch=True,  # batch_size=1 anyway
                transfer_every=1,  # every ns batches do a transfer-read
                n_reads_per_transfer=1,  # one forward read for each transfer
                # gamma_vec = [(args.dwmin/args.Wmax),1],
                scale_transfer_lr=True,  # in relative terms to SGD LR
                # transfer_lr_vec=[0.1,1,10], 
                transfer_lr=1, # same transfer LR as for SGD
                fast_lr=0.1,  # SGD update onto first matrix constant
                transfer_columns=True,  # transfer use columns (not rows)
                thres_scale= args.Wmax,  # Threshold on H
                # transfer_update=up_parameters(
                #      update_bl_management=True, update_management=True
                # )
            ),
            update=UpdateParameters(desired_bl=10,update_bl_management=True, update_management=True),
        )     
        # update onto A matrix needs to be increased somewhat
        rpu_config.backward.is_perfect = True
        rpu_config.forward.is_perfect = True
        rpu_config.device.transfer_forward.is_perfect = True
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        config = {
            'name': f'TT-v2',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }

    elif config_name == 'TT-v3':
        algorithm = 'ttv3'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.1
        rpu_config.device.out_chop_prob = 0.
        rpu_config.device.auto_scale = True
        # rpu_config.device.momentum=0.8

        
        config = {
            'name': f'TT-v3',
            # 'name': f'flr={rpu_config.device.fast_lr}-omega={rpu_config.mapping.weight_scaling_omega}-granularity={rpu_config.device.auto_granularity}-inc={rpu_config.device.in_chop_prob}-outc={rpu_config.device.out_chop_prob}-m={rpu_config.device.momentum}',
            # 'name': f'flr={rpu_config.device.fast_lr}-omega={rpu_config.mapping.weight_scaling_omega}-granularity={rpu_config.device.auto_granularity}-inc={rpu_config.device.in_chop_prob}-outc={rpu_config.device.out_chop_prob}-autoscale',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }
    elif config_name == 'TT-v4':
        algorithm = 'ttv4'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.1
        rpu_config.device.out_chop_prob = 0.
        rpu_config.device.auto_scale = True
        rpu_config.device.tail_weightening = 10
        # rpu_config.device.momentum=0.8
        
        config = {
            'name': f'TT-v4',
            # 'name': f'flr={rpu_config.device.fast_lr}-omega={rpu_config.mapping.weight_scaling_omega}-granularity={rpu_config.device.auto_granularity}-inc={rpu_config.device.in_chop_prob}-outc={rpu_config.device.out_chop_prob}-tw={rpu_config.device.tail_weightening}-autoscale',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }
    elif config_name == 'mp':
        algorithm = 'mp'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)

        config = {
            'name': f'mp',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': args.BATCH_SIZE,
        }
    else:
        raise NotImplementedError
    # rpu_config = SingleRPUConfig(device=ConstantStepDevice())
    return config


config = get_config(
    setting
    # 'FPSGD',
    # 'FPSGDM',
    # 'TT-v1'
    # 'TT-v2'
    # 'TT-v3'
    # 'TT-v4'
    # 'mp',
    # 'AnalogSGD'
)

no_tau_list = ['FP SGD']
dataset_name = 'MNIST-CNN'
name = config['name']
if config['name'] not in no_tau_list:
    name += f'-tau={tau}'
# path_name = f'{dataset_name}/{DEVICE_NAME}'
# path_name = f'{dataset_name}/TT-AW-tuning'
path_name = f'{dataset_name}/TT-AW-no-fit-state'

# name = config['name'] + '-test'
# path_name = f'{dataset_name}/{DEVICE_NAME}-{setting}'

rpu_config = config['rpu_config']

check_point_folder = f'checkpoints/{path_name}'
check_point_path = f'{check_point_folder}/{name}.pth'
log_path = f'runs/{path_name}/{name}'
logger = Logger(log_path)
if args.save_checkpoint and not os.path.isdir(check_point_folder):
    os.makedirs(check_point_folder)

# def main():
"""Train a PyTorch analog model with the MNIST dataset."""
# Load datasets.
train_dataset, validation_dataset = load_images(fraction=1)

# Prepare the model.
model = create_analog_network(rpu_config=rpu_config)

train(model, train_dataset, config, logger, check_point_path)

# Evaluate the trained model.
test_evaluation(model, validation_dataset)
