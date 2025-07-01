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

# Imports from PyTorch.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import numpy as np
import sys

from utils.logger import Logger
# sys.path.insert(0, '../aihwkit/src/')
import aihwkit
print('aihwkit path: ', aihwkit.__file__)
# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda
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

parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Add command line arguments
parser.add_argument('-SETTING', '--SETTING', type=str, help="", default='FP SGD')
parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
parser.add_argument('-tau', '--tau', type=float, help="", default=1)
parser.add_argument('-TTAWDC', '--TTv1-active-weight-decay-count', type=int, help="", default=0)
parser.add_argument('-TTAWDP', '--TTv1-active_weight_decay_probability', type=float, help="", default=0)
parser.add_argument('-save', '--save-checkpoint', action='store_true')
parser.add_argument('-Tcolumn', '--Tcolumn', type=int, help="", default='1')
parser.add_argument('-ns', '--ns', type=int, help="", default='1')
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
EPOCHS = 80
BATCH_SIZE = 8
N_CLASSES = 10

tau = args.tau
# DEVICE_NAME = 'PCM'
# DEVICE_NAME = 'HfO2'
# DEVICE_NAME = 'OM'
DEVICE_NAME = 'Softbounds'
# DEVICE_NAME = 'RRAM-offset'

if 'digital' in setting:
    lr = 0.1
else:
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
    
    
    # from aihwkit.utils.visualization import plot_device_compact
    # plot_device_compact(device_config_fit, n_traces=3, n_steps=200)
    return device_config_fit
   
def get_device(device_name='CS'):
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        # by defauly
        # dw_min = 0.001
        # w_max = 0.6 w_min = -0.6
        # state = 1200
        # dw_min = tau / 600
        return SoftBoundsReferenceDevice(
            # dw_min=dw_min,
            w_max=tau, w_min=-tau, construction_seed=10)
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
def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

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
    # model = convert_to_analog(model, rpu_config)
    # model = convert_to_digital(model)
    # print(model)
    return model
def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    return optimizer
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
def train(model, train_set, config, logger, checkpoint_path):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    optimizer_cls = config['optimizer_cls']
    
    classifier = nn.NLLLoss()
    # optimizer = create_sgd_optimizer(model)
    optimizer = optimizer_cls(model.parameters())
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = StepLR(optimizer, step_size=35, gamma=0.1)  # StepLR scheduler with steps at 35, 70, and 80 epochs
    
    # num_len = len(str(self.total_iterations))
    # num_format = '{:>' + f'{num_len}' + 'd}'
    # hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
    #     'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'

    time_init = time()
    test_loss, test_accuracy = test_evaluation(model, validation_dataset)
    log_str = f"Epoch {0} - Training loss: --------   Test Accuracy: {test_accuracy:.4f}"
    logger.write(0, log_str, {
        "Loss/test": test_loss,
        "Accuracy/test": test_accuracy
    })
        
    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        # for _ in range(10000):
        #     scheduler.step()
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Flatten MNIST images into a 784 vector.
            # images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            total_loss += loss.item()

        # Decay learning rate if needed.
        scheduler.step()

        train_loss = total_loss / len(train_set)
        test_loss, test_accuracy = test_evaluation(model, validation_dataset)
        
        log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        logger.write(epoch, log_str, {
            "Loss/train": train_loss,
            "Loss/test": test_loss,
            "Accuracy/test": test_accuracy,
            "State/lr": scheduler.get_last_lr()[0],
        })

    # 保存模型权重和优化器状态
        if args.save_checkpoint:
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
def get_config(config_name):
    if config_name == 'FP SGD':
        # FPSGD
        config = {
            'name': 'FP SGD',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
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
            'batch_size': BATCH_SIZE,
        }
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
    elif config_name == 'TT-v1':
        active_weight_decay_count = args.TTv1_active_weight_decay_count
        active_weight_decay_probability = args.TTv1_active_weight_decay_probability
        active_weight_decay_count = args.TTv1_active_weight_decay_count
        algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        # rpu_config.device.fast_lr = 0.01* args.Tcolumn/args.ns
        rpu_config.device.fast_lr = 0.01/args.ns
        print(rpu_config.device.fast_lr)
        # default n_reads_per_transfer = 1
        rpu_config.device.transfer_every = args.ns
        rpu_config.device.n_reads_per_transfer = args.Tcolumn
        rpu_config.device.no_self_transfer = False
        if active_weight_decay_count != 0 or active_weight_decay_probability != 0:
            rpu_config.device.active_weight_decay_count = active_weight_decay_count
            rpu_config.device.active_weight_decay_probability = active_weight_decay_probability
        # rpu_config.device.auto_granularity=15000
        
        config = {
            'name': f'TT-v1',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
        if active_weight_decay_count != 0:
            config['name'] += f'-T={active_weight_decay_count}'
        elif active_weight_decay_probability > 0:
            config['name'] += f'-T={active_weight_decay_probability}'
            
        if rpu_config.device.n_reads_per_transfer > 1:
            config['name'] += f'-st={rpu_config.device.n_reads_per_transfer}'
    elif config_name == 'TT-v2':
        algorithm = 'ttv2'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.1
        rpu_config.mapping.weight_scaling_omega = 0.3
        # rpu_config.device.fast_lr = 0.01
        rpu_config.device.fast_lr = 0.5
        # rpu_config.device.auto_granularity=15000
        rpu_config.device.auto_granularity=1000
        # rpu_config.device.auto_granularity=800
        
        
        config = {
            'name': f'TT-v2',
            # 'name': f'TT-v2-omega={rpu_config.mapping.weight_scaling_omega}',
            # 'name': f'TT-v2-flr={rpu_config.device.fast_lr}',
            # 'name': f'granularity={rpu_config.device.auto_granularity}',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
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
            'batch_size': BATCH_SIZE,
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
            'batch_size': BATCH_SIZE,
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
            'batch_size': BATCH_SIZE,
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
train_dataset, validation_dataset = load_images()

# Prepare the model.
model = create_analog_network(rpu_config=rpu_config)

# for images, labels in train_dataset:
#     get_model_size(model, images)

# Train the model.
train(model, train_dataset, config, logger, check_point_path)

# Evaluate the trained model.
test_evaluation(model, validation_dataset)


# if __name__ == "__main__":
#     # Execute only if run as the entry point into the program.
#     main()
