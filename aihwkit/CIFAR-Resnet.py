import math
import random
import argparse
import time
import os
from enum import Enum

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from utils.logger import Logger


import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AIHWKIT_SRC = os.path.join(CURRENT_DIR, 'aihwkit')
sys.path.insert(0, AIHWKIT_SRC)
# For warm start
import aihwkit
print('aihwkit path: ', aihwkit.__file__)
print('[Running]', ' '.join(sys.argv))
from aihwkit.simulator.parameters.io import IOParameters
# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog
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

parser = argparse.ArgumentParser(description="A simple command-line argument example")
parser.add_argument('--dataset', type = str, default = 'CIFAR10', help = 'use which dataset')
parser.add_argument('-opt', '--optimizer', type=str, help="", default='SGD')
parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
parser.add_argument('-RPU', '--RPU', type=str, help="", default='Softbounds')
parser.add_argument('-save', '--save-checkpoint', action='store_true')
parser.add_argument('--save-every', type=int, default=None)
parser.add_argument('-skip-BN', '--skip-BN', action='store_true')
parser.add_argument('-no-shortcut', '--no-shortcut', action='store_true')
parser.add_argument('-activation', '--activation', type=str, help="", default='Relu')
parser.add_argument('-tau', '--tau', type=float, help="", default=0.8)
parser.add_argument('-ns', '--number-of-states', type=int, help="", default=1200)
parser.add_argument('--io-perfect', '--io-perfect', action='store_true')
parser.add_argument('-io-perfect-F', '--io-perfect-forward', action='store_true')
parser.add_argument('-io-perfect-B', '--io-perfect-backward', action='store_true')
parser.add_argument('-io-perfect-T', '--io-perfect-transfer', action='store_true')
parser.add_argument('--io-inp-res-bit', type=int, help="", default=-1)
parser.add_argument('--io-out-res-bit', type=int, help="", default=-1)
parser.add_argument('--io-inp-noise', type=float, help="", default=-1)
parser.add_argument('--io-out-noise', type=float, help="", default=-1)
parser.add_argument('--update-bl', type=int, help="", default=-1)
parser.add_argument("-block-number", metavar="N", type=int, nargs="+", help=" ")
parser.add_argument("-block-type", metavar="N", type=str, nargs="+", help=" ")
parser.add_argument('-TTAWDC', '--TTv1-active-weight-decay-count', type=int, help="", default=0)
parser.add_argument('-TTAWDP', '--TTv1-active_weight_decay_probability', type=float, help="", default=0)
parser.add_argument('-TTST', '--TTv1-self-transfer', action='store_true')


args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{args.CUDA}")
    # DEVICE = torch.cuda.set_device(args.CUDA)
else:
    DEVICE = torch.device("cpu")
print('Using Device: ', DEVICE)

print('[Save checkpoint]', args.save_checkpoint)
class opt_T(Enum):
    TORCH = 1
    KIT_FP = 2
    KIT_ANALOG = 3

EPOCHS = 200
    
tau = args.tau
number_of_states = args.number_of_states
optimizer_str = args.optimizer
skip_BN = args.skip_BN
no_shortcut = args.no_shortcut
activation_str = args.activation
if args.io_inp_res_bit != -1:
    io_inp_res = 1/(2**args.io_inp_res_bit - 2)
if args.io_out_res_bit != -1:
    io_out_res = 1/(2**args.io_out_res_bit - 2)
# RPU_NAME = 'PCM'
# RPU_NAME = 'HfO2'
# RPU_NAME = 'OM'
# RPU_NAME = 'Softbounds'
# RPU_NAME = 'RRAM-offset'
RPU_NAME = args.RPU

num_blocks_list = args.block_number
blocks_type_list = args.block_type
assert len(num_blocks_list) == 4
assert len(blocks_type_list) == 6
assert activation_str in ['Relu', 'Sigmoid', 'Tanh', 'None'], 'unknow activation'
       
def get_opt_type(optimizer_str):
    torch_list = [
        'SGD', 'SGD-plain', 'AdamW'
        # 'GaLoreAdamW', 'GaLoreSGD', 'BLRSGD'
    ]
    fp_list = [
        'FP SGD', 'FP SGDM'
    ]
    analog_list = [
        'Analog SGD', 'TT-v1', 'ResL', 'TT-v2', 'TT-v3', 'TT-v4', 'mp'
    ]
    if optimizer_str in torch_list:
        return opt_T.TORCH
    elif optimizer_str in fp_list:
        return opt_T.KIT_FP
    elif optimizer_str in analog_list:
        return opt_T.KIT_ANALOG
    else:
        assert False, "unknown algorithm type"
opt_type = get_opt_type(args.optimizer)

lr = 0.2
if opt_type is opt_T.KIT_ANALOG:
    lr /= 2
    
def get_dataset(dataset_name):
    class Cutout(object):
        """Random erase the given PIL Image.
        It has been proposed in
        `Improved Regularization of Convolutional Neural Networks with Cutout`.
        `https://arxiv.org/pdf/1708.04552.pdf`
        Arguments:
            p (float): probability of the image being perspectively transformed. Default value is 0.5
            s_l (float): min cut square ratio. Default value is 0.02
            s_h (float): max cut square ratio. Default value is 0.4
            r_1 (float): aspect ratio of cut square. Default value is 0.4
            r_2 (float): aspect ratio of cut square. Default value is 1/0.4
            v_l (int): low filling num. Default value is 0
            v_h (int): high filling num. Default value is 255
            pixel_level (bool): filling one number or not. Default value is False
        """

        def __init__(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.4, r_2=1 / 0.4,
                    v_l=0, v_h=255, pixel_level=False):
            self.p = p
            self.s_l = s_l
            self.s_h = s_h
            self.r_1 = r_1
            self.r_2 = r_2
            self.v_l = v_l
            self.v_h = v_h
            self.pixel_level = pixel_level

        @staticmethod
        def get_params(img, s_l, s_h, r_1, r_2):

            img_h, img_w = img.size
            img_c = len(img.getbands())
            s = np.random.uniform(s_l, s_h)
            # if you img_h != img_w you may need this.
            # r_1 = max(r_1, (img_h*s)/img_w)
            # r_2 = min(r_2, img_h / (img_w*s))
            r = np.random.uniform(r_1, r_2)
            s = s * img_h * img_w
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)

            return left, top, h, w, img_c

        def __call__(self, img):
            if np.random.rand() > self.p:
                return img

            left, top, h, w, ch = self.get_params(img, self.s_l, self.s_h, self.r_1, self.r_2)

            if self.pixel_level:
                c = np.random.randint(self.v_l, self.v_h, (h, w, ch), dtype='uint8')
            else:
                c = np.random.randint(self.v_l, self.v_h) * np.ones((h, w, ch), dtype='uint8')
            c = Image.fromarray(c)
            img.paste(c, (left, top, left + w, top + h))
            return img

    # from PIL import Image, ImageEnhance, ImageOps
    # https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
    # https://github.com/kakaobrain/fast-autoaugment
    class SubPolicy(object):
        def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
            ranges = {
                "shearX": np.linspace(0, 0.3, 10),
                "shearY": np.linspace(0, 0.3, 10),
                "translateX": np.linspace(0, 150 / 331, 10),
                "translateY": np.linspace(0, 150 / 331, 10),
                "rotate": np.linspace(0, 30, 10),
                "color": np.linspace(0.0, 0.9, 10),
                "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
                "solarize": np.linspace(256, 0, 10),
                "contrast": np.linspace(0.0, 0.9, 10),
                "sharpness": np.linspace(0.0, 0.9, 10),
                "brightness": np.linspace(0.0, 0.9, 10),
                "autocontrast": [0] * 10,
                "equalize": [0] * 10,
                "invert": [0] * 10
            }

            # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
            def rotate_with_fill(img, magnitude):
                rot = img.convert("RGBA").rotate(magnitude)
                return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

            func = {
                "shearX": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                    Image.BICUBIC, fillcolor=fillcolor),
                "shearY": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                    Image.BICUBIC, fillcolor=fillcolor),
                "translateX": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                    fillcolor=fillcolor),
                "translateY": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                    fillcolor=fillcolor),
                "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
                "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
                "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
                "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
                "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
                "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
                "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
                "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
                "equalize": lambda img, magnitude: ImageOps.equalize(img),
                "invert": lambda img, magnitude: ImageOps.invert(img)
            }

            self.p1 = p1
            self.operation1 = func[operation1]
            self.magnitude1 = ranges[operation1][magnitude_idx1]
            self.p2 = p2
            self.operation2 = func[operation2]
            self.magnitude2 = ranges[operation2][magnitude_idx2]


        def __call__(self, img):
            if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
            if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
            return img

    class CIFAR10Policy(object):
        """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
            Example:
            >>> policy = CIFAR10Policy()
            >>> transformed = policy(image)
            Example as a PyTorch Transform:
            >>> transform=transforms.Compose([
            >>>     transforms.Resize(256),
            >>>     CIFAR10Policy(),
            >>>     transforms.ToTensor()])
        """
        def __init__(self, fillcolor=(128, 128, 128)):
            self.policies = [
                SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
                SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
                SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
                SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
                SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

                SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
                SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
                SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
                SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
                SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

                SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
                SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
                SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
                SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
                SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

                SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
                SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
                SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
                SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
                SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

                SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
                SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
                SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
                SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
                SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
            ]


        def __call__(self, img):
            policy_idx = random.randint(0, len(self.policies) - 1)
            return self.policies[policy_idx](img)

        def __repr__(self):
            return "AutoAugment CIFAR10 Policy"
    # transform_train = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.RandomVerticalFlip(),
    #     # torchvision.transforms.Resize(image_size),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #     )
    # ])
    transform_train = transforms.Compose([
        transforms.Pad(4),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32), 
        Cutout(),
        transforms.RandomHorizontalFlip(), 
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True,download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=False,download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/CIFAR100', train=True,download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='data/CIFAR100', train=False,download=True, transform=transform_test)
    else:
        raise ValueError(f"unknown dataset type: {dataset_name}")
    return trainset, testset

if args.dataset == 'CIFAR10':
    trainset, testset = get_dataset(args.dataset)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset, testset = get_dataset(args.dataset)
    num_classes = 100
BATCH_SIZE = 128
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)


def get_RPU_device(device_name=None):
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        return SoftBoundsReferenceDevice(
            dw_min=0.5,
            w_max=1, w_min=-1, 
            w_max_dtod=0., w_min_dtod=0.,
            # construction_seed=10
        )
    elif device_name == 'Exp':
        from aihwkit.simulator.configs.devices import ExpStepDevice
        dw_min = 0.001
        device = ExpStepDevice(
            dw_min=dw_min,
            w_max=tau, w_min=-tau,
            w_max_dtod=0, w_min_dtod=0
        )
        if args.res_gamma > 0:
            device.gamma_up = args.res_gamma
            device.gamma_down = args.res_gamma
        return device
    elif device_name == 'Pow':
        from aihwkit.simulator.configs.devices import PowStepDevice
        dw_min = 0.001
        device = PowStepDevice(
            dw_min=dw_min,
            pow_gamma_dtod=0,
            w_max=tau, w_min=-tau,
            w_max_dtod=0, w_min_dtod=0
        )
        if args.res_gamma > 0:
            device.pow_gamma = args.res_gamma
        return device
    elif device_name == 'LS':
        return LinearStepDevice(w_max_dtod=0.4)
    # elif device_name == 'RRAM':
    #     return get_RRAM()
    elif device_name == 'ReRamSB':
        from aihwkit.simulator.presets.devices import ReRamSBPresetDevice
        return ReRamSBPresetDevice()
    elif device_name == 'ReRamES':
        from aihwkit.simulator.presets.devices import ReRamESPresetDevice
        return ReRamESPresetDevice()
    elif device_name == 'EcRam':
        from aihwkit.simulator.presets.devices import EcRamPresetDevice
        return EcRamPresetDevice()
    elif device_name == 'EcRamMO':
        from aihwkit.simulator.presets.devices import EcRamMOPresetDevice
        return EcRamMOPresetDevice()
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

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label
class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def get_activate():
    if activation_str == 'Relu':
        return nn.ReLU(inplace=True)
    elif activation_str == 'Sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'Tanh':
        return nn.Tanh()
    elif activation_str == 'None':
        return nn.Identity()
    else:
        assert False, 'unknown activation'

def get_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        # Use AnalogConv2d if rpu_config is provided
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
   
class ResidualBlock(nn.Module):
    # Resnet-18 and Resnet-34
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        if skip_BN:
            self.left = nn.Sequential(
                get_conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                get_activate(),
                get_conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    get_conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
        else:
            self.left = nn.Sequential(
                get_conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                get_activate(),
                get_conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    get_conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        self.activation = get_activate()        

    def forward(self, x):
        out = self.left(x)
        if not no_shortcut:
            out += self.shortcut(x)
        out = self.activation(out)
        return out
class BottleneckBlock(nn.Module):
#For Resnet-50 
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels

        self.left = nn.Sequential(
            get_conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            get_activate(),

            get_conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            get_activate(),

            get_conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                get_conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.activation = get_activate()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_blocks_list=[2, 2, 2, 2], num_classes=10,
                 blocks_type_list=['D', 'D', 'D', 'D', 'D', 'D'], 
                 skip_BN=False, rpu_config=None):
        super(ResNet, self).__init__()
        self.inchannel = 64
        if skip_BN:
            self.conv1 = nn.Sequential(
                get_conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                get_activate(),
            )
        else:
            self.conv1 = nn.Sequential(
                get_conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                get_activate(),
            )

        assert len(num_blocks_list) == 4
        assert len(blocks_type_list) == 6
        for block_type in blocks_type_list:
            assert block_type in ['A', 'D'], 'unknown device type'
        self.layer1 = self.make_layer(ResidualBlock, 64,  num_blocks_list[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks_list[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks_list[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks_list[3], stride=2)
        self.fc = nn.Linear(512, num_classes)
        # self.fc = nn.Linear(2048, num_classes)
        # For Resnet-50 
        # self.relu = get_activate()
        self.avg_pool2d = nn.AvgPool2d(4,4)
        
        exclude_modules = [name for name, module in self.named_modules() if "lora" in name]
        print(f"Excluding modules: {exclude_modules}")
        if blocks_type_list[0] == 'A':
            self.conv1 = convert_to_analog(self.conv1, rpu_config, exclude_modules=exclude_modules)
        if blocks_type_list[1] == 'A':
            self.layer1 = convert_to_analog(self.layer1, rpu_config, exclude_modules=exclude_modules)
        if blocks_type_list[2] == 'A':
            self.layer2 = convert_to_analog(self.layer2, rpu_config, exclude_modules=exclude_modules)
        if blocks_type_list[3] == 'A':
            self.layer3 = convert_to_analog(self.layer3, rpu_config, exclude_modules=exclude_modules)
        if blocks_type_list[4] == 'A':
            self.layer4 = convert_to_analog(self.layer4, rpu_config, exclude_modules=exclude_modules)
        if blocks_type_list[5] == 'A':
            self.fc = convert_to_analog(self.fc, rpu_config, exclude_modules=exclude_modules)            
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            # self.inchannel = channels*4
            # For Resnet-50 
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avg_pool2d(out)
        # out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def ResNet8():
    return ResNet(ResidualBlock, num_blocks_list=num_blocks_list)
def ResNet18():
    return ResNet(ResidualBlock)

def get_optimizer(param_groups, optimizer_str):
    if optimizer_str == 'SGD':
        return optim.SGD(param_groups, lr=lr, momentum=0.9)
    elif optimizer_str == 'SGD-plain':
        return optim.SGD(param_groups, lr=lr)
    elif optimizer_str == 'AdamW':
        return optim.AdamW(param_groups, lr=lr, amsgrad=True)
    else:
        assert False, f"unknown optimizer: {optimizer_str}"
def get_AnalogSGD_optimizer_generator(lr=lr, *args, **kargs):
    def _generator(params):
        return AnalogSGD(params, lr=lr, *args, **kargs)
    return _generator
num_tile = 8
# Different versions of Residual Learning
def get_config(config_name):
    if config_name == 'FP SGD':
        # FPSGD
        config = {
            'name': 'FP SGD',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
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
            device=get_RPU_device(RPU_NAME)
        )
        config = {
            'name': 'Analog SGD',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
    elif config_name == 'TT-v1':
        active_weight_decay_count = args.TTv1_active_weight_decay_count
        active_weight_decay_probability = args.TTv1_active_weight_decay_probability
        algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_RPU_device("Softbounds")
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr =  0.1
        rpu_config.device.n_reads_per_transfer = 1
        rpu_config.device.no_self_transfer = (not args.TTv1_self_transfer)
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
    elif config_name == 'ResL':
        rpu_config = UnitCellRPUConfig(
            device=TransferCompound(
                unit_cell_devices=[
                    SoftBoundsDevice(w_min=-1, w_max=1, dw_min=0.5 )
                    for _ in range(num_tile)
                ],
                units_in_mbatch=True,
                transfer_every_vec=[3 * (2**n) for n in range(num_tile)],
                gamma_vec=[0.5**(num_tile - 1 - i) for i in range(num_tile)],
                transfer_lr_vec=[0.3 * (1.2)**n for n in reversed(range(num_tile))],
                scale_transfer_lr=True,
                transfer_columns=True,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
        )

        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True

        config = {
            'name': 'ResL',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }

    elif config_name == 'TT-v2':
        # algorithm = 'ttv2'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        # rpu_device = get_RPU_device(RPU_NAME)
        # rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
        rpu_config = UnitCellRPUConfig(
            device=BufferedTransferCompound(
                unit_cell_devices=[
                    SoftBoundsDevice(w_min=-1, w_max=1, dw_min=0.5)
                    for _ in range(2)
                ],
            )
        )
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.device.n_reads_per_transfer = 1
        rpu_config.device.fast_lr = 0.5
        rpu_config.device.transfer_lr = 1
        rpu_config.device.scale_transfer_lr = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        rpu_config.device.thres_scale= 0.9 
    
        
        config = {
            'name': f'TT-v2',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
        }
        
        if rpu_config.device.n_reads_per_transfer > 1:
            config['name'] += f'gran={rpu_config.device.auto_granularity}'
    elif config_name == 'TT-v3':
        algorithm = 'ttv3'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_device = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
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
        }
    elif config_name == 'TT-v4':
        algorithm = 'ttv4'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_device = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
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
        }
    elif config_name == 'mp':
        # algorithm = 'mp'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_config =DigitalRankUpdateRPUConfig(
            device=MixedPrecisionCompound(device=SoftBoundsDevice(w_min=-1, w_max=1,dw_min = 0.5)),
            # forward=io_parameters(),
            # backward=io_parameters(),
            # update=up_parameters(),
            # **kwargs,
        )
        config = {
            'name': f'mp',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
        }
    else:
        raise NotImplementedError
    
    def config_IO(io_param: "aihwkit.simulator.parameters.IOParameters", config):
        """Set the IO parameters for the config"""
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
                
    rpu_config = config['rpu_config']
    if opt_type is opt_T.KIT_ANALOG:
        if args.io_perfect:
            # perfect IO
            rpu_config.forward.is_perfect = True
            rpu_config.backward.is_perfect = True
            if config_name.startswith('TT'):
                rpu_config.device.transfer_forward.is_perfect = True
        else:
            # imperfect IO
            # for each of forward, backward, transfer, deal with input/output, resistively
            # ===================================================================
            # forward
            config_IO(rpu_config.forward, {
                "io_perfect":     args.io_perfect_forward,
                "io_inp_res_bit": args.io_inp_res_bit,
                "io_out_res_bit": args.io_out_res_bit,
                "io_inp_noise": args.io_inp_noise,
                "io_out_noise": args.io_out_noise,
            })
            # backward
            config_IO(rpu_config.backward, {
                "io_perfect":     args.io_perfect_backward,
                "io_inp_res_bit": args.io_inp_res_bit,
                "io_out_res_bit": args.io_out_res_bit,
                "io_inp_noise": args.io_inp_noise,
                "io_out_noise": args.io_out_noise,
            })
            # transfer
            if config_name.startswith('TT'):
                config_IO(rpu_config.device.transfer_forward, {
                    "io_perfect":     args.io_perfect_backward,
                    "io_inp_res_bit": args.io_inp_res_bit,
                    "io_out_res_bit": args.io_out_res_bit,
                    "io_inp_noise": args.io_inp_noise,
                    "io_out_noise": args.io_out_noise,
                })
    if args.update_bl != -1:
        rpu_config.update.desired_bl = args.update_bl
    return config
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    
def test_evaluation(model, val_set, criterion):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for images, labels in val_set:
            # Predict image.
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # images = images.view(images.shape[0], -1)
            pred = model(images)

            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
            loss = criterion(pred, labels)
            total_loss += loss.item()

    # print("\nNumber Of Images Tested = {}".format(total_images))
    # print("Model Accuracy = {}".format(predicted_ok / total_images))
    loss = total_loss / total_images
    accuracy = predicted_ok / total_images
    return loss, accuracy
triggered_count = 0
loss_history = []
def train():

    # ======================== define record name ========================
    # CIFAR10-Resnet-shallow
    dataset_name = f'{args.dataset}-Resnet-shallow'
    if activation_str != 'Relu':
        dataset_name += '-' + activation_str
    if skip_BN:
        dataset_name += '-sBN'
    if no_shortcut:
        dataset_name += '-woShortcut'
    if opt_type is opt_T.TORCH:
        config = None
        rpu_config = None
        model = ResNet(ResidualBlock, num_classes=num_classes, num_blocks_list=num_blocks_list)
        alg_name = f'torch-{args.optimizer}'
        # path_name = f'{dataset_name}/GPU'

    else:
        # aihwkit training; FP or Analog
        config = get_config(optimizer_str)
        rpu_config = config['rpu_config']
        model = ResNet(ResidualBlock, num_classes=num_classes, num_blocks_list=num_blocks_list, 
                       blocks_type_list=blocks_type_list, rpu_config=rpu_config)
        # model = ResNet(BottleneckBlock, num_classes=num_classes, num_blocks_list=num_blocks_list, 
        #                blocks_type_list=blocks_type_list, rpu_config=rpu_config)
        alg_name = config['name']
        
        if opt_type is opt_T.KIT_ANALOG:
           
            if RPU_NAME == 'Softbounds':
                alg_name += f'-tau={tau}'
            # prefix for number of states
            if number_of_states != 1200:
                alg_name += f'-ns={number_of_states}'
            # prefix for IO
            if args.io_perfect:
                alg_name += f'-ioperfect'
            else:
                if args.io_perfect_forward or args.io_perfect_backward or args.io_perfect_transfer:
                    alg_name += f'-io'
                    if args.io_perfect_forward:
                        alg_name += f'F'
                    if args.io_perfect_backward:
                        alg_name += f'B'
                    if args.io_perfect_transfer:
                        alg_name += f'T'
                else:
                    if args.io_inp_res_bit != -1:
                        alg_name += f'-inpres={args.io_inp_res_bit}'
                    if args.io_out_res_bit != -1:
                        alg_name += f'-outres={args.io_inp_res_bit}'
                if args.io_out_noise != -1:
                    alg_name += f'-outnoise={args.io_out_noise}'
            
            # prefix for update bit length
            if args.update_bl != -1:
                alg_name += f'-bl={args.update_bl}'
        
            if alg_name == 'mp' and args.MP_thershold_scale is not None:
                alg_name += f'-th={args.MP_thershold_scale}'
            if alg_name == 'mp' and args.MP_rank is not None:
                alg_name += f'-r={args.MP_rank}'
            if alg_name == 'mp' and args.MP_rank_p is not None:
                alg_name += f'-rp={args.MP_rank_p}'
                
    # D-D1-D1-D1-A1-A
    path_name_suffix = blocks_type_list[0] + '-' + 'x'.join(
        [block_type + str(n) 
            for n, block_type in zip(num_blocks_list, blocks_type_list[1:-1])]
    ) + '-'  + blocks_type_list[-1]
    # CIFAR10-Resnet-shallow/D-D1-D1-D1-A1-A
    path_name = f'{dataset_name}/' + path_name_suffix
    if RPU_NAME != 'Softbounds':
        # CIFAR10-Resnet-shallow/D-D1-D1-D1-A1-A/ReRamSB
        path_name += f'/{RPU_NAME}'
    # path_name = f'{dataset_name}/{RPU_NAME}'
    # ======================== ============== ========================
    
    
    log_path = f'runs/{path_name}/{alg_name}'
    logger = Logger(log_path)
    
    param_groups = model.parameters()
        
    
    # define the optimizer
    if opt_type is opt_T.TORCH:
        optimizer = get_optimizer(param_groups, args.optimizer)
    else:
        optimizer_cls = config['optimizer_cls']
        optimizer = optimizer_cls(param_groups)
        optimizer.regroup_param_groups(model)
        
    model = model.to(DEVICE)
    # logger.info(f"Model List")
    # logger.info([name for name, _ in model.named_modules()])
    # logger.info(f"==========================")
    # logger.info([name for name, _ in model.named_parameters()])
    # raise SystemExit

    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(10)
    criterion = LabelSmoothingLoss(10, 0.1)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    # from torchsummary import summary
    # summary(net, input_size=(3, 32, 32))  
    # net.train(mode=True)
    
    
    # logger.info("*" * 40)
    # logger.info(f"Starting training with the arguments")
    
    test_loss, test_accuracy = test_evaluation(model, testloader, criterion)
    log_str = f"Epoch {0} - Training loss: --------   Test Accuracy: {test_accuracy:.4f}"
    logger.write(0, log_str, {
        "Loss/test": test_loss,
        "Accuracy/test": test_accuracy
    })
    def aggressive_plateau(history, threshold=0.0001):
        if len(history) < 2:
            return False
        delta = history[-2] - history[-1]
        plateau = -delta > threshold
        print(f"[Aggressive] Δ={delta:.6f}, plateau={plateau}")
        return plateau
    
    allstart = time.time()
    test_loss, test_accuracy = test_evaluation(model, testloader, criterion)
    log_str = f"Epoch {0} - Training loss: --------   Test Accuracy: {test_accuracy:.4f}"
    logger.write(0, log_str, {
        "Loss/test": test_loss,
        "Accuracy/test": test_accuracy
    })
    def smooth_plateau(history, threshold=0.0001, window=5, max_violations=2):
        if len(history) < window + 1:
            return False
        recent = history[-(window + 1):]
        violations = 0

        for i in range(window):  
            delta = recent[i+1] - recent[i]
            if delta > threshold:
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
    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        # start = time.time() 
        model.train()
        for i, data in enumerate(trainloader, 0):
            # input size: [batch_size, channel_num, height, width]
            # [128, 3, 32, 32]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
            optimizer.step()
            total_loss += loss.item()

        # Decay learning rate if needed.
        scheduler.step()
        
        train_loss = total_loss / len(trainloader)
        test_loss, test_accuracy = test_evaluation(model, testloader, criterion)
        loss_history.append(train_loss)
        trigger_tile_switch_by_plateau(model, loss_history)
        # for warm start
        log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        logger.write(epoch, log_str, {
            "Loss/train": train_loss,
            "Loss/test": test_loss,
            "Accuracy/test": test_accuracy,
            "State/lr": scheduler.get_last_lr()[0],
        })
        
        if args.save_checkpoint and args.save_every is not None \
            and epoch % args.save_every == 0:
            check_point_folder = f'checkpoints/{path_name}'
            checkpoint_path = f'{check_point_folder}/{alg_name}-e{epoch}-e{num_tile}.pth'
            if not os.path.isdir(check_point_folder):
                os.makedirs(check_point_folder)
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
            print(f'Save checkpoint: {checkpoint_path}')
          
    print('Finished Training')
    allend = time.time()
    # print("time: ", allend - allstart)
    if args.save_checkpoint:
        check_point_folder = f'checkpoints/{path_name}'
        checkpoint_path = f'{check_point_folder}/{alg_name}-e{num_tile}-e{"state10"}.pth'
        if not os.path.isdir(check_point_folder):
            os.makedirs(check_point_folder)
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        print(f'Save checkpoint: {checkpoint_path}')
    
if __name__ ==  '__main__':
    train()