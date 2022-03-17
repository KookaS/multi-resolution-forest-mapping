from typing import Type
from numpy.core.fromnumeric import reshape
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

class SegmentationHead(nn.Sequential):
    """Segmentation head for a segmentation model"""
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity() # to be removed
        activation = nn.Identity() # to be removed
        super().__init__(conv2d, upsampling, activation)




