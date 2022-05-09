import torch
import torch.nn as nn
import torchvision
from skimage.util import random_noise
import matplotlib.pyplot as plt

def generate_simulated_image(img, resolution_original: float = 0.25, resolution_simulated: float = 1.0):
        scaled = nn.functional.interpolate(img, size=None, scale_factor=0.5, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        scaled = nn.functional.interpolate(scaled, size=None, scale_factor=0.5, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        # m = nn.AvgPool2d(2, stride=2)
        # res = m(m(img))
        gray = torchvision.transforms.functional.rgb_to_grayscale(scaled, num_output_channels=1)
        gray_gauss = torch.tensor(random_noise(gray, mode='gaussian', mean=0, var=0.05, clip=True), dtype=torch.float32)
        return gray_gauss