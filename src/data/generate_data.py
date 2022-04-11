import torch.nn as nn
import torchvision

def generate_simulated_image(img, resolution_original: float = 0.25, resolution_simulated: float = 1.0):

        # torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        # TODO noise
        m = nn.AvgPool2d(2, stride=2)
        res = m(m(img))
        gray = torchvision.transforms.functional.rgb_to_grayscale(res, num_output_channels=1)
        return gray