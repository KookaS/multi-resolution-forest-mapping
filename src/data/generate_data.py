import torch
import torch.nn as nn
import torchvision

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def generate_simulated_image(image, rgb_scaling=(1,1,1), posterize_bits=3, contrast_factor=0.8, gamma=0.45, gamma_gain=0.9, sharpness_factor=0, blur_size=31, blur_sigma=2, noise_mean=0., noise_std=0.2):
        scaled = image
        batch, ch, x, y = list(scaled.size())
        scaled = rgb_scaling[0]*scaled[:,0,:,:] + rgb_scaling[1]*scaled[:,1,:,:] + rgb_scaling[2]*scaled[:,2,:,:]
        scaled /= sum(rgb_scaling)
        scaled = torch.reshape(scaled, (batch,1,x,y))
        scaled = nn.functional.interpolate(scaled, size=None, scale_factor=0.5, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        scaled = nn.functional.interpolate(scaled, size=None, scale_factor=0.5, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        # scaled = torchvision.transforms.functional.rgb_to_grayscale(scaled, num_output_channels=1)
        scaled = torchvision.transforms.functional.gaussian_blur(scaled, kernel_size=blur_size, sigma=blur_sigma)
        scaled = AddGaussianNoise(mean=noise_mean, std=noise_std)(scaled)
        return scaled