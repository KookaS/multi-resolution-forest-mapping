from typing import Type
from numpy.core.fromnumeric import reshape
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

def complete_flat_correction(corr_activations):
    corr_remainder = -torch.sum(corr_activations, dim=1, keepdim=True)
    return torch.cat((corr_activations, corr_remainder), dim=1)

def complete_hierarchical_correction(corr_activations):
    corr_remainder = -torch.sum(corr_activations[:, :-1], dim=1, keepdim=True)
    #corr_remainder2 = -corr_activations[:, -1:]
    #return torch.cat((corr_activations[:, :-1], corr_remainder, corr_activations[:, -1:], corr_remainder2), dim=1)
    return torch.cat((corr_activations[:, :-1], corr_remainder, corr_activations[:, -1:]), dim=1)

class SegmentationHead(nn.Sequential):
    """Segmentation head for a segmentation model"""
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity() # to be removed
        activation = nn.Identity() # to be removed
        super().__init__(conv2d, upsampling, activation)

class RuleSegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, aux_channels, thresholds, rules, act_encoding,
                    corr_channels=2, kernel_size=3, decision = 'f'):
        """
        Args:
            - thresholds: list or tuple of tensors
        """
        super().__init__()

        self.corr_channels = corr_channels
        self.out_channels = out_channels
        # check types and dimensions
        try:
            single_channel = sum(aux_channels) == 1
        except TypeError:
            single_channel = aux_channels == 1
            aux_channels = [aux_channels]
        if single_channel:
            if not isinstance(thresholds, list):
                thresholds = [thresholds]
        else: # check type and length of thresholds
            if not isinstance(thresholds, list):
                raise TypeError('"thresholds" should be a list containing the thresholds for each intermediate variable')
        self.aux_var = len(aux_channels) 
        # register buffers
        self.register_buffer('rules', rules)
        self.register_buffer('act_encoding', act_encoding)
        for i, t in enumerate(thresholds):
            self.register_buffer('thresholds_'+str(i), t)
        n_cat = torch.tensor([len(t) + 1 for t in thresholds])
        self.register_buffer('n_cat', n_cat)
        #self.register_buffer('aux_channels', aux_channels)

        self.conv_aux = nn.Conv2d(in_channels, sum(aux_channels), kernel_size=kernel_size, 
                            padding=kernel_size // 2)
        
        self.process_aux = self.process_regr_logits # lambda aux_actv: [f(x) for f, x in zip(process_func, aux_actv.split(aux_channels, dim=1))]
        self.get_categories = self.regr_to_cat # lambda aux_actv: [f(x) for f, x in zip(cat_func, aux_actv)]
        # self.conv_corr = nn.Conv2d(in_channels+out_channels, corr_channels, kernel_size=2*kernel_size+1, 
        #                     padding=kernel_size)
        self.conv_corr = nn.Sequential(nn.Conv2d(in_channels+out_channels, in_channels//2, kernel_size=2*kernel_size+1, 
                                                                                                padding=kernel_size),
                                       nn.Conv2d(in_channels//2, corr_channels, kernel_size=2*kernel_size+1, 
                                                                                                padding=kernel_size))
        self.residual_correction = self.apply_additive_correction #self.apply_linear_correction

        if decision == 'f':
            self.complete_correction = complete_flat_correction
        else:
            self.complete_correction = complete_hierarchical_correction
            
    def load_buffers(self, buffer_val_list, device):
        self.rules = buffer_val_list[0].to(device)
        self.act_encoding = buffer_val_list[1].to(device)
        # TODO
        # for i, t in enumerate(buffer_val_list[2:-1]):
        #     self.register_buffer('thresholds_'+str(i), t)   
        self.n_cat = buffer_val_list[-1].to(device)
        
    
    def regr_to_cat(self, x):
        # the aux channels are treated separately because they might have a different number of thresholds
        h, w = x.shape[-2:]
        n = x.shape[0]
        cat = torch.empty_like(x, dtype=torch.long)
        # thresholds = list(self.thresholds())
        for idx, threshold in enumerate(self.thresholds()):
            t = threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, n, h, w)
            comp = x[:, idx] >= t
            cat[:, idx] = torch.sum(comp.long(), axis=0)
        return cat

    def process_regr_logits(self, x):
        # x: (N, aux_var, H, W)
        # squeeze then ReLU to clip negative values to 0
        # return f.relu(x[:, 0], inplace=False) 
        return f.relu(x, inplace=False)
    
    def thresholds(self):
        for i in range(self.aux_var):
            yield self.__getattr__('thresholds_'+str(i))

    def intersect(self, cat):
        y = cat[:,-1]
        for i in range(self.aux_var - 2, -1, -1):
            y = y + cat[:, i] * self.n_cat[i+1] 
        y = self.rules[y]
        
        return y

    def apply_linear_correction(self, rule_activations, corr_activations):
        # corr_activations should have a depth of 2 * n_classes
        activations = rule_activations * corr_activations[:, :self.out_channels] + corr_activations[:, self.out_channels:] 
        return activations

    def apply_additive_correction(self, rule_activations, corr_activations):
        activations = rule_activations + corr_activations 
        return activations

    def forward(self, x):
        x_aux, x_corr = x
        # compute auxiliary activations
        aux_activations = self.conv_aux(x_aux)
        proc_aux_activations = self.process_aux(aux_activations)
        # apply rules
        rule_cat = self.intersect(self.get_categories(proc_aux_activations))
        rule_activations = self.act_encoding[rule_cat].transpose(2, 3).transpose(1, 2) #.movedim((0, 3, 1, 2), (0, 1, 2, 3)) #translate categories into hard-coded activations vectors
        # compute correction activations
        corr_activations = self.conv_corr(torch.cat((x_corr, rule_activations), dim = 1))
        corr_complete_activations = self.complete_correction(corr_activations)
        # apply correction
        final_activations = self.residual_correction(rule_activations, corr_complete_activations)
        return final_activations, rule_cat, corr_complete_activations, proc_aux_activations


