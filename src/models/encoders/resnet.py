from copy import deepcopy
import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from typing import Type, Union, List, Optional


class ResNetEncoder(nn.Module):
    """
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        in_channels_high_res: int = 3,
        in_channels_low_res: int = 1,
        out_channels: List[int] = [64, 64, 128, 256, 512], #first element is the output of the first convolution
        layers: List[int] = [2, 2, 2, 2],
        aux_in_channels: Optional[int] = None,
        aux_in_position: Optional[int] = None,
    ) -> None:
        
        super(ResNetEncoder, self).__init__()

        # check arguments
        if len(out_channels) != len(layers) + 1:
            raise ValueError('len(out_channels) should be len(layers) + 1')
        if len(out_channels) < 2:
            raise ValueError('out_channels should have at least 2 elements so that '
                                'the encoder has at least 1 residual block')

        self.in_channels_high_res = in_channels_high_res 
        self.in_channels_low_res = in_channels_low_res 
        self._out_channels = out_channels
        self.aux_in_position = aux_in_position
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = out_channels[0] # used by self._make_layer()
        self.dilation = 1

        try:
            if aux_in_channels is None:
                #low resolution images
                self.stages_low_res = self._make_stages_low_res(block=block, layers=layers)
                self.forward = self._forward_1_input
            else:
                # high resolution images
                self.stages_high_res = self._make_stages_high_res(block=block, layers=layers)
                self.forward = self._forward_2_inputs
        except Exception as e:
            raise e

        # parameter initialization done at SegmentationModel level

    @property
    def out_channels(self): #used by decoder to make u-net skip connections
        """Return channels dimensions for each tensor of forward output of encoder"""
        return [self.in_channels] + self._out_channels

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # print('self.inplanes',self.inplanes, 'planes',planes, 'stride',stride, 'downsample',downsample, 
        #                     'previous_dilation',previous_dilation, 'norm_layer',norm_layer)
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            dilation = previous_dilation, norm_layer = norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_1_input(self, x):
        """
        Forward method for a single input
        """
        features = []
        # torch.Size([8, 3, 512, 512])
        for i in range(len(self.stages_low_res)):
            # print('x   :::: ', x.shape)
            # print(self.stages_low_res[i])
            x = self.stages_low_res[i](x)
            features.append(x)
        # print('_forward_1_input',x.shape)
        return features


    def _forward_2_inputs(self, x_0, x_1):
        """
        Forward method for 2 inputs (main input + auxiliary input)
        Not the fastest (if statement for each stage) but flexible
        """
        features = []
        x = x_0
        # print(x.shape, x_1.shape)
        for i in range(len(self.stages_high_res)):
            if self.aux_in_position == i:
                # print('torch.cat: ', x.shape, x_1.shape)
                x = torch.cat((x, x_1), dim = 1)
            # print('x   :::: ', x.shape)
            # print(self.stages_high_res[i])
            x = self.stages_high_res[i](x)
            features.append(x)
        # print('_forward_2_input',x.shape)
        return features

    def _make_stages_high_res(self, block, layers):
        # create layers
        conv1 = nn.Conv2d(self.in_channels_high_res, self._out_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        bn1 = self._norm_layer(self._out_channels[0])
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # first block of residual blocks
        if self.aux_in_position == 1:
            self.inplanes += self.aux_in_position
        layer1 = self._make_layer(block, self._out_channels[1], layers[0])

        stages = [  nn.Sequential(conv1, bn1, relu),
                    nn.Sequential(maxpool, layer1) ]

        # next blocks of residual blocks
        if len(self._out_channels) > 2:
            for i in range(2, len(self._out_channels)):
                if self.aux_in_position == i:
                    self.inplanes += self.aux_in_position
                stages.append(self._make_layer(block, self._out_channels[i], layers[i-1], stride=2))

        return nn.ModuleList(stages)

    def _make_stages_low_res(self, block, layers):
        # create layers
        conv1 = nn.Conv2d(self.in_channels_low_res, self._out_channels[0], kernel_size=7, stride=1, padding=3,
                               bias=False) # lower stride
        bn1 = self._norm_layer(self._out_channels[0])
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)# lower stride

        # first block of residual blocks
        if self.aux_in_position == 1:
            self.inplanes += self.aux_in_position
        layer1 = self._make_layer(block, self._out_channels[1], layers[0])
        # print('_make_layer ', layer1, self._out_channels[1], layers[0])

        stages = [  nn.Sequential(conv1, bn1, relu),
                    nn.Sequential(maxpool, layer1) ]

        # next blocks of residual blocks
        if len(self._out_channels) > 2:
            for i in range(2, len(self._out_channels)):
                if self.aux_in_position == i:
                    self.inplanes += self.aux_in_position
                # print('_make_layer ', self._make_layer(block, self._out_channels[i], layers[i-1], stride=2), self._out_channels[i], layers[i-1])
                stages.append(self._make_layer(block, self._out_channels[i], layers[i-1], stride=2))

        return nn.ModuleList(stages)



