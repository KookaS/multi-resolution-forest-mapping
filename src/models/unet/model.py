from typing import List
from .decoder import UnetDecoder
from ..encoders import ResNetEncoder
from ..base import SegmentationModel
from ..base import SegmentationHead
import torch

class Unet(SegmentationModel):
    """
    Unet model with a ResNet-18-like encoder, and possibly an auxiliary input source
    """

    def __init__(
        self,
        encoder_depth: int = 4,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 1,
        upsample: bool = False,
        aux_in_channels: int = 0,
        aux_in_position: int = 1,
        encoder_inputs =[64, 65, 128, 256, 512],
        **kwargs
    ):
        """
        Args:
            - encoder_depth: number of blocks in the ResNet encoder, each block itself 
                containing 2 correction blocks (Resnet-18). encoder_depth does not include
                    the initial conv and maxpool layers
            - decoder_channels: number of output channels of each decoder layer
            - in_channels: number of channels of the main input
            - classes: number of classes (i.e. number of output channels)
            - upsample: whether to upsample or not the activations at the end of each 
                decoder block. The upsampling is done via a transposed convolution with 
                upsampling factor 2. If a single value is specified the same value will
                be used for all decoder blocks.
            - aux_in_channels: number of channels of the auxiliary input
            - aux_in_position: position of the auxiliary input in the model:
                0: concatenated to the main input before entering the model
                1: before the 1st block of the encoder
                2: before the 2nd block of the encoder
                3: etc.
        """
        
        super().__init__()
        layers, out_channels = self.set_channels(aux_in_channels, aux_in_position, encoder_depth)
        encoder, decoder, segmentation_head = self._get_model_blocks()
        self.encoder = encoder(in_channels_high_res = in_channels,
                        aux_in_channels = aux_in_channels,
                        out_channels = out_channels,
                        layers = layers,
                        aux_in_position = aux_in_position)

        self.encoder_sim = encoder(in_channels_high_res = in_channels,
                        aux_in_channels = None,
                        out_channels = out_channels,
                        layers = layers,
                        aux_in_position = 0)

        self.decoder = decoder(
            encoder_channels=self.encoder._out_channels,
            decoder_channels=decoder_channels,
            upsample = upsample,
            use_batchnorm=True
        )
        self.segmentation_head = segmentation_head(
            in_channels=decoder_channels[-1], 
            out_channels=classes,
            kernel_size=3,
            **kwargs
        )

        self.initialize()

    def _get_model_blocks(self):
        return ResNetEncoder, UnetDecoder, SegmentationHead

    def set_channels(self, aux_in_channels, aux_in_position, encoder_depth):
        if (aux_in_channels is None) != (aux_in_position is None):
            raise ValueError('aux_in_channels and aux_in_position should be both specified')
        # architecture based on Resnet-18
        out_channels = [64, 64, 128, 256, 512]
        out_channels = out_channels[:encoder_depth+1]
        if aux_in_position is not None:
            out_channels[aux_in_position] += aux_in_channels
        layers = [2] * (len(out_channels)-1)
        return layers, out_channels

    
