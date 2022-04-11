from xmlrpc.client import boolean
import torch
from . import initialization as init
import numpy as np

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, *x, sim: boolean):
        """
        Pass x through model's encoder, decoder and heads
        """
        features = None
        if sim:
            features = self.encoder_sim(*x)
        else:
            features = self.encoder(*x)
        decoder_output = self.decoder(*features)
        output = self.segmentation_head(decoder_output)

        return output

    def encode(self, *x, sim: boolean):
        features = None
        if sim:
            features = self.encoder_sim(*x)
        else:
            features = self.encoder(*x)
        return features
    
    def decode(self, *features):
        decoder_output = self.decoder(*features)
        output = self.segmentation_head(decoder_output)

        return output
