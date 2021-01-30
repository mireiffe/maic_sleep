'''
Model structures
'''
from torch import nn
from torch.nn import functional as F

from . import layers


class ForTest(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.tl = layers.TestLayer(in_ch, out_ch)
    
    def forward(self, x):
        x = self.tl(x)
        return x
