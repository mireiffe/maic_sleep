'''
Model structures
'''
from torch import nn
from torch.nn import functional as F
from torchvision import models

import layers


class ForTest(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.tl = layers.TestLayer(in_ch, out_ch)
    
    def forward(self, x):
        x = self.tl(x)
        return x

class resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet50 = models.resnet50(pretrained=True)

        self.resnet50.conv1.in_channels=1

        _ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(_ftrs, 5)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        self.resnet50(x)
