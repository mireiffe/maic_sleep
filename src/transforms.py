import torch
from torch import nn
import torchvision.transforms.functional as TF


class RandomGamma(nn.Module):
    """
    gamma correction
    """
    def __init__(self, p=0.75):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            gamma = float(torch.rand(1) * 1.5 + .5)
            return TF.adjust_gamma(img, gamma=gamma, gain=1)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomContrast(nn.Module):
    """
    random contrast adjustment
    """
    def __init__(self, p=0.75):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            scale = float(torch.rand(1) * .5  + .7)
            return TF.adjust_contrast(img, contrast_factor=scale)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)