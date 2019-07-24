from __future__ import print_function
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1
    )


class DsBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DsBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        out = self.conv(x)
        out = self.relu(out)
        out = self.mp(out)

        return out

class AlexNet(nn.Module):
    def __init__(self, in_channels, ftr_nb, fc_nb):
        super(AlexNet, self).__init__()
        self.fc_nb = fc_nb
        self.down_convs = []
        self.fcs = []
        for i in range(len(down_convs)):
            if i == 0:
                ins = in_channels
            else:
                ins = ftr_nb[i-1]
            outs = ftr_nb[i]
            self.down_convs.append(DsBlock(ins, outs))
        for i in range(len(fc_nb)):
            ins = fc_nb[i]
            outs = fc_nb[i+1]
            self.fcs.append(nn.Linear(ins, outs))
    def forward(self, x):
        for i, module in enumerate(self.down_convs):
            x = module(x)
        for i, module in enumerate(self.fcs):
            x = module(x)
        return x

if __doc__ == "__main__":
    model = AlexNet(1, [16, 32, 64, 64, 32, 32, 16], [12*16, 2500, 400]).cuda()
    x = torch.FloatTensor(np.random.random((2, 1, 400, 512))).cuda()
    y = model(x)
