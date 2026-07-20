"""
ResNet-18 Backbone — 消融 baseline
===================================
"""
import torch.nn as nn
from torchvision.models import resnet18


class ResNetBackbone(nn.Module):
    def __init__(self, in_chans=1, pretrained=False):
        super().__init__()
        net = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        if in_chans != 3:
            net.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1; self.layer2 = net.layer2
        self.layer3 = net.layer3; self.layer4 = net.layer4

    def forward(self, x):
        x = self.maxpool(self.stem(x))
        c2 = self.layer1(x); c3 = self.layer2(c2)
        c4 = self.layer3(c3); c5 = self.layer4(c4)
        return [c2, c3, c4, c5], [], []
