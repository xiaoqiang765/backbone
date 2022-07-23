"""MobileNet"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['mobilenet']


# DepthWiseSeparable Convolution
class DWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(DWSConv, self).__init__()
        self.depth_wise = nn.Sequential(Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, **kwargs),
                                        BatchNorm2d(in_channels), ReLU(inplace=True))
        self.point_wise = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                                        BatchNorm2d(out_channels), ReLU(inplace=True))

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


# Conv
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size, **kwargs), BatchNorm2d(out_channels),
                                  ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# MobileNet
class MobileNet(nn.Module):
    def __init__(self, width_multiplier=1.0, num_classes=176):
        super(MobileNet, self).__init__()
        a = width_multiplier
        self.stem = Conv(3, int(32*a), (3, 3), stride=(2, 2), padding=1)
        self.conv1 = nn.Sequential(DWSConv(int(32*a), int(64*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(64*a), int(128*a), (3, 3), stride=(2, 2), padding=1, bias=False),
                                   DWSConv(int(128*a), int(128*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(128*a), int(256*a), (3, 3), stride=(2, 2), padding=1, bias=False),
                                   DWSConv(int(256*a), int(256*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(256*a), int(512*a), (3, 3), stride=(2, 2), padding=1, bias=False),
                                   DWSConv(int(512*a), int(512*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(512*a), int(512*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(512*a), int(512*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(512*a), int(512*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(512*a), int(512*a), (3, 3), padding=1, bias=False),
                                   DWSConv(int(512*a), int(1024*a), (3, 3), stride=(2, 2), padding=1, bias=False))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(1024*a), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenet():
    return MobileNet()


if __name__ == '__main__':
    net = mobilenet()
    net.cuda()
    summary(net, (3, 224, 224))


