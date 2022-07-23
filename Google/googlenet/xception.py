""" Xception model"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, MaxPool2d
from torchsummary import summary

__all__ = ['xception']


# 定义深度分离卷积
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SeparableConv, self).__init__()
        self.depth_wise = Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False, **kwargs)
        self.point_wise = Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


# 定义entry flow
class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.stem = nn.Sequential(Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False), BatchNorm2d(32),
                                  ReLU(), Conv2d(32, 64, kernel_size=(1, 1)), BatchNorm2d(64), ReLU())
        self.conv1_residual = nn.Sequential(SeparableConv(64, 128, (3, 3), padding=1), BatchNorm2d(128), ReLU(),
                                            SeparableConv(128, 128, (3, 3), padding=1), BatchNorm2d(128),
                                            MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv1_shortcut = nn.Sequential(Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False), BatchNorm2d(128))
        self.conv2_residual = nn.Sequential(ReLU(), SeparableConv(128, 256, (3, 3), padding=1), BatchNorm2d(256), ReLU(),
                                            SeparableConv(256, 256, (3, 3), padding=1), BatchNorm2d(256),
                                            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))
        self.conv2_shortcut = nn.Sequential(Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2)), BatchNorm2d(256))
        self.conv3_residual = nn.Sequential(ReLU(), SeparableConv(256, 728, (3, 3), padding=1), BatchNorm2d(728),
                                            ReLU(), SeparableConv(728, 728, (3, 3), padding=1), BatchNorm2d(728),
                                            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))
        self.conv3_shortcut = nn.Sequential(Conv2d(256, 728, kernel_size=(1, 1), stride=(2, 2)), BatchNorm2d(728))

    def forward(self, x):
        x = self.stem(x)
        residual = self.conv1_residual(x)
        shortcut = self.conv1_shortcut(x)
        x = residual+shortcut
        residual = self.conv2_residual(x)
        shortcut = self.conv2_shortcut(x)
        x = residual+shortcut
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual+shortcut
        return x


# Middle Flow实现
class MiddleFlowBlock(nn.Module):
    def __init__(self):
        super(MiddleFlowBlock, self).__init__()
        self.block = nn.Sequential(ReLU(), SeparableConv(728, 728, (3, 3), padding=1), BatchNorm2d(728),
                                   ReLU(), SeparableConv(728, 728, (3, 3), padding=1), BatchNorm2d(728),
                                   ReLU(), SeparableConv(728, 728, (3, 3), padding=1), BatchNorm2d(728))

    def forward(self, x):
        return x + self.block(x)


class MiddleFlow(nn.Module):
    def __init__(self, block, times=8):
        super(MiddleFlow, self).__init__()
        self.block = self._make_layer(block, times)

    @staticmethod
    def _make_layer(block, times):
        layers = []
        for i in range(times):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Exit Flow
class ExitFlow(nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.conv1_residual = nn.Sequential(ReLU(), SeparableConv(728, 728, (3, 3), padding=1), BatchNorm2d(728), ReLU(),
                                            SeparableConv(728, 1024, (3, 3), padding=1), BatchNorm2d(1024),
                                            MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv1_shortcut = nn.Sequential(Conv2d(728, 1024, kernel_size=(1, 1), stride=(2, 2)), BatchNorm2d(1024))
        self.flow = nn.Sequential(SeparableConv(1024, 1536, (3, 3), padding=1), BatchNorm2d(1536), ReLU(),
                                  SeparableConv(1536, 2048, (3, 3), padding=1), BatchNorm2d(2048), ReLU(),
                                  AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        residual = self.conv1_residual(x)
        shortcut = self.conv1_shortcut(x)
        x = residual+shortcut
        x = self.flow(x)
        return x


# Xception
class Xception(nn.Module):
    def __init__(self, block, num_classes=176):
        super(Xception, self).__init__()
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow(block)
        self.exit_flow = ExitFlow()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def xception():
    return Xception(MiddleFlowBlock)


if __name__ == '__main__':
    net = xception()
    net.cuda()
    summary(net, (3, 299, 299))




