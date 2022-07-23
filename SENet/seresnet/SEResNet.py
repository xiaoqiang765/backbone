"""SEResNet"""
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear
from torch import nn
from torchsummary import summary

__all__ = ['seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152']


# BasicResidualSEBlock
class BasicResidualSEBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super(BasicResidualSEBlock, self).__init__()
        self.residual = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
                                             padding=1, bias=False), BatchNorm2d(out_channels), ReLU(inplace=True),
                                      Conv2d(out_channels, out_channels*self.expansion, kernel_size=(3, 3),
                                             stride=(1, 1), padding=1),
                                      BatchNorm2d(out_channels*self.expansion), ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels*self.expansion, kernel_size=(1, 1),
                                               stride=(stride, stride)), BatchNorm2d(out_channels*self.expansion))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(Linear(out_channels*self.expansion, out_channels*self.expansion//r),
                                        ReLU(inplace=True),
                                        Linear(out_channels*self.expansion//r, out_channels*self.expansion),
                                        nn.Sigmoid())
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        output = shortcut + residual*excitation.expand_as(residual)
        return self.relu(output)


# BottleneckResidualSEBlock
class BottleneckResidualSEBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()
        self.residual = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                                      BatchNorm2d(out_channels), ReLU(inplace=True),
                                      Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
                                             padding=1), BatchNorm2d(out_channels), ReLU(inplace=True),
                                      Conv2d(out_channels, out_channels*self.expansion, kernel_size=(1, 1), bias=False),
                                      BatchNorm2d(out_channels*self.expansion), ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels*self.expansion, kernel_size=(1, 1),
                                                 stride=(stride, stride)), BatchNorm2d(out_channels*self.expansion))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(Linear(out_channels*self.expansion, out_channels*self.expansion//r),
                                        ReLU(inplace=True),
                                        Linear(out_channels*self.expansion//r, out_channels*self.expansion),
                                        nn.Sigmoid())
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        output = shortcut + residual*excitation.expand_as(residual)
        return self.relu(output)


# SEResidual
class SEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=176):
        super().__init__()
        self.in_channels = 64
        self.pre = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), padding=1),
                                 BatchNorm2d(64), ReLU(inplace=True))
        self.stage1 = self._make_stage(block, num_blocks[0], 64, 1)
        self.stage2 = self._make_stage(block, num_blocks[1], 128, 2)
        self.stage3 = self._make_stage(block, num_blocks[2], 256, 2)
        self.stage4 = self._make_stage(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Linear(self.in_channels, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_stage(self, block, num_block, out_channels, stride):
        layers = []
        strides = [stride]+[1]*(num_block-1)
        for i in strides:
            layers.append(block(self.in_channels, out_channels, i))
            self.in_channels = out_channels*block.expansion
        return nn.Sequential(*layers)


# seresnet18
def seresnet18():
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2])


# seresnet34
def seresnet34():
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3])


# seresnet50
def seresnet50():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3])


# seresnet101
def seresnet101():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3])


#seresnet152
def seresnet152():
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3])


if __name__ == '__main__':
    net = seresnet101()
    net.cuda()
    summary(net, (3, 224, 224))