"""ResNext"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary
import torch.nn.functional as F

__all__ = ['resnext50', 'resnext101', 'resnext152']

CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64


# ResNextBottleneck
class ResNextBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        c = CARDINALITY  # 分组数量
        d = int(DEPTH*out_channels/BASEWIDTH) # 中间层每层通道数
        self.residual = nn.Sequential(Conv2d(in_channels, c*d, kernel_size=(1, 1), groups=c, bias=False),
                                      BatchNorm2d(c*d), ReLU(inplace=True),
                                      Conv2d(c*d, c*d, kernel_size=(3, 3), stride=(stride, stride), padding=1, groups=c,
                                             bias=False), BatchNorm2d(c*d), ReLU(inplace=True),
                                      Conv2d(c*d, 4*out_channels, kernel_size=(1, 1), bias=False),
                                      BatchNorm2d(4*out_channels))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels*4, kernel_size=(1, 1), stride=(stride, stride),
                                                 bias=False), BatchNorm2d(out_channels*4))

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return F.relu(residual+shortcut)


# ResNext
class ResNext(nn.Module):
    def __init__(self, block, num_blocks, num_classes=176):
        super().__init__()
        self.in_channels = 64
        self.pre = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), padding=1, bias=False),
                                 BatchNorm2d(64), ReLU(inplace=True))
        self.stage1 = self._make_stage(block, num_blocks[0], 64, 1)
        self.stage2 = self._make_stage(block, num_blocks[1], 128, 2)
        self.stage3 = self._make_stage(block, num_blocks[2], 256, 2)
        self.stage4 = self._make_stage(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

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
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*4
        return nn.Sequential(*layers)


# resnext50
def resnext50():
    return ResNext(ResNextBottleNeck, [3, 4, 6, 3])


# resnext101
def resnext101():
    return ResNext(ResNextBottleNeck, [3, 4, 23, 3])


# resnext152
def resnext152():
    return ResNext(ResNextBottleNeck, [3, 4, 36, 3])


if __name__ == '__main__':
    net = resnext50()
    net.cuda()
    summary(net, (3, 224, 224))