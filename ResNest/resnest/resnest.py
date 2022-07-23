"""ResNest"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary
import torch.nn.functional as F

__all__ = ['resnest50', 'resnest101', 'resnest152']


# Split Attention
class rSoftMax(nn.Module):
    def __init__(self, cardinal=1, radix=2):
        super().__init__()
        self.groups = cardinal
        self.radix = radix

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.groups, self.radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(batch_size, -1, 1, 1)
        return x


# SplitAttention
class SplitAttentionBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1, radix=2, cardinal=1, reduction_factor=4, **kwargs):
        super(SplitAttentionBlock, self).__init__()
        self.radix = radix
        self.cardinal = cardinal
        self.radix_conv = nn.Sequential(Conv2d(in_channels, channels*radix, (kernel_size, kernel_size), (stride, stride),
                                               groups=radix*cardinal, **kwargs), BatchNorm2d(channels*radix),
                                        ReLU(inplace=True))
        inter_channels = max(32, int(in_channels*radix/reduction_factor))
        self.fc1 = nn.Sequential(Conv2d(channels, inter_channels, kernel_size=(1, 1), stride=(1, 1), groups=cardinal,
                                        bias=False), BatchNorm2d(inter_channels), ReLU(inplace=True))
        self.fc2 = nn.Sequential(Conv2d(inter_channels, channels*radix, kernel_size=(1, 1), stride=(1, 1), groups=cardinal,
                                        bias=False), BatchNorm2d(channels*radix), ReLU(inplace=True))
        self.rsoftmax = rSoftMax(cardinal, radix)

    def forward(self, x):
        x = self.radix_conv(x)
        batch_size, r_channels = x.shape[:2]
        splits = torch.split(x, int(r_channels/self.radix), dim=1)
        gap = sum(splits)
        gap = F.adaptive_avg_pool2d(gap, 1)
        att = self.fc1(gap)
        att = self.fc2(att)
        att = self.rsoftmax(att)
        atts = torch.split(att, int(r_channels/self.radix), dim=1)
        output = sum([split*attention for split, attention in zip(splits, atts)])
        return output.contiguous()


# BottleNeck
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, radix=2, cardinal=1, bottleneck_width=64):
        super(BottleNeck, self).__init__()
        group_width = int(planes*(bottleneck_width/64))*cardinal
        self.radix = radix
        self.conv1 = nn.Sequential(Conv2d(in_planes, group_width, kernel_size=(1, 1), stride=(1, 1)),
                                   BatchNorm2d(group_width), ReLU(inplace=True))
        self.conv2 = SplitAttentionBlock(group_width, group_width, kernel_size=3, stride=stride, padding=1,
                                         cardinal=cardinal, radix=radix)
        self.conv3 = nn.Sequential(Conv2d(group_width, planes*4, kernel_size=(1, 1), stride=(1, 1)),
                                   BatchNorm2d(planes*4))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*4:
            self.shortcut = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                                          Conv2d(in_planes, planes*4, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                          BatchNorm2d(planes*4))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.shortcut:
            residual = self.shortcut(residual)
        output = residual+out
        return self.relu(output)


# ResNest
class ResNest(nn.Module):
    def __init__(self, block, num_blocks, radix=2, cardinal=1, bottleneck_width=64, num_classes=176):
        super(ResNest, self).__init__()
        self.radix = radix
        self.cardinal = cardinal
        self.bottleneck_width = bottleneck_width
        self.deep_stem = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(128), ReLU(inplace=True))
        self.in_channels = 128
        self.layer1 = self._make_stage(block, num_blocks[0], 1, 64)
        self.layer2 = self._make_stage(block, num_blocks[1], 2, 128)
        self.layer3 = self._make_stage(block, num_blocks[2], 2, 256)
        self.layer4 = self._make_stage(block, num_blocks[3], 2, 512)
        self.relu = ReLU(inplace=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        x = self.deep_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

    def _make_stage(self, block, num_block, stride, channels):
        strides = [stride]+[1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)


# resnest50
def resnest50():
    return ResNest(BottleNeck, [3, 4, 6, 3])


def resnest101():
    return ResNest(BottleNeck, [3, 4, 23, 3])


def resnest152():
    return ResNest(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = resnest152()
    net.cuda()
    summary(net, (3, 224, 224))



