"""ShuffleNet v1"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['shufflenetv1']


# BasicConv
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size, **kwargs),
                                  BatchNorm2d(out_channels), ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# PointWise_Conv
class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PointWiseConv, self).__init__()
        self.conv = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(1, 1), **kwargs),
                                  BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x)


# DepthWiseConv
class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DepthWiseConv, self).__init__()
        self.conv = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, **kwargs),
                                  BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x)


# ChannelShuffle
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, h, w = x.size()
        x = x.view(batch, self.groups, int(channels/self.groups), h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, -1, h, w)
        return x


# ShuffleUnit
class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, stage, groups):
        super(ShuffleUnit, self).__init__()
        self.bottleneck = nn.Sequential(PointWiseConv(in_channels, int(out_channels/4), groups=groups),
                                        ReLU(inplace=True))
        if stage == 2:
            self.bottleneck = nn.Sequential(PointWiseConv(in_channels, int(out_channels/4)), ReLU(inplace=True))
        self.channel_shuffle = ChannelShuffle(groups)
        self.depth_wise = nn.Sequential(DepthWiseConv(int(out_channels/4), int(out_channels/4), stride=(stride, stride),
                                                      groups=int(out_channels/4)), BatchNorm2d(int(out_channels/4)))
        self.expand = nn.Sequential(PointWiseConv(int(out_channels/4), out_channels, groups=groups),
                                    BatchNorm2d(out_channels))
        self.fusion = self._add
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=stride, padding=1))
            self.expand = nn.Sequential(PointWiseConv(int(out_channels/4), (out_channels-in_channels), groups=groups),
                                        BatchNorm2d(out_channels-in_channels))
            self.fusion = self._cat

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.bottleneck(x)
        x = self.channel_shuffle(x)
        x = self.depth_wise(x)
        x = self.expand(x)
        out = self.fusion(residual, x)
        return self.relu(out)

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)


# ShuffleNet v1
class ShuffleNetV1(nn.Module):
    def __init__(self, num_blocks, num_classes=176, groups=3):
        super(ShuffleNetV1, self).__init__()
        if groups == 1:
            out_channels = [24, 144, 288, 576]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]
        self.stem = nn.Sequential(BasicConv(3, out_channels[0], (3, 3), stride=2, padding=1),
                                  nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.in_channels = out_channels[0]
        self.stage2 = self._make_stage(ShuffleUnit, num_blocks[0], out_channels[1], stride=2, stage=2, groups=groups)
        self.stage3 = self._make_stage(ShuffleUnit, num_blocks[1], out_channels[2], stride=2, stage=3, groups=groups)
        self.stage4 = self._make_stage(ShuffleUnit, num_blocks[2], out_channels[3], stride=2, stage=4, groups=groups)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_stage(self, block, num_blocks, out_channels, stride, stage, groups):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride, stage=stage, groups=groups))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


# shufflenetv1
def shufflenetv1():
    return ShuffleNetV1([4, 8, 4])


if __name__ == '__main__':
    net = shufflenetv1()
    net.cuda()
    summary(net, (3, 224, 224))