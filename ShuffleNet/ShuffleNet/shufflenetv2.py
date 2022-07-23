"""ShuffleNetV2"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['shufflenetv2']


# channel split
def channel_split(x, split):
    assert x.size(1) == split*2
    return torch.split(x, split, dim=1)


# channel_shuffle
def channel_shuffle(x, groups):
    batch_size, channel, height, width = x.size()
    x = x.view(batch_size, groups, int(channel/groups), height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


# ShuffleUnit
class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        if stride != 1 or self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(stride, stride),
                                                 padding=1, groups=in_channels), BatchNorm2d(in_channels),
                                          Conv2d(in_channels, int(out_channels/2), kernel_size=(1, 1)),
                                          BatchNorm2d(int(out_channels/2)), ReLU(inplace=True))
            self.residual = nn.Sequential(Conv2d(in_channels, in_channels, kernel_size=(1, 1)), BatchNorm2d(in_channels),
                                          ReLU(inplace=True),
                                          Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(stride, stride),
                                                 padding=1, groups=in_channels), BatchNorm2d(in_channels),
                                          Conv2d(in_channels, int(out_channels/2), kernel_size=(1, 1)),
                                          BatchNorm2d(int(out_channels/2)), ReLU(inplace=True))
        else:
            self.shortcut = nn.Sequential()
            in_channels = int(in_channels/2)
            self.residual = nn.Sequential(Conv2d(in_channels, in_channels, kernel_size=(1, 1)),
                                          BatchNorm2d(in_channels), ReLU(inplace=True),
                                          Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, groups=in_channels),
                                          BatchNorm2d(in_channels),
                                          Conv2d(in_channels, in_channels, kernel_size=(1, 1)), BatchNorm2d(in_channels),
                                          ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            shortcut, residual = channel_split(x, int(self.in_channels/2))
        else:
            shortcut = x
            residual = x
        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        output = torch.cat([shortcut, residual], dim=1)
        output = channel_shuffle(output, 2)
        return output


# ShuffleNetV2
class ShuffleNetV2(nn.Module):
    def __init__(self, block, num_block, num_classes, ratio=1.0):
        super(ShuffleNetV2, self).__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1.0:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2.0:
            out_channels = [244, 488, 976, 1024]
        else:
            print('error ratio')
        self.stem = nn.Sequential(Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=1), BatchNorm2d(24),
                                  ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.in_channels = 24
        self.stage2 = self._make_stage(block, num_block[0], out_channels[0], 2)
        self.stage3 = self._make_stage(block, num_block[1], out_channels[1], 2)
        self.stage4 = self._make_stage(block, num_block[2], out_channels[2], 2)
        self.conv5 = nn.Sequential(Conv2d(out_channels[2], out_channels[3], kernel_size=(1, 1)),
                                   BatchNorm2d(out_channels[3]), ReLU(inplace=True))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_stage(self, block, num_block, out_channels, stride):
        layers = []
        strides = [stride]+[1]*(num_block-1)
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


# shufflenetv2
def shufflenetv2():
    return ShuffleNetV2(ShuffleUnit, [4, 8, 4], 176)


if __name__ == '__main__':
    net = shufflenetv2()
    net.cuda()
    summary(net, (3, 224, 224))