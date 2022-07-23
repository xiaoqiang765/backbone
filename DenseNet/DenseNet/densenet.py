"""DenseNet"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet264']


# BottleNeck
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channels = 4 * growth_rate
        self.bottleneck = nn.Sequential(BatchNorm2d(in_channels), ReLU(inplace=True),
                                        Conv2d(in_channels, inner_channels, kernel_size=(1, 1), bias=False),
                                        BatchNorm2d(inner_channels), ReLU(inplace=True),
                                        Conv2d(inner_channels, growth_rate, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        return torch.cat([self.bottleneck(x), x], dim=1)


# transition
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(BatchNorm2d(in_channels),
                                        Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                                        BatchNorm2d(out_channels), nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition(x)


# DenseNet
class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate=32, reduction=0.5, num_classes=176):
        super(DenseNet, self).__init__()
        inner_channels = 2 * growth_rate
        self.growth_rate = growth_rate
        self.pre = nn.Sequential(Conv2d(3, inner_channels, kernel_size=(7, 7), stride=(2, 2), padding=3),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.dense_layers = nn.Sequential()
        for index in range(len(num_blocks)-1):
            self.dense_layers.add_module(f'dense_layers{index}', self._make_dense_block(block, num_blocks[index],
                                                                                    inner_channels))
            inner_channels += growth_rate*num_blocks[index]
            out_channels = int(inner_channels*reduction)
            self.dense_layers.add_module(f'dense_layer_transition{index}', Transition(inner_channels, out_channels))
            inner_channels = out_channels
        self.dense_layers.add_module(f'dense_layers{len(num_blocks)-1}',
                                     self._make_dense_block(block, num_blocks[len(num_blocks)-1], inner_channels))
        inner_channels += growth_rate * num_blocks[len(num_blocks)-1]
        self.dense_layers.add_module('bn', BatchNorm2d(inner_channels))
        self.dense_layers.add_module('relu', ReLU(inplace=True))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        output = self.pre(x)
        output = self.dense_layers(output)
        output = self.avg_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_block(self, block, num_block, in_channels):
        dense_block = nn.Sequential()
        for i in range(num_block):
            dense_block.add_module(f'dense_block_layer{i}', block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


# densenet
def densenet121():
    return DenseNet(BottleNeck, [6, 12, 24, 16], growth_rate=32)


def densenet169():
    return DenseNet(BottleNeck, [6, 12, 32, 32])


def densenet201():
    return DenseNet(BottleNeck, [6, 12, 48, 32])


def densenet264():
    return DenseNet(BottleNeck, [6, 12, 64, 48])


if __name__ == '__main__':
    net = densenet264()
    net.cuda()
    summary(net, (3, 224, 224))
