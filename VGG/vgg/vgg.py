import torch
from torch import nn
import math
from torchsummary import summary

__all__ = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(512, 1000))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(net_config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in net_config:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


config = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
          'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
          'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
          'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def vgg11():
    return VGG(make_layers(config['A']))


def vgg11_bn():
    return VGG(make_layers(config['A'], batch_norm=True))


def vgg13():
    return VGG(make_layers(config['B']))


def vgg13_bn():
    return VGG(make_layers(config['B'], batch_norm=True))


def vgg16():
    return VGG(make_layers(config['D']))


def vgg16_bn():
    return VGG(make_layers(config['D'], batch_norm=True))


def vgg19():
    return VGG(make_layers(config['E']))


def vgg19_bn():
    return VGG(make_layers(config['E'], batch_norm=True))


if __name__ == '__main__':
    net = vgg19()
    net.cuda()
    summary(net, (3, 224, 224))



