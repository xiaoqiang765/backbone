"""ResNet"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


# BasicBlock for ResNet18 and ResNet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
                                                      padding=1, bias=False), BatchNorm2d(out_channels), ReLU(inplace=True),
                                               Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=(3, 3),
                                                      padding=1, bias=False), BatchNorm2d(out_channels*BasicBlock.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*BasicBlock.expansion:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=(1, 1),
                                                 stride=(stride, stride), bias=False), BatchNorm2d(out_channels*BasicBlock.expansion))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        output = residual+shortcut
        return self.relu(output)


# BottleNeck for ResNet50+
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_function = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                                               BatchNorm2d(out_channels), ReLU(inplace=True),
                                               Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
                                                      padding=1, bias=False), BatchNorm2d(out_channels), ReLU(inplace=True),
                                               Conv2d(out_channels, out_channels*BottleNeck.expansion, kernel_size=(1, 1),
                                                      bias=False), BatchNorm2d(out_channels*BottleNeck.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*BottleNeck.expansion:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=(1, 1),
                                                 stride=(stride, stride), bias=False), BatchNorm2d(out_channels*BottleNeck.expansion))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        output = residual+shortcut
        return self.relu(output)


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=176):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
                                  BatchNorm2d(64), ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block1 = self._make_layer(block, num_block[0], 64, 1)
        self.block2 = self._make_layer(block, num_block[1], 128, 2)
        self.block3 = self._make_layer(block, num_block[2], 256, 2)
        self.block4 = self._make_layer(block, num_block[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block.expansion
        return nn.Sequential(*layers)


# ResNet18
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# ResNet34
def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


# ResNet50
def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


# ResNet101
def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


# ResNet152
def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = resnet152()
    net.cuda()
    summary(net, (3, 224, 224))


