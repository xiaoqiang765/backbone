"""Inception_ResnetV2"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, AvgPool2d, MaxPool2d
from torchsummary import summary

__all__ = ['inception_resnet_v2']


# 基础卷积块conv-bn-relu
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, **kwargs)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Inception_stem(the right of the figure 2)
class InceptionStem(nn.Module):
    def __init__(self, in_channels):
        super(InceptionStem, self).__init__()
        self.conv1 = nn.Sequential(BasicConv(in_channels, 32, kernel_size=(3, 3), stride=(2, 2)),
                                   BasicConv(32, 32, kernel_size=(3, 3)),
                                   BasicConv(32, 64, kernel_size=(3, 3), padding=1))
        self.branch3x3_pool = MaxPool2d(kernel_size=(3, 3), stride=2)
        self.branch3x3_conv = BasicConv(64, 96, kernel_size=(3, 3), stride=(2, 2))
        self.branch7x7_1 = nn.Sequential(BasicConv(160, 64, kernel_size=(1, 1)), BasicConv(64, 96, kernel_size=(3, 3)))
        self.branch7x7_2 = nn.Sequential(BasicConv(160, 64, kernel_size=(1, 1)), BasicConv(64, 64, kernel_size=(7, 1), padding=(3, 0)),
                                         BasicConv(64, 64, kernel_size=(1, 7), padding=(0, 3)), BasicConv(64, 96, kernel_size=(3, 3)))
        self.branch_pool1 = BasicConv(192, 192, kernel_size=(3, 3), stride=(2, 2))
        self.branch_pool2 = MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = [self.branch3x3_pool(x), self.branch3x3_conv(x)]
        x = torch.cat(x, dim=1)
        x = [self.branch7x7_1(x), self.branch7x7_2(x)]
        x = torch.cat(x, dim=1)
        x = [self.branch_pool1(x), self.branch_pool2(x)]
        x = torch.cat(x, dim=1)
        return x


class InceptionRA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionRA, self).__init__()
        self.branch3x3_stack = nn.Sequential(BasicConv(in_channels, 32, kernel_size=(1, 1)),
                                             BasicConv(32, 48, kernel_size=(3, 3), padding=1),
                                             BasicConv(48, 64, kernel_size=(3, 3), padding=1))
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 32, kernel_size=(1, 1)),
                                       BasicConv(32, 32, kernel_size=(3, 3), padding=1))
        self.branch1x1 = BasicConv(in_channels, 32, kernel_size=(1, 1))
        self.reduction = Conv2d(128, 384, kernel_size=(1, 1))
        self.shortcut = Conv2d(in_channels, 384, kernel_size=(1, 1))
        self.bn = BatchNorm2d(384)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch3x3_stack(x), self.branch3x3(x), self.branch1x1(x)]
        residual = torch.cat(residual, dim=1)
        residual = self.reduction(residual)
        shortcut = self.shortcut(x)
        output = self.bn(residual+shortcut)
        output = self.relu(output)
        return output


class InceptionRB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionRB, self).__init__()
        self.branch7x7 = nn.Sequential(BasicConv(in_channels, 128, kernel_size=(1, 1)),
                                       BasicConv(128, 160, kernel_size=(1, 7), padding=(0, 3)),
                                       BasicConv(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.branch1x1 = BasicConv(in_channels, 192, kernel_size=(1, 1))
        self.reduction = Conv2d(384, 1154, kernel_size=(1, 1))
        self.shortcut = Conv2d(in_channels, 1154, kernel_size=(1, 1))
        self.bn = BatchNorm2d(1154)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch7x7(x), self.branch1x1(x)]
        residual = torch.cat(residual, dim=1)
        residual = self.reduction(residual)
        shortcut = self.shortcut(x)
        output = self.bn(shortcut+residual)
        output = self.relu(output)
        return output


class InceptionRC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionRC, self).__init__()
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 192, kernel_size=(1, 1)),
                                       BasicConv(192, 224, kernel_size=(1, 3), padding=(0, 1)),
                                       BasicConv(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.branch1x1 = BasicConv(in_channels, 192, kernel_size=(1, 1))
        self.reduction = Conv2d(448, 2048, kernel_size=(1, 1))
        self.shortcut = Conv2d(in_channels, 2048, kernel_size=(1, 1))
        self.bn = BatchNorm2d(2048)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch3x3(x), self.branch1x1(x)]
        residual = torch.cat(residual, dim=1)
        residual = self.reduction(residual)
        shortcut = self.shortcut(x)
        output = self.bn(shortcut+residual)
        output = self.relu(output)
        return output


class InceptionRRA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(InceptionRRA, self).__init__()
        self.branch3x3_stack = nn.Sequential(BasicConv(in_channels, k, kernel_size=(1, 1)),
                                             BasicConv(k, l, kernel_size=(3, 3), padding=1),
                                             BasicConv(l, m, kernel_size=(3, 3), stride=(2, 2)))
        self.branch3x3 = BasicConv(in_channels, n, kernel_size=(3, 3), stride=(2, 2))
        self.branch_pool = MaxPool2d(kernel_size=3, stride=2)
        self.out_channels = in_channels+n+m

    def forward(self, x):
        x = [self.branch3x3_stack(x), self.branch3x3(x), self.branch_pool(x)]
        output = torch.cat(x, dim=1)
        return output


class InceptionRRB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionRRB, self).__init__()
        self.branch3x3a = nn.Sequential(BasicConv(in_channels, 256, kernel_size=(1, 1)),
                                        BasicConv(256, 384, kernel_size=(3, 3), stride=2))
        self.branch3x3b = nn.Sequential(BasicConv(in_channels, 256, kernel_size=(1, 1)),
                                        BasicConv(256, 288, kernel_size=(3, 3), stride=(2, 2)))
        self.branch3x3stack = nn.Sequential(BasicConv(in_channels, 256, kernel_size=(1, 1)),
                                            BasicConv(256, 288, kernel_size=(3, 3), padding=1),
                                            BasicConv(288, 320, kernel_size=(3, 3), stride=(2, 2)))
        self.branch_pool = MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = [self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x), self.branch_pool(x)]
        output = torch.cat(x, dim=1)
        return output


class InceptionResNetV2(nn.Module):
    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, num_classes=176):
        super(InceptionResNetV2, self).__init__()
        self.stem = InceptionStem(3)
        self.inception_resnet_a = self._generate_inception_module(384, 384, A, InceptionRA)
        self.reduction_a = InceptionRRA(384, k, l, m, n)
        out_channels = self.reduction_a.out_channels
        self.inception_resnet_b = self._generate_inception_module(out_channels, 1154, B, InceptionRB)
        self.reduction_b = InceptionRRB(1154)
        self.inception_resnet_c = self._generate_inception_module(2146, 2048, C, InceptionRC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.2)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _generate_inception_module(in_channels, out_channels, num_block, block):
        layers = nn.Sequential()
        for l in range(num_block):
            layers.add_module(f'{block.__name__}_{l}', block(in_channels))
            in_channels = out_channels
        return layers


def inception_resnet_v2():
    return InceptionResNetV2(5, 10, 5)


if __name__ == '__main__':
    net = inception_resnet_v2()
    net.cuda()
    summary(net, (3, 299, 299))





