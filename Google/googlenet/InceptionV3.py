"""InceptionV3 in pytorch"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['inceptionv3']


# 基本卷积块
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, **kwargs)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# same naive inception module
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv(in_channels, 64, kernel_size=(1, 1))
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 48, kernel_size=(1, 1)),
                                       BasicConv(48, 64, kernel_size=(5, 5), padding=2))
        self.branch5x5 = nn.Sequential(BasicConv(in_channels, 64, kernel_size=(1, 1)),
                                       BasicConv(64, 96, kernel_size=(3, 3), padding=1),
                                       BasicConv(96, 96, kernel_size=(3, 3), padding=1))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv(in_channels, pool_features, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out3x3 = self.branch3x3(x)
        out5x5 = self.branch5x5(x)
        out_pool = self.branch_pool(x)
        output = torch.cat([out1x1, out3x3, out5x5, out_pool], dim=1)
        return output


# downsample
# Factorization into smaller convolutions
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv(in_channels, 384, kernel_size=(3, 3), stride=(2, 2))
        self.branch5x5 = nn.Sequential(BasicConv(in_channels, 64, kernel_size=1),
                                       BasicConv(64, 96, kernel_size=(3, 3), padding=1),
                                       BasicConv(96, 96, kernel_size=(3, 3), stride=(2, 2)))
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out3x3 = self.branch3x3(x)
        out5x5 = self.branch5x5(x)
        out_pool = self.branch_pool(x)
        output = torch.cat([out3x3, out5x5, out_pool], dim=1)
        return output


# Factorization Convolution with large Filter size
class InceptionC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionC, self).__init__()
        self. branch1x1 = BasicConv(in_channels, 192, kernel_size=(1, 1))
        self.branch7x7 = nn.Sequential(BasicConv(in_channels, out_channels, kernel_size=(1, 1)),
                                       BasicConv(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
                                       BasicConv(out_channels, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.branch7x7_stack = nn.Sequential(BasicConv(in_channels, out_channels, kernel_size=(1, 1)),
                                             BasicConv(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
                                             BasicConv(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
                                             BasicConv(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
                                             BasicConv(out_channels, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv(in_channels, 192, kernel_size=(1, 1)))

    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out7x7 = self.branch7x7(x)
        out7x7_stack = self.branch7x7_stack(x)
        out_pool = self.branch_pool(x)
        output = torch.cat([out1x1, out7x7, out7x7_stack, out_pool], dim=1)
        return output


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 192, kernel_size=(1, 1)),
                                       BasicConv(192, 320, kernel_size=(3, 3), stride=(2, 2)))
        self.branch7x7 = nn.Sequential(BasicConv(in_channels, 192, kernel_size=(1, 1)),
                                       BasicConv(192, 192, kernel_size=(1, 7), padding=(0, 3)),
                                       BasicConv(192, 192, kernel_size=(7, 1), padding=(3, 0)),
                                       BasicConv(192, 192, kernel_size=(3, 3), stride=(2, 2)))
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out_branch3x3 = self.branch3x3(x)
        out_branch7x7 = self.branch7x7(x)
        out_pool = self.branch_pool(x)
        output = torch.cat([out_branch3x3, out_branch7x7, out_pool], dim=1)
        return output


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv(in_channels, 320, kernel_size=(1, 1))
        self.branch3x3_1 = BasicConv(in_channels, 384, kernel_size=(1, 1))
        self.branch3x3_a = BasicConv(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_b = BasicConv(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3stack_1 = BasicConv(in_channels, 448, kernel_size=(1, 1))
        self.branch3x3stack_2 = BasicConv(448, 384, kernel_size=(3, 3), padding=1)
        self.branch3x3stack_a = BasicConv(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_b = BasicConv(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv(in_channels, 192, kernel_size=1))

    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out3x3 = self.branch3x3_1(x)
        out3x3 = [self.branch3x3_a(out3x3), self.branch3x3_b(out3x3)]
        out3x3 = torch.cat(out3x3, dim=1)
        out3x3stack = self.branch3x3stack_1(x)
        out3x3stack = self.branch3x3stack_2(out3x3stack)
        out3x3stack = [self.branch3x3stack_a(out3x3stack), self.branch3x3stack_b(out3x3stack)]
        out3x3stack = torch.cat(out3x3stack, dim=1)
        out_pool = self.branch_pool(x)
        output = torch.cat([out1x1, out3x3, out3x3stack, out_pool], dim=1)
        return output


class InceptionV3(nn.Module):
    def __init__(self, num_classes=176):
        super(InceptionV3, self).__init__()
        self.layer = nn.Sequential(BasicConv(3, 32, kernel_size=(3, 3), stride=(2, 2)),
                                   BasicConv(32, 32, kernel_size=(3, 3)),
                                   BasicConv(32, 64, kernel_size=(3, 3)),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   BasicConv(64, 80, kernel_size=(1, 1)),
                                   BasicConv(80, 192, kernel_size=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.mix5b = InceptionA(192, pool_features=32)
        self.mix5c = InceptionA(256, pool_features=64)
        self.mix5d = InceptionA(288, pool_features=64)
        # down sample
        self.mix6a = InceptionB(288)
        self.mix6b = InceptionC(768, 128)
        self.mix6c = InceptionC(768, 160)
        self.mix6d = InceptionC(768, 160)
        self.mix6e = InceptionC(768, 192)
        # down sample
        self.mix7a = InceptionD(768)
        self.mix7b = InceptionE(1280)
        self.mix7c = InceptionE(2048)
        # feature size
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer(x)
        x = self.mix5b(x)
        x = self.mix5c(x)
        x = self.mix5d(x)
        x = self.mix6a(x)
        x = self.mix6b(x)
        x = self.mix6c(x)
        x = self.mix6d(x)
        x = self.mix6e(x)
        x = self.mix7a(x)
        x = self.mix7b(x)
        x = self.mix7c(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def inceptionv3():
    return InceptionV3()


if __name__ == '__main__':
    net = inceptionv3()
    net.cuda()
    summary(net, (3, 224, 224))



