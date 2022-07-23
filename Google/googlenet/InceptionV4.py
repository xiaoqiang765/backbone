"""InceptionV4, Inception_resnet"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, AvgPool2d, MaxPool2d
from torchsummary import summary

__all__ = ['inceptionv4']


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


# pure inceptionV3 A
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_avg = nn.Sequential(AvgPool2d(kernel_size=3, stride=1, padding=1),
                                        BasicConv(in_channels, 96, kernel_size=(1, 1)))
        self.branch1x1 = BasicConv(in_channels, 96, kernel_size=(1, 1))
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 64, kernel_size=(1, 1)),
                                       BasicConv(64, 96, kernel_size=(3, 3), padding=1))
        self.branch5x5 = nn.Sequential(BasicConv(in_channels, 64, kernel_size=(1, 1)),
                                       BasicConv(64, 96, kernel_size=(3, 3), padding=1),
                                       BasicConv(96, 96, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        x = [self.branch_avg(x), self.branch1x1(x), self.branch3x3(x), self.branch5x5(x)]
        x = torch.cat(x, dim=1)
        return x


class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionA, self).__init__()
        self.branch3x3_stack = nn.Sequential(BasicConv(in_channels, k, kernel_size=(1, 1)),
                                             BasicConv(k, l, kernel_size=(3, 3), padding=1),
                                             BasicConv(l, m, kernel_size=(3, 3), stride=(2, 2)))
        self.branch3x3 = BasicConv(in_channels, n, kernel_size=(3, 3), stride=(2, 2))
        self.branch_pool = MaxPool2d(kernel_size=3, stride=2)
        self.out_channels = in_channels+m+n

    def forward(self, x):
        x = [self.branch3x3_stack(x), self.branch3x3(x), self.branch_pool(x)]
        output = torch.cat(x, dim=1)
        return output


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch_avg = nn.Sequential(AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConv(in_channels, 128, kernel_size=(1, 1)))
        self.branch1x1 = BasicConv(in_channels, 384, kernel_size=(1, 1))
        self.branch7x7 = nn.Sequential(BasicConv(in_channels, 192, kernel_size=(1, 1)),
                                       BasicConv(192, 224, kernel_size=(7, 1), padding=(3, 0)),
                                       BasicConv(224, 256, kernel_size=(1, 7), padding=(0, 3)))
        self.branch7x7_stack = nn.Sequential(BasicConv(in_channels, 192, kernel_size=(1, 1)),
                                             BasicConv(192, 192, kernel_size=(1, 7), padding=(0, 3)),
                                             BasicConv(192, 224, kernel_size=(7, 1), padding=(3, 0)),
                                             BasicConv(224, 224, kernel_size=(1, 7), padding=(0, 3)),
                                             BasicConv(224, 256, kernel_size=(7, 1), padding=(3, 0)))

    def forward(self, x):
        x = [self.branch_avg(x), self.branch1x1(x), self.branch7x7(x), self.branch7x7_stack(x)]
        output = torch.cat(x, dim=1)
        return output


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv(in_channels, 256, kernel_size=(1, 1)),
            BasicConv(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=1)
        )
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 192, kernel_size=(1, 1)),
                                       BasicConv(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=1))
        self.branch_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [self.branch7x7(x), self.branch3x3(x), self.branch_pool(x)]
        output = torch.cat(x, dim=1)
        return output


class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch_avg = nn.Sequential(AvgPool2d(kernel_size=3, stride=1, padding=1),
                                        BasicConv(in_channels, 256, kernel_size=(1, 1)))
        self.branch1x1 = BasicConv(in_channels, 256, kernel_size=(1, 1))
        self.branch3x3 = BasicConv(in_channels, 384, kernel_size=(1, 1))
        self.branch3x3_1 = BasicConv(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2 = BasicConv(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch5x5 = nn.Sequential(BasicConv(in_channels, 384, kernel_size=(1, 1)),
                                       BasicConv(384, 448, kernel_size=(1, 3), padding=(0, 1)),
                                       BasicConv(448, 512, kernel_size=(3, 1), padding=(1, 0)))
        self.branch5x5_1 = BasicConv(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch5x5_2 = BasicConv(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        out1 = self.branch_avg(x)
        out2 = self.branch1x1(x)
        out3 = self.branch3x3(x)
        out3 = [self.branch3x3_1(out3), self.branch3x3_2(out3)]
        out3 = torch.cat(out3, dim=1)
        out4 = self.branch5x5(x)
        out4 = [self.branch5x5_1(out4), self.branch5x5_2(out4)]
        out4 = torch.cat(out4, dim=1)
        output = torch.cat([out1, out2, out3, out4], dim=1)
        return output


class InceptionV4(nn.Module):
    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, num_classes=176):
        super(InceptionV4, self).__init__()
        self.stem = InceptionStem(3)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.out_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avg_pool = AvgPool2d(7)
        self.dropout = nn.Dropout2d(0.2)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _generate_inception_module(in_channels, out_channels, num_blocks, block):
        layers = nn.Sequential()
        for l in range(num_blocks):
            layers.add_module(f'{block.__name__}_{l}', block(in_channels))
            in_channels = out_channels
        return layers


def inceptionv4():
    return InceptionV4(4, 7, 3)


if __name__ == '__main__':
    net = InceptionV4(4, 7, 3)
    net.cuda()
    summary(net, (3, 299, 299))

