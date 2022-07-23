"""mobilenet v2"""
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['mobilenetv2']


# linearBottleneck
class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6):
        super(LinearBottleNeck, self).__init__()
        self.residual = nn.Sequential(Conv2d(in_channels, in_channels*t, kernel_size=(1, 1), bias=False),
                                      BatchNorm2d(in_channels*t), ReLU(inplace=True),
                                      Conv2d(in_channels*t, in_channels*t, kernel_size=(3, 3), stride=(stride, stride),
                                             padding=1, groups=in_channels*t, bias=False),
                                      BatchNorm2d(in_channels*t), ReLU(inplace=True),
                                      Conv2d(in_channels*t, out_channels, kernel_size=(1, 1), bias=False),
                                      BatchNorm2d(out_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual = residual + x
        return residual


# MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=176):
        super(MobileNetV2, self).__init__()
        self.stem = nn.Sequential(Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
                                  BatchNorm2d(32), ReLU(inplace=True))
        self.stage1 = LinearBottleNeck(32, 16, 1, t=1)
        self.stage2 = self._make_stage(16, 24, 2, 6, 2)
        self.stage3 = self._make_stage(24, 32, 2, 6, 3)
        self.stage4 = self._make_stage(32, 64, 2, 6, 4)
        self.stage5 = self._make_stage(64, 96, 1, 6, 3)
        self.stage6 = self._make_stage(96, 160, 2, 6, 2)
        self.stage7 = LinearBottleNeck(160, 320, 1)
        self.stage8 = nn.Sequential(Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    BatchNorm2d(1280), ReLU(inplace=True))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Conv2d(1280, num_classes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.avg(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def _make_stage(in_channels, out_channels, stride, t, num_block):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t=t))
        in_channels = out_channels
        while num_block-1:
            layers.append(LinearBottleNeck(in_channels, out_channels, 1, t=t))
            num_block = num_block-1
        return nn.Sequential(*layers)


def mobilenetv2():
    return MobileNetV2()


if __name__ == '__main__':
    net = mobilenetv2()
    net.cuda()
    summary(net, (3, 224, 224))