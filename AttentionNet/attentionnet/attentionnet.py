"""attentionNet"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['attentionnet56', 'attentionnet92']


# ResidualUnit
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualUnit, self).__init__()
        inner_channels = int(out_channels/4)
        self.residual = nn.Sequential(BatchNorm2d(in_channels), ReLU(inplace=True),
                                      Conv2d(in_channels, inner_channels, kernel_size=(1, 1)),
                                      BatchNorm2d(inner_channels), ReLU(inplace=True),
                                      Conv2d(inner_channels, inner_channels, kernel_size=(3, 3), stride=(stride, stride),padding=1),
                                      BatchNorm2d(inner_channels), ReLU(inplace=True),
                                      Conv2d(inner_channels, out_channels, kernel_size=(1, 1)))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride)))

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return residual + shortcut


# AttentionModule1
class AttentionModule1(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super(AttentionModule1, self).__init__()
        assert in_channels == out_channels
        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_down1 = self._make_residual(in_channels, out_channels, r)
        self.soft_down2 = self._make_residual(in_channels, out_channels, r)
        self.soft_down3 = self._make_residual(in_channels, out_channels, r)
        self.soft_down4 = self._make_residual(in_channels, out_channels, r)
        self.soft_up1 = self._make_residual(in_channels, out_channels, r)
        self.soft_up2 = self._make_residual(in_channels, out_channels, r)
        self.soft_up3 = self._make_residual(in_channels, out_channels, r)
        self.soft_up4 = self._make_residual(in_channels, out_channels, r)
        self.shortcut_short = ResidualUnit(in_channels, out_channels, 1)
        self.shortcut_long = ResidualUnit(in_channels, out_channels, 1)
        self.sigmoid = nn.Sequential(nn.BatchNorm2d(out_channels), ReLU(inplace=True),
                                     Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
                                     BatchNorm2d(out_channels), ReLU(inplace=True),
                                     Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
                                     nn.Sigmoid())
        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))
        x_t = self.trunk(x)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_down1(x_s)
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut_long = self.shortcut_long(x_s)
        x_s = F.max_pool2d(x_s, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_down2(x_s)
        shape2 = (x_s.size(2), x_s.size(3))
        shortcut_short = self.shortcut_short(x_s)
        x_s = F.max_pool2d(x_s, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_down3(x_s)
        x_s = self.soft_down4(x_s)
        x_s = self.soft_up1(x_s)
        x_s = self.soft_up2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short
        x_s = self.soft_up3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long
        x_s = self.soft_up4(x_s)
        x_s = F.interpolate(x_s, size=input_size)
        x_s = self.sigmoid(x_s)
        x = (1+x_s)*x_t
        x = self.last(x)
        return x

    @staticmethod
    def _make_residual(in_channels, out_channels, num):
        layers = []
        for _ in range(num):
            layers.append(ResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


#
class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels
        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_down1 = self._make_residual(in_channels, out_channels, r)
        self.soft_down2 = self._make_residual(in_channels, out_channels, r)
        self.soft_down3 = self._make_residual(in_channels, out_channels, r)
        self.soft_up1 = self._make_residual(in_channels, out_channels, r)
        self.soft_up2 = self._make_residual(in_channels, out_channels, r)
        self.soft_up3 = self._make_residual(in_channels, out_channels, r)
        self.shortcut = ResidualUnit(in_channels, out_channels, 1)
        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))
        x_t = self.trunk(x)
        # first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_down1(x_s)
        # 14 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut = self.shortcut(x_s)
        # second downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_down2(x_s)
        # mid
        x_s = self.soft_down3(x_s)
        x_s = self.soft_up1(x_s)
        # first upsample out 14
        x_s = self.soft_up2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut
        # second upsample out 28
        x_s = self.soft_up3(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)
        return x

    @staticmethod
    def _make_residual(in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(ResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


# attentionmodule3
class AttentionModule3(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels
        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_down1 = self._make_residual(in_channels, out_channels, r)
        self.soft_down2 = self._make_residual(in_channels, out_channels, r)
        self.soft_up1 = self._make_residual(in_channels, out_channels, r)
        self.soft_up2 = self._make_residual(in_channels, out_channels, r)
        self.shortcut = ResidualUnit(in_channels, out_channels, 1)
        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))
        x_t = self.trunk(x)
        # first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_down1(x_s)
        # mid
        x_s = self.soft_down2(x_s)
        x_s = self.soft_up1(x_s)
        # first upsample out 14
        x_s = self.soft_up2(x_s)
        x_s = F.interpolate(x_s, size=input_size)
        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)
        return x

    @staticmethod
    def _make_residual(in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(ResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


# attentionnet
class AttentionNet(nn.Module):
    def __init__(self, num_block, num_classes=176):
        super().__init__()
        self.pre = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                 BatchNorm2d(64), ReLU(inplace=True))
        self.stage1 = self._make_stage(64, 256, num_block[0], AttentionModule1)
        self.stage2 = self._make_stage(256, 512, num_block[1], AttentionModule2)
        self.stage3 = self._make_stage(512, 1024, num_block[2], AttentionModule3)
        self.stage4 = nn.Sequential(ResidualUnit(1024, 2048, 2),
                                    ResidualUnit(2048, 2048, 1),
                                    ResidualUnit(2048, 2048, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_stage(self, in_channels, out_channels, num, block):
        layers = []
        layers.append(ResidualUnit(in_channels, out_channels, 2))
        for _ in range(num):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

def attentionnet56():
    return AttentionNet([1, 1, 1])


def attentionnet92():
    return AttentionNet([1, 2, 3])


if __name__ == '__main__':
    net = attentionnet92()
    net.cuda()
    summary(net, (3, 224, 224))