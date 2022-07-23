"""SKNet"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['sknet26', 'sknet50', 'sknet101']


# SKConv
class SKConv(nn.Module):
    def __init__(self, in_channels, num_branch=2, num_group=32, r=16, stride=1, l=32):
        """
        SKConv
        :param in_channels:输入通道
        :param num_branch: 分支数量
        :param num_group: 分组卷积分组数
        :param r: 注意力机制中压缩系数
        :param stride: 步长
        :param l: 最低通道数
        """
        super().__init__()
        d = max(int(in_channels/r), l)
        self.num_branch = num_branch
        self.in_channels = in_channels
        self.convs = nn.ModuleList([])
        for i in range(num_branch):
            self.convs.append(nn.Sequential(Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(stride, stride),
                                                   padding=1+i, dilation=(1+i, 1+i), groups=num_group, bias=False),
                                            BatchNorm2d(in_channels), ReLU(inplace=True)))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(Conv2d(in_channels, d, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                BatchNorm2d(d), ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(num_branch):
            self.fcs.append(Conv2d(d, in_channels, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        features = [conv(x) for conv in self.convs]
        features = torch.cat(features, dim=1)
        features = features.view(batch_size, self.num_branch, self.in_channels, features.shape[2], features.shape[3])
        features_u = torch.sum(features, dim=1)
        features_s = self.gap(features_u)
        features_z = self.fc(features_s)
        attentions = [fc(features_z) for fc in self.fcs]
        attentions = torch.cat(attentions, dim=1)
        attentions = attentions.view(batch_size, self.num_branch, self.in_channels, 1, 1)
        attentions = self.softmax(attentions)
        features_v = torch.sum(attentions*features, dim=1)
        return features_v


# SKUnit
class SKUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, m=2, g=32, r=16, stride=1, l=32):
        super().__init__()
        self.conv1 = nn.Sequential(Conv2d(in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   BatchNorm2d(mid_channels), ReLU(inplace=True))
        self.conv_sk = SKConv(mid_channels, num_branch=m, num_group=g, r=r, stride=stride, l=l)
        self.conv3 = nn.Sequential(Conv2d(mid_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride),
                                                 bias=False), BatchNorm2d(out_channels))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.conv1(x)
        residual = self.conv_sk(residual)
        residual = self.conv3(residual)
        output = self.relu(residual+shortcut)
        return output


# SKNet
class SKNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=176):
        super().__init__()
        self.pre = nn.Sequential(Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
                                 BatchNorm2d(64), ReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(block, num_blocks[0], 64, 128, 256, 1)
        self.stage2 = self._make_stage(block, num_blocks[1], 256, 256, 512, 2)
        self.stage3 = self._make_stage(block, num_blocks[2], 512, 512, 1024, 2)
        self.stage4 = self._make_stage(block, num_blocks[3], 1024, 1024, 2048, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.max_pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_stage(block, num_block, in_c, mid_c, out_c, stride=1):
        layers = [block(in_c, mid_c, out_c, stride=stride)]
        for i in range(num_block-1):
            layers.append(block(out_c, mid_c, out_c, stride=1))
        return nn.Sequential(*layers)


# SKNet26
def sknet26():
    return SKNet(SKUnit, [2, 2, 2, 2])


# sknet50
def sknet50():
    return SKNet(SKUnit, [3, 4, 6, 3])


def sknet101():
    return SKNet(SKUnit, [3, 4, 23, 3])


if __name__ == '__main__':
    net = sknet101()
    net.cuda()
    summary(net, (3, 224, 224))