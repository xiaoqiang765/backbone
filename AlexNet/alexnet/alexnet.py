# 实现AlexNet
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary

__all__ = ['AlexNet']


# 实现LRN层
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNEL=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNEL
        if ACROSS_CHANNEL:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1,
                                        padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1, padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
            x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4), padding=0),
                                  ReLU(inplace=True), LRN(local_size=5, alpha=0.0001, beta=0.75),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  Conv2d(96, 256, kernel_size=(5, 5), padding=2, groups=2),
                                  ReLU(inplace=True), LRN(local_size=5, alpha=0.0001, beta=0.75),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  Conv2d(256, 384, kernel_size=(3, 3), padding=1), ReLU(inplace=True),
                                  Conv2d(384, 384, kernel_size=(3, 3), padding=1, groups=2),
                                  ReLU(inplace=True), Conv2d(384, 256, kernel_size=(3, 3), padding=1, groups=2),
                                  ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),)
        self.classifier = nn.Sequential(nn.Linear(256*6*6, 4096), ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(4096, 4096), ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = AlexNet()
    net.cuda()
    summary(net, (3, 227, 227))