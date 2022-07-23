import torch
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d
from torch import nn
from torchsummary import summary


__all__ = ['GoogLeNet']


class Inception(nn.Module):
    def __init__(self, in_c, out1_c, mid13_c, out13_c, mid15_c, out15_c, max_out_c):
        super(Inception, self).__init__()
        self.conv1 = nn.Sequential(Conv2d(in_c, out1_c, kernel_size=(1, 1), stride=(1, 1)), BatchNorm2d(out1_c),
                                   ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2d(in_c, mid13_c, kernel_size=(1, 1), stride=(1, 1)), BatchNorm2d(mid13_c),
                                   ReLU(inplace=True),
                                   Conv2d(mid13_c, out13_c, kernel_size=(3, 3), stride=(1, 1), padding=1), BatchNorm2d(out13_c),
                                   ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2d(in_c, mid15_c, kernel_size=(1, 1), stride=(1, 1)), BatchNorm2d(mid15_c),
                                   ReLU(inplace=True),
                                   Conv2d(mid15_c, out15_c, kernel_size=(5, 5), stride=(1, 1), padding=2), BatchNorm2d(out15_c),
                                   ReLU(inplace=True))
        self.conv4 = nn.Sequential(MaxPool2d(kernel_size=3, stride=1, padding=1),
                                   Conv2d(in_c, max_out_c, kernel_size=(1, 1)), BatchNorm2d(max_out_c), ReLU(inplace=True))

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)
        output = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        return output


class GoogLeNet(nn.Module):
    def __init__(self, num_class=176):
        super(GoogLeNet, self).__init__()
        self.stem_layer = nn.Sequential(Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3), BatchNorm2d(64),
                                        ReLU(inplace=True), MaxPool2d(3, 2, 1), Conv2d(64, 64, kernel_size=(1, 1)),
                                        BatchNorm2d(64), ReLU(inplace=True), Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                        BatchNorm2d(192), ReLU(inplace=True), MaxPool2d(3, 2, 1))
        self.inception_layer1 = nn.Sequential(Inception(192, 64, 96, 128, 16, 32, 32),
                                              Inception(256, 128, 128, 192, 32, 96, 64),
                                              MaxPool2d(3, 2, 1))
        self.inception_layer2 = nn.Sequential(Inception(480, 192, 96, 208, 16, 48, 64),
                                              Inception(512, 160, 112, 224, 24, 64, 64),
                                              Inception(512, 128, 128, 256, 24, 64, 64),
                                              Inception(512, 112, 144, 288, 32, 64, 64),
                                              Inception(528, 256, 160, 320, 32, 128, 128),
                                              MaxPool2d(3, 2, 1))
        self.inception_layer3 = nn.Sequential(Inception(832, 256, 160, 320, 32, 128, 128),
                                              Inception(832, 384, 192, 384, 48, 128, 128),
                                              nn.AdaptiveAvgPool2d((1, 1)))
        self.dropout = nn.Dropout2d(0.4)
        self.output_layer = nn.Linear(1024, num_class)

    def forward(self, x):
        output = self.stem_layer(x)
        output = self.inception_layer1(output)
        output = self.inception_layer2(output)
        output = self.inception_layer3(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.output_layer(output)
        return output


def googlenet():
    return GoogLeNet()


if __name__ == '__main__':
    net = googlenet()
    net.cuda()
    summary(net, (3, 224, 224))



