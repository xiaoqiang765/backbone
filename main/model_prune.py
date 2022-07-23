from torchvision import models
from torch import nn
from torchvision.models import SqueezeNet1_1_Weights
import torch

WEIGHTS_PATH = ''

model = models.squeezenet1_1(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.6, inplace=True),
    nn.Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
net_weight = model.load_state_dict(torch.load(WEIGHTS_PATH))