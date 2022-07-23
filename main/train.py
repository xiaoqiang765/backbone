# implement main function
import os
import random
import numpy as np
from tqdm import tqdm
import sys
from torch import nn

import torch

import utils
from inputs import *
from net_config import *
from torch.utils.data import DataLoader
from torchvision import models

from utils import *
from torchvision.models import SqueezeNet1_1_Weights

# 随机数设定
use_cuda = torch.cuda.is_available()
seed = 11213
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dir = '/home/xiaoqiang/mlearning/car_competition/data/train_dir'
val_dir = '/home/xiaoqiang/mlearning/car_competition/data/val_dir'
train_label = '/home/xiaoqiang/mlearning/car_competition/data/train.csv'
val_label = '/home/xiaoqiang/mlearning/car_competition/data/val.csv'
output_dir = './output'
BACKBONE = 'final_squeezenet0722'
TRICKS = 'brightness_blur_cutout'
CONFIG1 = 'drop_out0.5'


def main(save_figure_name=None):
    """train function"""
    # data augment and dataloader
    os.makedirs(output_dir, exist_ok=True)
    train_data = MyData(train_dir, train_label, is_train=True)
    val_data = MyData(val_dir, val_label, is_train=False)
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], drop_last=True, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=args['batch_size'])
    # load net
    model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
    model = model.cuda()
    # optimizer, criterion, scheduler
    optimizer = torch.optim.__dict__[args['optimizer']](model.parameters(), **args['optimizer_para'])
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler']](optimizer, **args['scheduler_para'])
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = utils.LabelSmoothCrossLoss(reduction='mean')
    # train and valid
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    best_val_acc = 0
    for epoch in tqdm(range(args['num_epochs'])):
        train_loss, train_acc, valid_loss, valid_acc = train(model, train_loader, criterion, optimizer,
                                                             valid_data=valid_loader)
        backbone_dir = os.path.join(output_dir, BACKBONE)
        os.makedirs(backbone_dir, exist_ok=True)
        if best_val_acc < valid_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(backbone_dir, f'best_{BACKBONE}_{TRICKS}_{CONFIG1}.pth'))
        if best_val_acc > 99.99:
            print(f'当前epoch为：{epoch+1},最佳精度为：{best_val_acc}')
            break
        history['train_loss'].append(train_loss), history['train_acc'].append(train_acc), \
        history['valid_loss'].append(valid_loss), history['valid_acc'].append(valid_acc)
        if args['scheduler']:
            scheduler.step()
    if save_figure_name:
        plot_figure(history, save_figure=save_figure_name)
    print(f'best_acc: {best_val_acc}')


if __name__ == '__main__':
    main(f'./train_log/{BACKBONE}_{TRICKS}_{CONFIG1}.png')
