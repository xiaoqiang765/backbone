# implement main function
import os
import random
import numpy as np
from tqdm import tqdm
import sys

import torch
from inputs import *
from net_config import *
from torch.utils.data import DataLoader

from utils import *
import copy
sys.path.append('/home/xiaoqiang/Net/model/resnet')
from ResNet import resnet152


# 随机数设定
use_cuda = torch.cuda.is_available()
seed = 11213
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(save_figure_name=None):
    """train function"""
    # data augment and dataloader
    dataset = MyData(image_dir='/home/xiaoqiang/Net/data/train_set',
                        label_dir='/home/xiaoqiang/Net/data/train.csv', transform=transform_train)
    train_data, valid_data = data_split(dataset, valid_ratio=0.2)
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'])
    # load net
    model = resnet152()
    model.cuda()
    # optimizer, criterion, scheduler
    optimizer = torch.optim.__dict__[args['optimizer']](model.parameters(), **args['optimizer_para'])
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler']](optimizer, **args['scheduler_para'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # train and valid
    history = {'train_loss': [], 'train_acc': [], 'valid_loss':[], 'valid_acc': []}
    for epoch in tqdm(range(args['num_epochs'])):
        train_loss, train_acc, valid_loss, valid_acc = train(model, train_loader, criterion, optimizer, valid_loader=valid_loader)
        history['train_loss'].append(train_loss), history['train_acc'].append(train_acc), history['valid_loss'].append(valid_loss), history['valid_acc'].append(valid_acc)
        if args['scheduler']:
            scheduler.step()
    if save_figure_name:
        plot_figure(history, save_figure=save_figure_name)


if __name__ == '__main__':
    main('./train_picture/resnet152_200')
