import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from PIL import Image

__all__ = ['MyData', 'transform_train', 'transform_valid', 'data_split']


# 数据集整理
class MyData(Dataset):
    """实现输入数据的整理，具体函数需根据数据集进行调整"""
    def __init__(self, image_dir, label_dir=None, transform=None):
        super(MyData, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        if label_dir:
            self.df = pd.read_csv(label_dir)
            self.label_dict = {}
            for i, label in enumerate(self.df['label'].unique()):
                self.label_dict[label] = i
            self.reverse_label_dict = {k: v for v, k in self.label_dict.items()}
            self.df['encoded'] = self.df['label'].map(self.label_dict)
        else:
            self.df = pd.DataFrame({'image': os.listdir(self.image_dir),
                                    'label': np.zeros(len(os.listdir(self.image_dir))),
                                    'encoded': np.zeros(len(os.listdir(self.image_dir)))})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.df.iloc[item, 0])
        image = np.array(Image.open(image_path).convert('RGB'))
        image_label = torch.tensor(self.df.iloc[item, 2])
        if self.transform:
            image = self.transform(image)
        return image, image_label


# 数据增强
transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_valid = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# data_split
def data_split(dataset, valid_ratio):
    data_size = len(dataset)
    valid_size = int(np.floor(data_size*valid_ratio))
    data_idx = list(range(data_size))
    np.random.shuffle(data_idx)
    train_idx = data_idx[valid_size:]
    valid_idx = data_idx[:valid_size]
    train_data = torch.utils.data.Subset(dataset, train_idx)
    valid_data = torch.utils.data.Subset(dataset, valid_idx)
    return train_data, valid_data


