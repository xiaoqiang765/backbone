import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA

__all__ = ['MyData']


# 数据集整理
class MyData(Dataset):
    """实现输入数据的整理，具体函数需根据数据集进行调整"""
    def __init__(self, image_dir, label_path=None, is_train=False, is_pred=False):
        super(MyData, self).__init__()
        self.image_dir = image_dir
        self.label_path = label_path
        self.is_train = is_train
        self.is_pred = is_pred
        if self.label_path:
            self.df = pd.read_csv(self.label_path)
        else:
            self.df = pd.DataFrame({'image': map(lambda x: x.split('.')[0], os.listdir(self.image_dir)),
                                    'label': np.zeros(len(os.listdir(self.image_dir)))})
        # data augment
        self.transform_train = A.Compose([
            # crop
            # A.RandomCrop(250, 250, p=0.5),
            A.Resize(height=224, width=224),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomBrightness(limit=0.1, p=0.5),
            ], p=1),
            # A.GaussNoise(),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(rotate_limit=1, p=0.5),
            # FancyPCA(alpha=0.1, p=0.5),
            # iso noise
            # A.ISONoise(p=0.5),
            # cut out
            A.Cutout(p=0.5),
            # blur
            A.OneOf([
                A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.5),
            # Pixels
            # A.OneOf([
            #     A.IAAEmboss(p=0.5),
            #     A.IAASharpen(p=0.5),
            # ], p=1),
            # Affine
            # A.OneOf([
            #     A.ElasticTransform(p=0.5),
            #     A.IAAPiecewiseAffine(p=0.5),
            # ], p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        # val data augment
        self.transform_valid = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)])
        if self.is_train:
            self.data_transforms = self.transform_train
        else:
            self.data_transforms = self.transform_valid

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = os.sep.join([self.image_dir, str(self.df.iloc[item, 0])+'.jpg'])
        image_idx = self.df.iloc[item, 0]
        image = np.array(Image.open(image_path).convert('RGB'))
        image_label = torch.tensor(self.df.iloc[item, 1])
        image = self.data_transforms(image=image)['image']
        if self.is_pred:
            return image, image_idx
        else:
            return image, image_label





