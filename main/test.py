"""
Author: xiao qiang
Time: 2022/7/21 10:07 
Version: env==torch py==3.9
"""
import torch
import pandas as pd
import argparse
import os
from torchvision import models
from torch import nn
from inputs import *
from torch.utils.data import DataLoader


BACKBONE = 'squeezenet'


# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='test images in image_dir')
    parser.add_argument('image_dir', type=str, help='the path of test image_dir')
    parser.add_argument('net_weight', type=str, help='the path of net weight')
    parser.add_argument('--result', default='result', type=str, help='the path to save result')
    args = parser.parse_args()
    return args


# 相关设置
args = parse_args()
use_gpu = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 128
input_size = 224
# 载入模型
# 根据模型修改
model = models.squeezenet1_1(pretrained=False)
model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1)),
                                 nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
# 加载数据
test_data = MyData(args.image_dir, is_pred=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
print(f'测试数据集图片个数为：{len(test_data)}')
# 模型权重加载
print(f'加载权重.....')
model.load_state_dict(torch.load(args.net_weight))
print(f'{args.net_weight}权重加载成功, 开始推理......')
# gpu
if use_gpu:
    model = model.cuda()
model.eval()
pred_list = []
image_path = []
with torch.no_grad():
    for images, image_ids in test_loader:
        if use_gpu:
            images = images.cuda()
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        pred_list += preds.cpu().numpy().tolist()
        image_path += image_ids

# 结果
result = pd.DataFrame({'image': image_path, 'label': pred_list})
os.makedirs(args.result, exist_ok=True)
result_path = os.path.join(args.result, 'result.csv')
result.to_csv(result_path, index=False, header=False)



