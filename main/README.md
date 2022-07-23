## 汽车颜色分类
赛题：
颜色分类，禁止数据扩充，要求轻量化模型
### 一.消融思路
#### 数据相关
         1.数据分析：
           （1）数据分割：7类占比偏低，采用权重采样分割
           （2）k_fold: 时间较少，要求轻量化模型，无法k折平均，取消k-fold交叉
         2.防止过拟合：
           （1）数据增强：无法提交，增强采用常规增强
           （2）标签平滑：label smooting
#### 模型相关：
            1.选取轻量模型shufflenet，squeezenet，mnas等，测试baseline
            2.消融方案：时间有限，基本训练策略确定baseline，选取重点消融
            3.模型全局剪枝
            4.更新torchvision，获取最新权重
#### 脚本解释：
            1.测试：python test.py ${test_dir}, ${model_pth}, --out-dir
            2.训练：python train.py ## 相关文件py文件内修改
            3.train_log:训练日志，loss图等
            4.output：模型参数保存
#### 消融实验：
            1.config: SGD lr:0.01, omentum: 0.9, CosineAnnealingLR，batch=128，poch=100
            2.test:
                (1)squeezenet
                   a.baseline:val_best_acc:97.89%
                   b.aug_all:val_best_acc:97.22%
                   c.aug_brightness: 98.00%
                   d.aug_brightness_gaussian:97.88%
                   e.aug_brightness_hvflip_rotation:97.33%
                   f.aug_brightness_hvflip:97.55%
                   e.aug_brightness_blur: 98.22%
                   augment方案:brightness_blur augment + 常规augment

#### classifier:
            1.groups conv, bias=False, dilation=2,dropout等