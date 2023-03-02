"""
Author: xiao qiang
Time: 2023/2/26 14:50 
Version: env==torch py==3.9
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmengine.utils import to_2tuple
from mmcv.cnn import build_conv_layer, build_activation_layer, build_norm_layer


class AdaptivePadding(nn.Module):
    """
    Applies padding adaptively to the input
    This module can make input get fully covered by filter you specified.It support two modes 'same' and 'corner'
    the 'same' mode is same with 'SAME' padding mode in tensorflow, pad zero around input, the 'corner' mode would
    pad zero to bottom right.
    Args:
        kernel_size(int|tuple): size of the kernel, default:1
        stride(int|tuple): stride of the filter, default:1
        dilation(int|tuple): spacing between kernel elements, default:1
        padding(str): support 'same' and 'corner', 'corner' mode would pad zero to bottom right, and 'same' mode would
            pad zero around input, default: 'corner'
    """
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """
        calculate the padding size of input
        Args:
            input_shape: Arrange as (H, W)
        Returns:
            Tuple[int]: the padding size along the original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                # F.pad(x, [left, right, top, bottom])
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2])
        return x


class PatchEmbed(BaseModule):
    """
    Image to Patch embedding
    use a conv layer to implement patch-embed.
    Args:
        in_channels(int): the number of input channels, defaults:3
        embed_dims(int): the dimension of embedding, default:768
        conv_type(str): the type of convolution to generate patch embedding, default:'conv2d'.
        kernel_size(int): the kernel_size of embedding conv, default: 16
        stride(int): the slide stride of embedding conv, default:16
        padding(int|tuple|string): the padding length of embedding conv, when it is a string, it means the mode of
            adaptive padding, support 'same' and 'corner' now, default:'corner'
        dilation(int): the dilation rate of embedding conv, default: 1
        bias(bool): bias of embed conv, default True
        norm_cfg(dict, optional): config dict for normalization layer, default:None
        input_size(int|tuple|None): the size of input, which will be used to calculate the out size, only works when
            'dynamic_size' is False, default:None
        init_cfg(dict, optional): the config for initialization, default:None
    """
    def __init__(self, in_channels=3, embed_dims=768, conv_type='Conv2d', kernel_size=16,
                 stride=16, padding='corner', dilation=1, bias=True, norm_cfg=None, input_size=None, init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            padding = 0
        else:
            self.adaptive_padding = 0
            padding = to_2tuple(padding)
        self.projection = build_conv_layer(dict(type=conv_type),
                                           in_channels=in_channels,
                                           out_channels=embed_dims,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           bias=bias)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)
            # 卷积后输出尺寸计算：out_h = (h+2*padding-kernel_size)/stride+1
            # 带有空洞卷积，有效卷积核大小为:new_kernel_size = kernel_size+(kernel_size-1)*(dilation-1)
            # 将有效卷积核大小带入卷积输出尺寸计算即可
            h_out = (input_size[0] + 2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)//stride[0]+1
            w_out = (input_size[1] + 2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)//stride[1]+1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        # tensor.flatten(start_dim, end_dim)
        # shape:[n, embed_dims, h, w] -> [n, embed_dims, h*w] -> [n, h*w, embed_dims] h*w表示token个数
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size







