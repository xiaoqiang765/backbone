"""
Author: xiao qiang
Time: 2023/2/22 08:31 
Version: env==torch py==3.9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcls.models.utils.layer_scale import LayerScale


class MultiheadAttention(BaseModule):
    """
    Multi-head Attention Module.
    This module implements multi-head attention that supports different input dims and embed dims. and it also supports
    a shortcut from 'value', which is useful if input dims is not same with embed dims.
    Args:
        embed_dims(int): the embedding dimension.
        num_heads(int): parallel attention heads.
        input_dims(int, Optional): the input dimension, and if None, use 'embed_dims',defaults to None.
        attn_drop(float): dropout rate of the dropout layer after the attention calculation of query and key, defaults 0
        proj_drop(float): dropout rate of the dropout layer after the output projection.
        dropout_layer(dict): the dropout config before adding the shortcut, defaults to dict(type='Dropout', drop_prob=0)
        qkv_bias(bool): if True, add a learnable bias to q, k, v, defaults: True
        qk_scale(float, optional): override default qk scale of 'head_dim**-0.5', if set, defaults None
        proj_bias（bool）:if True, add a learnable bias to output projection, defaults to True.
        v_shortcut（bool）:add a shortcut from value to output,it is usually used if input_dims is different from embed_dims
        defaults:False
        init_cfg(dict, optional):the config fro initialization,defaults to None
    """
    def __init__(self, embed_dims, num_heads, input_dims=None, attn_drop=0., proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.), qkv_bias=True, qk_scale=None, proj_bias=True,
                 v_shortcut=False, use_layer_scale=False, init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)
        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.qkv = nn.Linear(input_dims, embed_dims*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = build_dropout(dropout_layer)
        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        # image shape: [B, N, patch_dim]
        B, N, _ = x.shape
        # input: [B, N, patch_dim], qkv后:[B, N, 3*embed_dims]
        # qkv: 获得指定head的qkv矩阵，reshape qkv:[B, N, 3, num_heads, head_dims] -> [3, B, num_heads, N, head_dims]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        # q,k,v shape: [B, num_heads, N, head_dims]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q@k.transpose(-2, -1):[B, num_heads, N, head_dims]@[B, num_heads, head_dims, N] = [B, num_heads, N, N]
        # 计算每张图片中每个head上q与k的attention, 后求softmax与dropout
        attn = (q@k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, num_heads, N, N]@[B, num_heads, N, head_dims] = [B, num_heads, N, head_dims]->[B, N, num_heads, head_dims]
        x = (attn@v).transpose(1, 2).reshape(B, N, self.embed_dims)
        # 多头注意力机制出来后concat后再次进行线性映射，并且添加可学习的bias
        x = self.proj(x)
        # 对映射后的输出进行dropout，并进行gamma1缩放，后再进行drop_path正则化
        # x:[B, N, embed_dims]
        x = self.out_drop(self.gamma1(self.proj_drop(x)))
        if self.v_shortcut:
            x = v.squeeze(1)+x
        return x


def resize_pos_embed(pos_embed, src_shape, dst_shape, mode='bicubic', num_extra_tokens=1):
    """
    Resize pos_embed weights.
    Args:
        pos_embed(torch.Tensor): Position embedding weights with shape [1, L, C].
        src_shape(tuple): The resolution of down_sampled origin training image, in format(H, W).
        dst_shape(tuple): The resolution of down_sampled new training image, in format(H, W)
        mode(str): Algorithm used for up_sampling, choose one from 'nearest', 'linear','bilinear', 'bicubic' and
            'trilinear'
        num_extra_tokens(int): The number of extra tokens, such as cls_token, defaults to 1
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, f'the length of "pos_embed" should equal src_h*src_w+extra_tokens'
    extra_tokens = pos_embed[:, :num_extra_tokens]
    src_weight = pos_embed[:, num_extra_tokens:]
    # src_weight: [1, L-extra_tokens, C] -> [1, src_h*src_w, C] -> [1, C, src_h*src_w]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)
    dst_weight = F.interpolate(src_weight, size=dst_shape, align_corners=False, mode=mode)
    # dst_weight: [1, C, dst_h*dst_w] - >[1, C, L] -> [1, L, C]
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    return torch.cat((extra_tokens, dst_weight), dim=1)




