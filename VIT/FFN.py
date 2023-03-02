"""
Author: xiao qiang
Time: 2023/2/25 22:31 
Version: env==torch py==3.9
"""
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_activation_layer, build_norm_layer, Linear
from mmcv.cnn.bricks.drop import build_dropout


class FFN(BaseModule):
    """
    Implements feed-forward networks with identity connection.
    Args:
        embed_dims(int): the feature dimension.Same as MultiHeadAttention, defaults:256
        feedforward_channels(int): the hidden dimension of FFNs. defaults: 1024
        num_fcs(int, optional): the number of fully_connected layers in FFNs, default:2
        act_cfg(dict, optional): the activation config for FFNs, defaults: dict(type='ReLU')
        ffn_drop(float, optional): Probability of an element to be zeroed in FFN, default:0.0
        add_identity(bool, optional): whether to add the identity connection, default: True.
        dropout_layer(obj:ConfigDict): the dropout_layer used when adding the shortcut.
        init_cfg(obj: ConfigDict): the config for initialization, default:None
    """
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=dict(type='ReLU'),
                 ffn_drop=0., dropout_layer=None, add_identity=True, init_cfg=None, **kwargs):
        super(FFN, self).__init__(init_cfg=init_cfg)
        assert num_fcs >= 2, f'num_fcs should be no less than 2, but got {num_fcs}'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs-1):
            layers.append(nn.Sequential(Linear(in_channels, feedforward_channels), self.activate, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)




