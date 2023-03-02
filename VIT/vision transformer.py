"""
Author: xiao qiang
Time: 2023/2/21 22:37 
Version: env==torch py==3.9
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from mmengine.model import BaseModule, ModuleList
from mmcv.cnn import build_norm_layer
from mmcls.models.utils import MultiheadAttention, to_2tuple, resize_pos_embed
from mmcv.cnn.bricks.transformer import FFN
from mmcls.models.backbones.base_backbone import BaseBackbone
from patch_embed import PatchEmbed
from mmengine.model.weight_init import trunc_normal_


class TransformerEncoderLayer(BaseModule):
    """
    Implement one encoder layer in vision transformer.
    Args:
        embed_dims（int）: the feature dimension.
        num_heads（int）: parallel attention heads.
        feedforward_channels（int）:the hidden dimension for FFNs.
        drop_rate（float）:probability of element to be zeroed after the feed forward layer, defaults 0.
        attn_drop_rate（float）: the drop out rate for attention output weights.
        drop_path_rate（float）:stochastic depth rate, defaults to 0.
        num_fcs（int）: the number of fully_connected layers for FFNs, defaults to 2.
        qkv_bias（bool）:enable bias for qkv if True, defaults to True.
        act_cfg（dict）: the activation config for FFNS, defaults to dict(type='GELU')
        norm_cfg（dict）:config dict for normalization layer, defaults to dict(type='LN')
        init_cfg（dict）: initialization config dict, defaults to None.
    """
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., num_fcs=2, qkv_bias=True, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = MultiheadAttention(embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate,
                                       proj_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                                       qkv_bias=qkv_bias)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=num_fcs,
                       ffn_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate, act_cfg=act_cfg))

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init_xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x


class VisionTransformer(BaseBackbone):
    """
    Vision Transformer: A pytorch implement of 'an image is worth 16*16 words, transformers for image_recognition'
    Args:
        arch(str|dict): vision transformer architecture, if use string, choose from 'small','base','large','deit-tiny',
            'deit-small' and 'deit-base', if use dict, it should have below keys:
            - **embed_dims**(int): the dimension of embedding.
            - **num_layers**(int): the number of transformer encoder layers.
            - **num_heads**(int): the number of heads in attention modules.
            - **feedforward_channels**(int): the hidden dimensions in feedforward modules
            defaults: 'base'
        img_size(int|tuple): the expected input image shape, because we support dynamic input shape, just set the argument
            to the most common input image shape, defaults to 224.
        patch_size(int|tuple): the patch size in patch embedding. defaults to 16.
        in_channels(int): the number of input channels, defaults:3
        out_indices(sequence|int): output from which stages, defaults to -1, means the last stage
        drop_rate(float): probability of an element to be zeroed. defaults to 0
        drop_path_rate(float): stochastic depth rate, defaults to 0.
        qkv_bias(bool): whether to add bias for qkv in attention modules.
        norm_cfg(dict): config dict for normalization layer, defaults to dict(type='LN')
        final_norm(bool): whether to add a layer to normalize final feature map, defaults to True.
        with_cls_token(bool): whether concatenating class token into image tokens as transformer input, defaults True
        avg_token(bool): whether to use the mean patch token for classification, if true, the model will only
            take the average of all patch tokens, defaults to False
        frozen_stages(int): stages to be frozen, -1 means not freezing any parameters, defaults to -1
        output_cls_token(bool): whether output cls token, if set true, with_cls_token must be true,defaults to True
        beit_style(bool): whether to use beit_style, defaults to False
        layer_scale_init_value(float): the initialization value for the learnable scaling of attention and FFN, default
            to 0.1
        interpolate_mode(str):select the interpolate mode for position embeding vector resize, defaults to 'bicubic'
        patch_cfg(dict): configs of patch embeding, defaults to an empty dict
        layer_cfgs(sequence|dict): configs of each transformer layer in encoder, defaults to an empty dict
        init_cfg(dict, optional): initialization config dict, defaults to None
    """
    arch_zoo = {
        **dict.fromkeys(['s', 'small'], {'embed_dims': 768, 'num_layers': 8, 'num_heads': 8, 'feedforward_channels':768*3}),
        **dict.fromkeys(['b', 'base'], {'embed_dims': 768, 'num_layers': 12, 'num_heads': 12, 'feedforward_channels': 3072}),
        **dict.fromkeys(['l', 'large'], {'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'feedforward_channels': 4096}),
        **dict.fromkeys(['h', 'huge'], {'embed_dims': 1280, 'num_layers': 32, 'num_heads': 16, 'feedforward_channels': 5120}),
        **dict.fromkeys(['deit-t', 'deit-tiny'], {'embed_dims': 192, 'num_layers': 12, 'num_heads': 3, 'feedforward_channels': 192*4}),
        **dict.fromkeys(['deit-s', 'deit-small'], {'embed_dims': 384, 'num_layers': 12, 'num_heads': 6, 'feedforward_channels': 384*4}),
        **dict.fromkeys(['deit-b', 'deit-base'], {'embed_dims': 768, 'num_layers': 12, 'num_heads': 12, 'feedforward_channels': 768*4})
    }
    # some structures have multiple extra tokens, like Deit
    num_extra_tokens = 1 # cls_token

    def __init__(self, arch='base', img_size=224, patch_size=16, in_channels=3, out_indices=-1, drop_rate=0.,
                 drop_path_rate=0., qkv_bias=True, norm_cfg=dict(type='LN'), final_norm=True, with_cls_token=True,
                 avg_token=False, frozen_stages=-1, output_cls_token=True, beit_style=False, layer_scale_init_value=0.1,
                 interpolate_mode='bicubic', patch_cfg=dict(), layer_cfgs=dict(), init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            # set(dict) -> dict.keys
            assert arch in set(self.arch_zoo), f'arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'}
            assert isinstance(arch, dict) and essential_keys <= set(arch), f'custom arch needs a dict with keys ' \
                                                                           f'{essential_keys}'
            self.arch_settings = arch
        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        # img_size=224 -> (224, 224)
        self.img_size = to_2tuple(img_size)
        # set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        # execute an image to patch token
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0]*self.pathch_resolution[1]

        # set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if set output_cls_token to True'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        # set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_extra_tokens, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        # set output indices
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), f'out_indices must be sequence or int, but got {type(out_indices)}'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers
        self.out_indices = out_indices
        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs]*self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            if beit_style:
                pass
            else:
                self.layers.append(TransformerEncoderLayer(**_layer_cfg))
        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)
        self.avg_token = avg_token
        if avg_token:
            self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.embed_dims, postfix=2)
            self.add_module(self.norm2_name, norm2)
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(VisionTransformer, self).init_weights()
        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return
        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(f'resize the pos_embed shape from {ckpt_pos_embed_shape} to {self.pos_embed.shape}')
            ckpt_pos_embed_shape = to_2tuple(int(np.sqrt(ckpt_pos_embed_shape.shape[1]-self.num_extra_tokensda)))
            pos_embed_shape = self.patch_embed.init_out_size
            state_dict[name] = resize_pos_embed(state_dict[name], ckpt_pos_embed_shape, pos_embed_shape,
                                                self.interpolate_mode, self.num_extra_tokens)

    def _freeze_stages(self):
        # freeze order: pos_embed, pos_embed_dropout -> patch_embed -> cls_token -> layers -> if freeze last layers \
        # freeze the norm
        # 涉及dropout, bn层的冻结时需要eval()模式
        # freeze position embedding
        self.pos_embed.requires_grad = False
        # set dropout to eval mode
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls token
        self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages+1):
            m = self.layers[i-1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.norm1.eval()
            for param in self.norm1.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        # cls_token: [1, 1, n_dim] -> [B, 1, n_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # pos_embed + patch_embed
        x = x + resize_pos_embed(self.pos_embed, self.patch_resolution, patch_resolution, mode=self.interpolate_mode,
                                 num_extra_tokens=self.nu_extra_tokens)
        x = self.drop_after_pos(x)
        if not self.with_cls_token:
            x = x[:, 1:]
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    # patch_token:[B, h, w, c] -> [B, c, h, w]
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.avg_token:
                    # [B, h, w, c]
                    patch_token = patch_token.permute(0, 2, 3, 1)
                    patch_token = patch_token.reshape(B, patch_resolution[0]*patch_resolution[1], C).mean(dim=1)
                    patch_token = self.norm2(patch_token)
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)
        return tuple(outs)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        return resize_pos_embed(*args, **kwargs)


















