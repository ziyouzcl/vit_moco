"""
@来源: MoCo v3 vits.py（Attribution-NonCommercial 4.0）
       https://github.com/facebookresearch/moco-v3/blob/main/vits.py
@功能简述: 3D MoCo-ViT backbone 装配：在 VisionTransformer 上接 3D 位置编码与 MoCo 初始化
@主要用途: 供 MoCo_ViT 预训练或特征提取；YAML 通过 vit_3d_* 工厂函数实例化

@VisionTransformerMoCo3D 做了什么:
    1. 继承 vision_transformer.VisionTransformer（主体：patch_embed + blocks + head）
    2. pos_embed ← build_3d_sincos_position_embedding（固定 3D sin-cos，非可学习）
    3. init_weights(mode="moco") → 仅初始化 Transformer 内 nn.Linear（QKV 等）
    4. 若 patch_embed 为 PatchEmbedABC（如 PatchEmbed3D）：
         proj 用 MoCo Xavier（fan_in = in_chans × patch 体积）；可选 stop_grad_conv1 冻结 proj
       若为 ConvStem3D：不走第 4 步；stem 内 Conv3d 保持 PyTorch 默认 Kaiming

@与周边模块（零件 → 装配）:
    | 模块 | 文件 | 作用 |
    |------|------|------|
    | PatchEmbed3D | layers/patch_embed.py | 3D 体 → patch token（单层 Conv3d） |
    | ConvStem3D | 同上 | 多层 Conv3d stem → token |
    | build_3d_sincos_position_embedding | layers/position_embed.py | 3D 位置编码 |
    | VisionTransformer | vision_transformer.py | ViT 主体（Block / Attention） |
    | VisionTransformerMoCo3D | 本文件 | 组装上述零件 + MoCo 初始化 |

@前向数据流:
    输入 [B, C, D, H, W]
        → PatchEmbed3D 或 ConvStem3D  →  [B, N, embed_dim]
        → (+ cls_token) + 3D sin-cos pos_embed
        → Transformer Blocks → norm → head / 特征

@对外工厂（__all__）:
    vit_3d_small / vit_3d_base / vit_3d_conv_small / vit_3d_conv_base / vit_3d_base_patchsize8
    embed_layer: PatchEmbed3D（默认）或 ConvStem3D（conv 变体）
"""


import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from vit_moco.lib.models.vision_transformer import VisionTransformer, _cfg
from vit_moco.lib.layers.patch_embed import PatchEmbedABC, PatchEmbed3D, ConvStem3D
from vit_moco.lib.layers.position_embed import build_3d_sincos_position_embedding

__all__ = [
    'vit_3d_small', 
    'vit_3d_base',
    'vit_3d_conv_small',
    'vit_3d_conv_base',
    'vit_3d_base_patchsize8',
    'VisionTransformerMoCo3D',
]

class VisionTransformerMoCo3D(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 3D sin-cos position embedding
        self.pos_embed = build_3d_sincos_position_embedding(grid_size=self.patch_embed.grid_size, embed_dim=self.embed_dim, num_tokens=self.num_prefix_tokens)

        # weight initialization
        self.init_weights(mode="moco")

        if isinstance(self.patch_embed, PatchEmbedABC):
            # MoCo v3 Xavier for patch proj; leading factor is in_chans (2D 原版写死 3=RGB)
            in_chans = getattr(self.patch_embed, 'in_chans', 3)
            patch_vol = reduce(mul, self.patch_embed.patch_size, 1)
            val = math.sqrt(6. / float(in_chans * patch_vol + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False


def vit_3d_small(**kwargs):
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=PatchEmbed3D, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_base(**kwargs):
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=PatchEmbed3D, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_base_patchsize8(**kwargs):
    model = VisionTransformerMoCo3D(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=PatchEmbed3D, **kwargs)
    model.default_cfg = _cfg()
    return model