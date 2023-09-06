import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads  # 多头注意力的头数
        head_dim = dim // num_heads # 多头注意力的维度
        self.scale = head_dim ** -0.5   # 缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 对输入进行线性变换，变换的维度是dim * 3
        self.attn_drop = nn.Dropout(attn_drop)  # attention的dropout
        self.proj = nn.Linear(dim, dim)   # 对输入进行线性变换，变换的维度是dim
        self.proj_drop = nn.Dropout(proj_drop)  # 线性变换后的dropout

        self.rpe =rpe   # 是否使用相对位置编码
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape   # B: batch_size, N: 序列长度，C: 通道数
        # 首先对输入进行线性变换，变换的维度是dim * 3，然后将结果分成三份，分别作为q, k, v，得到的形状是[B, N, 3*num_heads, C // num_heads]
        # 然后通过reshape操作将张量重塑成[B, N, 3, num_heads, C // num_heads]
        # 然后通过permute操作变换张量的维度顺序为[3, B, num_heads, N, C // num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        # qkv[0]是q，qkv[1]是k，qkv[2]是v, qkv[0]的形状是[B, num_heads, N, C // num_heads]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # 通过矩阵乘法计算注意力，得到的形状是[B, num_heads, N, N], 也就是每个头的注意力矩阵,scale是缩放因子,N是序列长度
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        # 计算注意力矩阵的softmax，得到的形状是[B, num_heads, N, N], 也就是每个头的注意力矩阵
        attn = attn.softmax(dim=-1)
        # 对注意力矩阵进行dropout
        attn = self.attn_drop(attn)
        
        # 通过矩阵乘法计算注意力矩阵和v的乘积，可以理解为对每个位置的值进行加权求和，得到的形状是[B, num_heads, N, C // num_heads]
        # 然后通过transpose操作将张量重塑成[B, N, num_heads, C // num_heads]，然后通过reshape操作将张量重塑成[B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)    # 对输入进行线性变换，形状变成[B, N, C]
        x = self.proj_drop(x)   # dropout，得到的形状是[B, N, C]

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x