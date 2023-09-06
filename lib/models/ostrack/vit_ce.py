import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        # 判断img_size是否是tuple，如果是tuple，就直接赋值，如果不是tuple，就将img_size转换为tuple
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 进行patch embedding，将输入的图片转换为patch，形状由(B, C, H, W)转换为(B, N, C)，N是H*W
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # 获取patch的数量
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 用于分类的token，形状是(1, 1, embed_dim)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None # 用于蒸馏的token，形状是(1, 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # 位置编码，形状是(1, num_patches + self.num_tokens, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate) # dropout层, 用于对token进行dropout

        # drop_path_rate是随机深度衰减率，depth是transformer的层数，dpr是一个长度为depth的列表，列表中的每个元素都是一个随机深度衰减率.
        # 这样，列表 dpr 中的每个值表示在模型的每个层中应用随机深度的概率。随着层的增加，随机深度的概率会逐渐增加，从而实现随机深度衰减。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = [] # 用于存储CEBlock对象
        ce_index = 0    # 用于记录当前的ce_keep_ratio
        self.ce_loc = ce_loc    # 包含了需要进行CE的层的索引
        for i in range(depth):
            ce_keep_ratio_i = 1.0   # ce_keep_ratio_i设置为1.0，表示进行CE
            if ce_loc is not None and i in ce_loc:
                # 如果ce_loc不为None，且i在ce_loc中，就将ce_keep_ratio_i设置为ce_keep_ratio[ce_index]
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                # ce_index加1，以便在下一次迭代中获取下一个位置的ce_keep_ratio
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        # 获取batch size，高，宽 0：batch size 1：通道数 2：高 3：宽
        # z是模板图片，x是搜索图片，设定z大小是128*128，x大小是256*256
        B, H, W = x.shape[0], x.shape[2], x.shape[3]    # B是batch size，H是高，W是宽，在这里H和W都是256

        # patch embedding，将输入的图片转换为patch，形状由(B, C, H, W)转换为(B, N, C)，N是H*W
        x = self.patch_embed(x) 
        z = self.patch_embed(z) 

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # 添加positional embedding, z和x的positional embedding是不一样的, z的positional embedding是固定的，x的positional embedding是随机的
        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
        
        x = combine_tokens(z, x, mode=self.cat_mode)    # 将z和x拼接到第1维上，得到的形状是(B, N, C)，N是H*W
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        # 对x进行dropout，这里的dropout是对token进行dropout
        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]  # lens_z是模板区域的序列长度
        lens_x = self.pos_embed_x.shape[1]  # lens_x是搜索区域的序列长度

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device) # global_index_t是模板区域的全局索引,使用linspace函数生成一个等差数列，数列的起始值是0，终止值是lens_z-1，数列的长度是lens_z
        global_index_t = global_index_t.repeat(B, 1)    # 将global_index_t在第0维上进行复制，复制B次，得到的形状是(B, lens_z)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device) # global_index_s是搜索区域的全局索引,使用linspace函数生成一个等差数列，数列的起始值是0，终止值是lens_x-1，数列的长度是lens_x
        global_index_s = global_index_s.repeat(B, 1)    # 将global_index_s在第0维上进行复制，复制B次，得到的形状是(B, lens_x)
        removed_indexes_s = []  # removed_indexes_s是被消除的候选区域的索引
        for i, blk in enumerate(self.blocks):   # 遍历每个CEBlock
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)  # 对x进行CE，得到的形状是(B, N, C)，global_index_t是模板区域的全局索引，global_index_s是搜索区域的全局索引，removed_index_s是被消除的候选区域的索引，attn是注意力矩阵

            if self.ce_loc is not None and i in self.ce_loc:    # 如果self.ce_loc不为None，且i在self.ce_loc中，就将removed_index_s添加到removed_indexes_s中
                removed_indexes_s.append(removed_index_s)   # removed_index_s是被消除的候选区域的索引

        x = self.norm(x)    # 对x进行归一化，归一化的维度是dim，也就是输入的通道数，归一化的方式是LayerNorm，[1, 153, 768]
        lens_x_new = global_index_s.shape[1]    # lens_x_new是消除候选区域后的搜索区域的序列长度，此时的global_index_s是消除候选区域后的搜索区域的全局索引，形状是(B, lens_x_new)
        lens_z_new = global_index_t.shape[1]    

        z = x[:, :lens_z_new]   # z是消除候选区域后的模板区域，形状是(B, lens_z_new, C)
        x = x[:, lens_z_new:]   # x是消除候选区域后的搜索区域，形状是(B, lens_x_new, C)

        if removed_indexes_s and removed_indexes_s[0] is not None:  # 如果removed_indexes_s不为空，且removed_indexes_s[0]不为None
            # 将所有的removed_indexes_s拼接到一起，得到的形状是(B, lens_removed)，lens_removed是被消除的候选区域的数量
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new   # pruned_lens_x是被消除的候选区域的数量
            
            # pad_x是被消除的候选区域的tokens，形状是(B, pruned_lens_x, C)，这里的pad_x是全0的张量
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device) 
            
            # 将pad_x拼接到x的第1维上，得到的形状是(B, lens_x, C)，使得搜索区域的序列长度恢复到lens_x
            x = torch.cat([x, pad_x], dim=1)    # 将pad_x拼接到x的第1维上，得到的形状是(B, lens_x, C)

            # 将removed_indexes_cat拼接到global_index_s的第1维上，此时的global_index_s是消除候选区域后的搜索区域的全局索引，得到的形状是(B, lens_x_new + lens_removed)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1) # 形状是(B, lens_x_new + lens_removed) = (B, lens_x)
            # recover original token order
            C = x.shape[-1] # 获取x的最后一个维度，也就是通道数
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            # 先使用zeros_like函数创建一个全0的张量，形状是(B, lens_x, C)
            # scatter_函数的作用是将x中的元素按照index_all中的索引进行填充，得到的形状是(B, lens_x, C)
            # 其中index_all先进行unsqueeze操作，得到的形状是(B, lens_x, 1)，然后在最后一个维度上进行扩展，得到的形状是(B, lens_x, C)】
            # src=x可以理解为将x中的元素作为填充的元素，index_all可以理解为填充的索引
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)   # 对x进行恢复，得到的形状是(B, lens_x, C)

        # re-concatenate with the template, which may be further used by other modules
        # 重新将模板区域的tokens和搜索区域的tokens拼接到一起，得到的形状是(B, lens_z_new + lens_x, C)
        x = torch.cat([z, x], dim=1)    # 将z和x拼接到第1维上，得到的形状是(B, lens_z_new + lens_x, C)  [1, 320, 768]    

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu") # 加载预训练模型，map_location="cpu"表示将模型加载到cpu上
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)    # 加载预训练模型的参数，strict=False表示不严格匹配
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)    # patch_size是patch的大小，embed_dim是嵌入维度，depth是transformer的层数，num_heads是注意力头的数量
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
