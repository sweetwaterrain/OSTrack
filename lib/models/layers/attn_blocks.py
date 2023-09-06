import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    # 
    lens_s = attn.shape[-1] - lens_t    # lens_s是搜索区域的序列长度, attn.shape[-1]是序列的长度, lens_t是模板的序列长度
    bs, hn, _, _ = attn.shape   # bs是batch_size, hn是头数, attn.shape是[B, num_heads, L_t + L_s, L_t + L_s]

    lens_keep = math.ceil(keep_ratio * lens_s)  # lens_keep是保留的搜索区域的序列长度, keep_ratio是保留比例,math.ceil向上取整
    if lens_keep == lens_s:
        return tokens, global_index, None   # 如果lens_keep等于lens_s，就直接返回tokens, global_index, None

    # 取出attn的前lens_t行，和从lens_t开始的所有列，得到的形状是[B, num_heads, L_t, L_s]，表示模板和搜索区域的注意力矩阵
    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:  # 如果box_mask_z不为None，就使用box_mask_z
        # box_mask_z通过
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])    
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        # 先在维度2上求均值，形状为[B, H, L_s], 再在维度1上求均值，形状为[B, L_s]
        # attn_t表示搜索区域中与模板区域相关的注意力权重的加权平均值
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    # 对attn_t进行排序，得到的形状是[B, L_s]，dim=1表示在第1个维度上进行排序，descending=True表示降序排列,返回的是排序后的值和索引
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]    # topk_attn是前lens_keep个值，topk_idx是前lens_keep个值的索引
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]    # non_topk_attn是后lens_s - lens_keep个值，non_topk_idx是后lens_s - lens_keep个值的索引

    # global_index是全局索引，使用gather函数，得到保留的候选区域的索引
    keep_index = global_index.gather(dim=1, index=topk_idx) 
    # global_index是全局索引，使用gather函数，得到被消除的候选区域的索引
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]   # tokens_t是模板区域的tokens
    tokens_s = tokens[:, lens_t:]   # tokens_s是搜索区域的tokens

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape    # B是batch_size, L是序列长度，C是通道数
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    # token_s是搜索区域的tokens，形状是[B, L_s, C]，topk_idx是保留的候选区域的索引，形状是[B, lens_keep]，
    # 对topk_idx进行unsqueeze操作，得到的形状是[B, lens_keep, 1], 然后在最后一个维度上进行扩展，得到的形状是[B, lens_keep, C]
    # 使用gather函数，得到保留的候选区域的tokens，形状是[B, lens_keep, C]
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1) #  tokens_new是消除候选区域后的tokens，形状是[B, L_t + lens_keep, C]

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        # 对输入进行归一化，归一化的维度是dim，也就是输入的通道数，归一化的方式是LayerNorm
        self.norm1 = norm_layer(dim)
        # 进行多头注意力计算，注意力的头数是num_heads，qkv_bias是是否使用偏置，attn_drop是attention的dropout，proj_drop是attention后的dropout
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 如果drop_path大于0，就使用DropPath，否则使用nn.Identity()，用于实现随机深度（DropPath）。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # 进行MLP计算，MLP的输入维度是dim，MLP的隐藏层维度是dim * mlp_ratio，MLP的激活函数是act_layer，MLP的dropout是drop
        # MLP实现特征的非线性变换
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # keep_ratio_search是搜索区域的保留比例，如果keep_ratio_search小于1，就进行候选消除
        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True) # x_attn: B, N, C, attn: B, H, N, N，H是头数，N是序列长度，C是通道数
        x = x + self.drop_path(x_attn)  # 对x_attn进行dropout，得到的形状是[B, N, C], 然后与输入x相加，得到的形状是[B, N, C], 这里的加法是残差连接
        lens_t = global_index_template.shape[1] # lens_t是模板的序列长度

        removed_index_search = None # removed_index_search是被消除的候选区域的索引
        # 如果keep_ratio_search小于1, 并且keep_ratio_search为None，就进行候选消除
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            # 如果keep_ratio_search为None，就使用self.keep_ratio_search
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            # 对输入进行候选消除，得到的形状是[B, N, C], keep_index是保留的候选区域的索引，removed_index_search是被消除的候选区域的索引
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
