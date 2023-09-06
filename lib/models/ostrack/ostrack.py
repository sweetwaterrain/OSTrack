"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        # 是否使用aux_loss，aux_loss是什么？，aux_loss是辅助损失，用于辅助训练
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            # feat_sz_s: 特征图的大小, feat_len_s: 特征图的大小的平方
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        # x: 特征提取结果，aux_dict: 辅助字典，包含了中间层的输出
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        # 判断x是否是list，如果是list，取最后一个元素，x如果是list，就说明有多个输出，这里只取最后一个输出
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        # out的输出形状是{'pred_boxes': (B, Nq, 4), 'score_map': (B, Nq, 1, H, W)}, 'size_map': (B, Nq, 2, H, W), 'offset_map': (B, Nq, 2, H, W)}, 'backbone_feat': (B, HW1+HW2, C)
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
            cat_feature 是backbone(ViT)的输出, 形状为(B, HW1+HW2, C), HW1是模板的特征图大小, HW2是搜索区域的特征图大小
        """
        # enc_opt是取cat_feature的后半部分（self.feat_len_s长度），也就是搜索区域的特征图，enc_opt的形状是(B, HW2, C)
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # 先进行维度扩展，形状变成(B, HW2, C, 1)，然后进行维度转换，形状变成(B, 1, C, HW2)，这个维度置换操作是为了符合后续处理的要求，特别是与box head的输入格式匹配。
        # contigous()函数是为了保证内存是连续的，这样才能进行view操作
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous() # (B, 1, C, HW2)
        # 调用size()函数获取opt的形状，bs是batch size，Nq表示查询数量（number of queries），在目标跟踪中，通常为1，表示只有一个查询。
        # C是通道数，HW是特征图的空间尺寸。
        bs, Nq, C, HW = opt.size() 
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s) # (B*Nq, C, H, W)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            # 得分图的形状是(B*Nq, 1, H, W)，bbox的形状是(B*Nq, 4)，size_map的形状是(B*Nq, 2, H, W)，offset_map的形状是(B*Nq, 2, H, W)
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            
            outputs_coord = bbox # (B*Nq, 4)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training: # 如果Model中指定了预训练模型，且不是OSTrack的预训练模型，就使用指定的预训练模型，并且是训练模式
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim # 这一步是为了获取backbone的输出维度，这个维度是用于box head的输入维度
        patch_start_index = 1   # 这个参数是用于指定哪些层需要进行finetune，这里指定从第一层开始finetune

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)   # 这一步是为了指定哪些层需要进行finetune，这里指定从第一层开始finetune

    box_head = build_box_head(cfg, hidden_dim)  # 这一步是为了构建box head，这里的hidden_dim是backbone的输出维度，也就是box head的输入维度

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )   # 这一步是为了构建OSTrack模型，这里的backbone是上面构建的backbone，box_head是上面构建的box head，aux_loss是指定是否使用aux loss，head_type是指定box head的类型

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

