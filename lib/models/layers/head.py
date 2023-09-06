import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.models.layers.frozen_bn import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        """
            :param inplanes: 输入通道数
            :param channel: 输出通道数
            :param feat_sz: 特征图的大小
            :param stride: 特征图的步长
            :param freeze_bn: 是否冻结BN层
        """
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz  # 特征图的大小
        self.stride = stride    # 特征图的步长
        self.img_sz = self.feat_sz * self.stride    # 图像的大小，特征图的大小乘以步长

        # 卷积计算公式：(W - F + 2P) / S + 1

        # corner predict
        # conv 是自定义的函数，用于构建卷积层
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        # 这个卷积层用于预测中心位置的得分，它将输入特征图映射到一个单通道的特征图，用于预测中心位置的得分。
        # 在OSTrack中，这个得分是用于计算中心目标的位置和尺寸。
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # offset predict
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        # 这个卷积层用于预测中心位置的偏移量，它将输入特征图映射到一个两通道的特征图，用于预测中心位置的偏移量。
        # 通道数为2，是因为中心位置的偏移量有两个维度，一个是x方向，一个是y方向。
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size predict
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        # 这个卷积层用于预测中心位置的尺寸，它将输入特征图映射到一个两通道的特征图，用于预测中心位置的尺寸。
        # 通道数为2，是因为中心位置的尺寸有两个维度，一个是宽度，一个是高度。
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1) 

        # Xavier均匀初始化是一种常用的参数初始化方法，旨在使权重在前向传播过程中保持均匀的方差。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    """
        cal_bbox用于计算边界框坐标, 它的输入是:
                中心位置的预测得分图(score_map_ctr)
                目标尺度的预测得分图(score_map_size)
                目标位置偏移的预测得分图(score_map_offset)。 
    """
    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        # 这行代码使用torch.max()函数找到score_map_ctr中每个通道上的最大值，并返回最大值和对应的索引。
        # 从二维形状(B, 1, H, W)压缩为一维形状(B, H * W)，其中每个通道的元素都被展平。
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)   # flatten(1)表示从第一个维度将后面的转化成1维，dim=1表示按行取最大值, keepdim=True表示保持维度不变
        idx_y = idx // self.feat_sz # idx_y表示索引值对应的垂直方向上的坐标
        idx_x = idx % self.feat_sz  # idx_x表示索引值对应的水平方向上的坐标

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)   # 将idx的维度从(B, 1)扩展为(B, 2, 1)，复制每个索引值在第二个维度上。
        
        # 这行代码使用索引操作从 size_map 获取与 idx 相对应的尺寸
        # size_map 是一个形状为 (batch_size, 2, H，W)，其中每个通道存储着预测目标的宽度和高度。
        # idx 是一个形状为 (batch_size, 2, 1) 的索引张量，其中每个元素是对应通道中要提取的尺寸值的索引。
        # 通过执行 size_map.flatten(2)，可以将 size_map 从形状 (batch_size, 2, H，W) 平铺为形状 (batch_size, 2, HW)，以便在接下来的索引操作中进行使用。
        # 然后，使用 gather 函数根据索引 idx 在第三个维度上进行检索，得到形状为 (batch_size, 2, 1) 的张量，其中包含了对应索引的尺寸值。
        size = size_map.flatten(2).gather(dim=2, index=idx)

        # 这行代码使用索引操作从 offset_map 获取与 idx 相对应的偏移量。
        # 通过执行 offset_map.flatten(2)，可以将 offset_map 张量的最后两个维度展平为一个维度，得到形状为 (batch_size, 2, feat_sz*feat_sz) 的张量。
        # 然后，使用 gather 函数根据索引 idx 在第三个维度上进行检索，得到形状为 (batch_size, 2, 1) 的张量，其中包含了对应索引的偏移值。
        # 最后，通过执行 squeeze(-1)，可以将结果张量的最后一个维度压缩，得到形状为 (batch_size, 2) 的张量 offset，其中存储了对应偏移值。
        # 这样，offset 张量就可以用于计算边界框的位置。
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h

        # 这行代码使用 torch.cat() 函数将 idx_x 和 idx_y 连接起来，得到形状为 (batch_size, 2) 的张量，其中包含了中心位置的坐标。
        # /self.feat_sz 是为了将坐标值从特征图空间转换到图像空间。
        # squeeze(-1) 是为了压缩 size 张量的最后一个维度, 得到形状为 (batch_size, 2) 的张量。
        # 然后cat()函数将 idx_x, idx_y 和尺寸张量连接起来，得到形状为 (batch_size, 4) 的张量(bbox)，其中包含了中心位置的坐标。
        bbox = torch.cat([
                (idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz, (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                size.squeeze(-1)
                ], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)  # [1, 1, 16, 16]

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4) # [1, 2, 16, 16]

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)   # [1, 2, 16, 16]
        # 返回目标中心的预测得分图(score_map_ctr)和目标尺度的预测得分图(score_map_size)，使用sigmoid函数映射到(0,1)范围，避免了得分值过于极端的情况，同时保留了预测得分的相对大小关系。
        # score_map_offset用于表示目标位置偏移的预测得分图。由于偏移可以取负值或较大的正值，不需要经过sigmoid函数处理，因为它的取值范围不需要限制在 (0, 1) 之间。
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if cfg.MODEL.HEAD.TYPE == "MLP":
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)