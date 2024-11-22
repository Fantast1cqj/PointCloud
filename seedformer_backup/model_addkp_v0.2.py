'''
==============================================================

SeedFormer: Point Cloud Completion
-> SeedFormer + key poins cross attention

==============================================================

Author:
Date: 2024-10-17
version: 0.2
==============================================================
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import os
import numpy as np
from models.utils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor
def square_distance(src, dst):
       """
       src: (B, N, F)
       dst: (B, M, F)
       return: (B, N, F)  N points M feature
       """
       pos_src = src.unsqueeze(2) # (B, N, 1, 3)
       pos_dst = dst.unsqueeze(1) # (B, 1, M, 3)
       
       pos_em = pos_src - pos_dst # (B, N, M, 3)
       pos_sq = pos_em.pow(2).sum(dim=2) # (B, N, 3)
   
       return pos_sq
   
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    # points_flip = points.permute(0, 2, 1)
    # idx_flip = idx.permute(0, 2, 1)
    
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class FeatureExtractor_source(nn.Module):
    def __init__(self, out_dim=1024, n_knn=20):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor_source, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, partial_cloud):
        """
        Args:
             partial_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)  l1_xyz：512个点xyz坐标 l1_points：512个点，128维特征
        l1_points = self.transformer_1(l1_points, l1_xyz)   # l1_points用transformer增强特征
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)  512个点变128个，特征变256
        l2_points = self.transformer_2(l2_points, l2_xyz)   # l2_points用transformer增强特征
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)   128个点变成1个点，特征变成一个out_dim维的全局特征

        return l3_points, l2_xyz, l2_points

class FeatureExtractor_force(nn.Module):     # 直接把 partial_cloud 和 kp 加起来
    def __init__(self, out_dim=1024, n_knn=20):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor_force, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, partial_cloud, kp):
        """
        Args:
             partial_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        mix_point = torch.cat((partial_cloud, kp), dim=2)
        l0_xyz = mix_point
        l0_points = mix_point

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)  l1_xyz：512个点xyz坐标 l1_points：512个点，128维特征
        l1_points = self.transformer_1(l1_points, l1_xyz)   # l1_points用transformer增强特征
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)  512个点变128个，特征变256
        l2_points = self.transformer_2(l2_points, l2_xyz)   # l2_points用transformer增强特征
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)   128个点变成1个点，特征变成一个out_dim维的全局特征

        return l3_points, l2_xyz, l2_points





class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024, n_knn=20):
        """
        Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)
        self.sa_kp_1 = PointNet_SA_Module_KNN(128, 8, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_kp_2 = PointNet_SA_Module_KNN(128, 8, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        
        
        
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        self.kp_transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.kp_transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        
        

    def forward(self, partial_cloud, kp):
        """
        Args:
             partial_cloud: b, 3, n
             kp: b,3,128

        Returns:
            l3_points: (B, out_dim, 1)
        """
       
        
        l0_xyz = partial_cloud  # (B, 3, 2048)
        l0_points = partial_cloud
        
        kp_xyz_0 = kp
        kp_feat_0 = kp
        
        
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)  l1_xyz：512个点xyz坐标 l1_points：512个点，128维特征
        l1_points = self.transformer_1(l1_points, l1_xyz)   # l1_points用transformer增强特征
        kp_xyz1, kp_feat1, pk_id1 = self.sa_kp_1(kp_xyz_0, kp_feat_0)  # (B, 3, 128)  (B, 128, 128)
        kp_feat1 = self.kp_transformer_1(kp_feat1, kp_xyz1)            # (B, 128, 128)
        
        
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)  512个点变128个，特征变256
        l2_points = self.transformer_2(l2_points, l2_xyz)   # l2_points用transformer增强特征
        kp_xyz2, kp_feat2, pk_id2 = self.sa_kp_2(kp_xyz1, kp_feat1)    # (B, 3, 128)  (B, 256, 128)
        kp_feat2 = self.kp_transformer_2(kp_feat2, kp_xyz2)            # (B, 256, 128)
        

        
        # l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)   128个点变成1个点，特征变成一个out_dim维的全局特征

        return l2_xyz, l2_points, kp_xyz2, kp_feat2


# 加 cross attention
class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=20, factor=2, attn_channel=True):
        super(SeedGenerator, self).__init__()
        # self.cross_atten = UpTransformer_cross(nhead = 4, dropout = 0.0)  # output: (B, 256, 128)
        self.uptrans = UpTransformer(256, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel, up_factor=factor, scale_layer=None)  # 输入256维度特征，输出128维特征
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor (B, feat_dim, 1)    feat_dim 维度的全局特征 1个
            patch_xyz: (B, 3, 128)           128个点 来自 partial 点云的降采样
            patch_feat: (B, 256, 128)   128个点的局部特征 seed_dim=256 维度特征
            kp_xyz: (B, 3, 24)
            kp_feat: (B, 256, 24)
        """
        # patch_feat = self.cross_atten(patch_xyz, kp_xyz, kp_feat, patch_feat, kp_feat)  # (B, 256, 128)
        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)  # (B, 128, 256)  进行上采样128个点变256个点 特征由256维变为128维
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1)) # 上采样后的特征与全局特征连接
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256) 再次将全局特征与处理后的特征连接
        completion = self.mlp_4(x3)  # (B, 3, 256)  得到种子点的三维坐标
        return completion, x3

class UpTransformer_cross(nn.Module):
    def __init__(self, nhead, dropout):
        super(UpTransformer_cross, self).__init__()
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # new2
        # self.nhead = 4
        # dropout = 0.0
        d_model_out = 256
        dim_feedforward = 1024
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.activation1 = torch.nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.pos_encoder = nn.Conv1d(256,256,1)

    def forward(self, pos_pc, pos_kp, key, query, value):   # upfeat 是上采样的 Seed Feature
        """
        Seed Generator: 
            value: (B, kp_f=256, N) keypoint
            key: (B, kp_f=256, N) keypoint
            pos_kp: (B, 3, N) N = 24
            
            pos_pc: position (B, 3, 128)
            query: patch_feat (B, 256, 128)
        """
        # print("value & key shape:", value.shape) # torch.Size([6, 256, 24])
        # print("pos_kp shape:", pos_kp.shape) # torch.Size([6, 3, 24])
        # print("pos_pc shape:", pos_pc.shape) # torch.Size([6, 3, 128])
        # print("query shape:", query.shape) # torch.Size([6, 256, 128])
        # exit()
        # new2 
        kp = pos_kp.permute(0, 2, 1) # (B, 24, 3)
        pc = pos_pc.permute(0, 2, 1) # (B, 128, 3)
        kp = self.fc_delta(kp) # (B, 24, 256)
        pc = self.fc_delta(pc) # (B, 128, 256)
        dists_kp = square_distance(kp, pc) # (B, 24, 256)
        dists_pc = square_distance(pc, kp) # (B, 128, 256)
        
        dists_kp = dists_kp.permute(0, 2, 1) # (B, 256, 24)
        dists_kp = self.pos_encoder(dists_kp).permute(0, 2, 1)
        
        dists_pc = dists_pc.permute(0, 2, 1)
        dists_pc = self.pos_encoder(dists_pc).permute(0, 2, 1)
        
        
        # print("dists_kp: ", dists_kp)
        # exit()
        
        q = query.permute(0, 2, 1) + dists_pc # (B, 128, 256)
        k = key.permute(0, 2, 1) + dists_kp   # (B, 24, 256)
        v = value.permute(0, 2, 1) + dists_kp # (B, 24, 256)
        # kt = k.transpose(1, 2) # (B, 256, 64)
        # attention = torch.bmm(q, kt) # (B, 128, 64)
        # result = torch.bmm(attention, v) # (B, 128, 256)
        q = q.permute(1, 0, 2) # (N, B, F)  (128, B, 256)
        k = k.permute(1, 0, 2) # (M, B, F)  (24, B, 256)
        v = v.permute(1, 0, 2) # (M, B, F)  (24, B, 256)
        src12 = self.multihead_attn1(query = q,
                                     key = k,
                                     value = v)[0]
        
        # Add & Norm
        q = q + self.dropout12(src12)  # 残差连接
        q = self.norm12(q)
        # FFN
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(q))))
        # Add
        q = q + self.dropout13(src12)
        q = q.permute(1, 2, 0) # (B, F, N) (B, 256, 128)
        
        return q

        
class patch_kp_mix(nn.Module):
    def __init__(self):
        super(patch_kp_mix, self).__init__()
        self.kpfeat_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1280, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
        )
        
        self.down_sample = PointNet_SA_Module_KNN(128, 16, 256, [256, 256], group_all=False, if_bn=False, if_idx=True) # 下采样到128个点
        
    def forward(self, patch_xyz, patch_feat, kp_xyz, kp_feat):
        """
        Inputs:
            
            patch_xyz: (B, 3, 128)
            patch_feat: (B, 256, 128)
            kp_xyz: (B, 3, 128)
            kp_feat: (B, 256, 128)
            
        """
        mix_point = torch.cat((kp_xyz, patch_xyz), dim=2) # (B, 3, 256)
        
        # 看 mix_point 可视化结果
        # first = mix_point.cuda(0)[0]
        # first = first.permute(1, 0)
        # tensor_cpu = first.cpu()      # 转换为 cpu 张量
        # kp_cpu = tensor_cpu.numpy()    # 张量转 numpy
        # file_name = f'mix_point_c.npy'
        # base_path = '../mix_point'
        # file_path = os.path.join(base_path, file_name)
        # np.save(file_path, kp_cpu)     # 保存所有
        
        
        # kp_feat = self.kpfeat_extractor(kp) # (B, 256, 128)
        feat_cat_256 = torch.cat((kp_feat, patch_feat), dim=2) # (B, 256, 256)
        
        # 不加 max 试试
        feat_cat_1024 = self.mlp1(feat_cat_256) # (B, 1024, 256)
        feat_max, _ = torch.max(feat_cat_1024, dim=2, keepdim=True) # (B, 1024, 1)
        feat_max_rep = feat_max.repeat(1, 1, 256) # (B, 1024, 256)
        feat_cat2 = torch.cat((feat_cat_256, feat_max_rep), dim=1) # (B, 1024+256=1280, 256)
        mix_feat = self.mlp2(feat_cat2) # (B, 256, 256)
        

        
        mixed_points, mixed_feat, id_ = self.down_sample(mix_point, mix_feat) # (B, 3, 128) (B, 256, 128)
        
        # 看 mixed_points 可视化结果
        # first2 = mixed_points.cuda(0)[0]
        # first2 = first2.permute(1, 0)
        # tensor_cpu2 = first2.cpu()      # 转换为 cpu 张量
        # kp_cpu2 = tensor_cpu2.numpy()    # 张量转 numpy
        # file_name2 = f'mixed_point_c.npy'
        # base_path2 = '../mix_point'
        # file_path2 = os.path.join(base_path2, file_name2)
        # np.save(file_path2, kp_cpu2)     # 保存所有
        # print("1111111111")
        # exit()
        
        # print("mixed")
        
        return mixed_points, mixed_feat
        
        



class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True, 
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)  # 具有残差连接的一维多层感知机  在 SeedGenerator 中 in_channel=256 out_channel=128
        self.conv_key = nn.Conv1d(in_channel, dim, 1)   # in_channel：输入数据的通道数  dim：即经过卷积后的特征维度 1：卷积核大小
        # kenel = 1, (B, feature_dim, N)中的 N 不变，feature_dim 改变
        self.conv_query = nn.Conv1d(in_channel, dim, 1)  # 1*1 卷积升维或降维
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):   # upfeat 是上采样的 Seed Feature
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        
        Seed Generator: 
            
            pos: patch_xyz (B, 3, 128) 
            key: patch_feat (B, 256, 128)
            query: patch_feat (B, 256, 128)
        """
        value = self.mlp_v(torch.cat([key, query], 1)) # (B, dim, N) 得到  V，torch.cat([key, query]: (B, 512, 128)  value: (B, 256, 128)
        identity = value
        key = self.conv_key(key) # (B, dim, N)  转换 key 的特征维度到 dim 维，往前找 dim 设置的是 128
        query = self.conv_query(query) # 一维卷积转换 query 的特征维度到 dim 维，kernel_size = 1 保持点数量不变 (B, dim, N)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous() # (B, 128, 3)
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)   # (B, N , self.n_knn) N 个点的最近邻索引

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k) k表示最近邻的k个点 (6,64,128,20)
        # print("query: shape")
        # print((query.reshape((b, -1, n, 1))).shape)
        # exit()
        
        qk_rel = query.reshape((b, -1, n, 1)) - key   # 得到qk组合

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)  位置编码

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat) # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn) # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel) # 创建了一个 qk_rel 张量形状相同且所有元素为零的张量

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel) # (B, dim, N*up_factor, k)  默认参数 up_factor = 2

        # softmax function
        attention = self.scale(attention)    # 得到 attention 矩阵

        # knn value is correct
        # grouping_operation(value, idx_knn) 将 value 根据每个点的 knn 索引，扩展到 (B, dim, N, k) 
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)  获得 value

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor) 对应论文中的 Elemental-wise Product 和 Aggregation 把 k 维度聚合了 对 j 进行加权求和
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity
        

class UpLayer(nn.Module):
    """
    Upsample Layer with upsample transformers
    """
    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate='three', attn_channel=True):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])

        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, attn_channel=attn_channel, up_factor=self.up_factor)

        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim*2, hidden_dim=dim, out_dim=dim)

        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):   # pcd_prev 为 论文中的P0-P3
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, feat_dim, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_new: Tensor, upsampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # Collect seedfeature   得到的 feat_upsample 对应论文里的 Seed Features
        if self.interpolate == 'nearest':
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3) # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':  # 三线性插值 seed 特征
            # three interpolate
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
            weight = dist_recip / norm # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))
        

        # Query mlps
        feat_1 = self.mlp_1(pcd_prev)   # 提取特征 输入通道xyz，第一层输出64维度，第二层输出128维
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),   # 在 feat_1 第二个维度找最大值 将最大值特征沿着点的维度复制
                            feat_upsample], 1)   # 拼接后的特征维度为 128*2 + seed_dim
        Q = self.mlp_2(feat_1)  # 获得Q

        # Upsample Transformers
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample) # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample) # (B, 128, N_prev*up_factor)

        # Get current features K
        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        # New point cloud
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)  获得点位移
        pcd_new = self.upsample(pcd_prev)  # pcd_prev上采样 N_prev*up_factor 个点
        pcd_new = pcd_new + delta

        return pcd_new, K_curr


class SeedFormer(nn.Module):
    """
    SeedFormer Point Cloud Completion with Patch Seeds and Upsample Transformer
    """
    def __init__(self, feat_dim=512, embed_dim=128, num_p0=512, n_knn=20, radius=1, up_factors=None, seed_factor=2, interpolate='three', attn_channel=True):
        """
        Args:
            feat_dim: dimension of global feature
            embed_dim: dimension of embedding feature
            num_p0: number of P0 coarse point cloud
            up_factors: upsampling factors
            seed_factor: seed generation factor
            interpolate: interpolate seed features (nearest/three)
            attn_channel: transformer self-attention dimension (channel/point)
        """
        super(SeedFormer, self).__init__()
        self.num_p0 = num_p0

        # Seed Generator
        self.feat_extractor = FeatureExtractor(out_dim=feat_dim, n_knn=n_knn)
        self.feat_extractor_source = FeatureExtractor_source(out_dim=feat_dim, n_knn=n_knn)
        
        self.feat_extractor_force = FeatureExtractor_force(out_dim=feat_dim, n_knn=n_knn)
        self.kpfeat_extractor = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        self.seed_generator = SeedGenerator(feat_dim=feat_dim, seed_dim=embed_dim, n_knn=n_knn, factor=seed_factor, attn_channel=attn_channel)
        
        self.mix = patch_kp_mix()
        
        # Upsample layers
        up_layers = []
        for i, factor in enumerate(up_factors):
            up_layers.append(UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, n_knn=n_knn, radius=radius, 
                             interpolate=interpolate, attn_channel=attn_channel))
        self.up_layers = nn.ModuleList(up_layers)
        
        self.feat_global = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)
        self.feat_transformer = vTransformer(256, dim=64, n_knn=n_knn)

    def forward(self, partial_cloud, kp):
        """
        Args:
            partial_cloud: (B, N, 3) batch 点数量 xyz坐标
            kp (B, 128, 3) 64 个 kepoint xyz 坐标
        """
        # Encoder
        feat, mixed_xyz, mixed_feat = self.forward_encoder(partial_cloud, kp)
        # 输出：全局特征(B, 512, 1)  降采样点云坐标(B, 3, 128)  patch_xyz的局部特征(B, 256, 128) kp_feat
        
        
        # 经过 encoder 得到残缺点云的降采样与特征

        # Decoder
        pred_pcds = self.forward_decoder(feat, partial_cloud, mixed_xyz, mixed_feat)

        return pred_pcds

    def forward_encoder(self, partial_cloud, kp):  # 输入残缺点云
        # feature extraction
        partial_cloud = partial_cloud.permute(0, 2, 1).contiguous() 
        kp = kp.permute(0, 2, 1).contiguous() # (B, 3, 128)
        
        # method 0
        # feat, mixed_xyz, mixed_feat = self.feat_extractor_source(partial_cloud) # (B, feat_dim, 1)
        
        # method 1
        patch_xyz, patch_feat, kp_xyz, kp_feat = self.feat_extractor(partial_cloud, kp) # feat: (B, 512, 1) patch_xyz: (B, 3, 128) patch_feat: (B, 256, 128)
        mixed_xyz, mixed_feat = self.mix(patch_xyz, patch_feat, kp_xyz, kp_feat)    # patch + kp  (B, 3, 128) (B, 256, 128)
        mixed_feat = self.feat_transformer(mixed_feat, mixed_xyz)
        point_, feat = self.feat_global(mixed_xyz, mixed_feat)    # feat global (B, 512, 1)
        
        
        # method 3
         
        
        # method 2
        # feat, mixed_xyz, mixed_feat =  self.feat_extractor_force(partial_cloud, kp)  # 直接把kp加在上面
        
        

        return feat, mixed_xyz, mixed_feat # feat: (B, 512, 1) patch_xyz: (B, 3, 128) patch_feat: (B, 256, 128) kp: (B,3,24) kp_feat: (B,256,24)

    def forward_decoder(self, feat, partial_cloud, mixed_xyz, mixed_feat):
        """
        Args:
            feat: Tensor, (B, feat_dim, 1)    全局特征 1 个
            partial_cloud: Tensor, (B, N, 3)  输入的残缺点云
            patch_xyz: (B, 3, 128)            降采样的点云 128个点
            patch_feat: (B, seed_dim, 128)    patch_xyz的局部特征 128个点 特征 256 维
            kp_xyz: (B, 3, 64)
            kp_feat: (B, 256, 64)
        """
        pred_pcds = []

        # Generate Seeds
        seed, seed_feat = self.seed_generator(feat, mixed_xyz, mixed_feat)  # (B, 3, 256)256个seed点  (B, 128, 256)seed点128维特征 生成 seed 和 seed 特征
        seed = seed.permute(0, 2, 1).contiguous() # (B, num_pc, 3)
        pred_pcds.append(seed)

        # Upsample layers
        pcd = fps_subsample(torch.cat([seed, partial_cloud], 1), self.num_p0) # (B, num_p0, 3)  把输入的残缺点云和生成的seed放一起下采样到 num_p0
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous() # (B, 3, num_p0)
        seed = seed.permute(0, 2, 1).contiguous() # (B, 3, 256)
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, seed_feat, K_prev)  # 输入Patch Seed(F, S)  pcd 为论文中的 P0-P3
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())  # 向 pred_pcds 后面添加元素

        return pred_pcds


###########################
# Recommended Architectures
###########################

def seedformer_dim128(**kwargs):
    model = SeedFormer(feat_dim=512, embed_dim=128, n_knn=20, **kwargs)
    return model


if __name__ == '__main__':

    model = seedformer_dim128(up_factors=[1, 2, 2])
    model = model.cuda()
    print(model)

    x = torch.rand(8, 2048, 3)
    x = x.cuda()

    y = model(x)
    print([pc.size() for pc in y])


