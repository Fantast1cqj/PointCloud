'''
==============================================================

key points generate(3D GS)


==============================================================

Author: Fan Quanjiang
Date: 2024.11.19
version: 0.70
note: 分开预测均值、方差、四元数；先 freeze cov 层，训练 means 层，再 freeze means 层，训练cov层。解决 v0.61 无法先训 means 再一起训 

next: 加一个loss, 所有点和gt的; decoder时候降采样一次加一次全局特征, cat两次; 用 so3 预测旋转矩阵
==============================================================
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor
from torch.distributions import MultivariateNormal
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
    
def sample_with_clipping(means, cov_matrices, num_samples):
    """
    Args:
        means: 均值 (B, N, 3)
        cov_matrices: 协方差矩阵 (B, N, 3, 3)
        num_samples: 采样数量
        lower_bound: 每个坐标的最小值
        upper_bound: 每个坐标的最大值

    Returns:
        sample_points: 裁剪后的样本 (B, N, num_samples, 3)
    """
    # print("cov_matrices: ", cov_matrices)
    stddev = torch.sqrt(torch.diagonal(cov_matrices, dim1=-2, dim2=-1))
    # print("stddev: ", stddev)
    
    mvn = MultivariateNormal(means, cov_matrices)
    samples = mvn.sample(torch.Size([num_samples]))   # (num_samples, B, N, 3)
    
    lower_bound = means -  2 * stddev  # (B, N, 3)
    upper_bound = means +  2 * stddev  # (B, N, 3)
    
    samples_clipped = samples.clone() 
    for i in range(3):  # 3是对应 x, y, z 轴
        # samples[..., i] = torch.clamp(samples[..., i], min=lower_bound[..., i], max=upper_bound[..., i])
        samples_clipped[..., i] = torch.max(torch.min(samples[..., i], upper_bound[..., i]), lower_bound[..., i])
    
    # print(samples)
    
    samples_clipped = samples_clipped.permute(1,2,0,3)
    # B, sample_num, N, _ = samples.shape
    # samples = samples.reshape(B, N * sample_num, 3)
    
    return samples_clipped


def normalize_quaternions(q):
    """
    对四元数张量进行归一化.
    
    input:
    q (torch.Tensor): 输入张量，形状为 (B, N, 4)
    
    output:
    torch.Tensor: 归一化后的四元数张量，形状为 (B, N, 4)
    """
    # 计算四元数的模长，模长的形状是 (B, N, 1)
    norm = torch.norm(q, dim=-1, keepdim=True)
    
    # 将四元数除以它的模长进行归一化
    q_normalized = q / norm
    
    return q_normalized


def quaternion_2_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵.
    
    参数:
    q (torch.Tensor): 四元数张量，形状为 (B, N, 4)
    
    返回:
    torch.Tensor: 旋转矩阵张量，形状为 (B, N, 3, 3)
    """
    # 提取四元数的各个分量
    q0 = q[:, :, 0]  # 标量部分
    q1 = q[:, :, 1]  # x 轴分量
    q2 = q[:, :, 2]  # y 轴分量
    q3 = q[:, :, 3]  # z 轴分量

    # 计算旋转矩阵
    R = torch.zeros(q.shape[0], q.shape[1], 3, 3, device=q.device)
    
    # 按照公式计算旋转矩阵
    R[:, :, 0, 0] = 1 - 2 * (q2**2 + q3**2)
    R[:, :, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, :, 0, 2] = 2 * (q1 * q3 + q0 * q2)
    
    R[:, :, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, :, 1, 1] = 1 - 2 * (q1**2 + q3**2)
    R[:, :, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    
    R[:, :, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, :, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, :, 2, 2] = 1 - 2 * (q1**2 + q2**2)
    
    return R


class conv1d(nn.Module):
    def __init__(self,in_c,out_c):
        super(conv1d,self).__init__()
        self.inc = in_c
        self.outc = out_c
        self.conv1 = nn.Conv1d(self.inc,self.outc,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm1d(self.outc)
        self.relu = nn.PReLU()
    
    def forward(self,x):
        return self.relu(self.bn(self.conv1(x)))

class mlp(nn.Module):
    def __init__(self,in_c,out_c):
        super(mlp,self).__init__()
        self.inc=in_c
        self.outc=out_c
        self.mlp = nn.Linear(self.inc,self.outc)
        self.relu = nn.PReLU()

    def forward(self,x):
        return self.relu(self.mlp(x))







class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder_SA_1 = PointNet_SA_Module_KNN(1024, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.encoder_SA_2 = PointNet_SA_Module_KNN(512, 16, 128, [256, 512], group_all=False, if_bn=False, if_idx=True)
        self.encoder_SA_3 = PointNet_SA_Module_KNN(None, None, 512, [512, 1024], group_all=True, if_bn=False)
        
        self.encoder_transformer_1 = vTransformer(128, dim=64, n_knn=16)
        self.encoder_transformer_2 = vTransformer(512, dim=64, n_knn=16)
        # self.encoder_transformer_3 = vTransformer(1024, dim=64, n_knn=8)
    
        
        
    def forward(self,partial):
        """
        Args:
            partial: (B, 3, N)  N = 2048
        """
        l0_xyz = partial.contiguous()
        l0_points = partial.contiguous()
        
        l1_xyz, l1_points, idx1 = self.encoder_SA_1(l0_xyz, l0_points)  # (B, 3, 1024)  (B, 128, 1024)
        l1_points = self.encoder_transformer_1(l1_points, l1_xyz)   # (B, 128, 1024)
        
        l2_xyz, l2_points, idx2 = self.encoder_SA_2(l1_xyz, l1_points)  # (B, 3, 512)  (B, 512, 512)
        l2_points = self.encoder_transformer_2(l2_points, l2_xyz)   # (B, 512, 512)
        
        l3_xyz, l3_points = self.encoder_SA_3(l2_xyz, l2_points)  # l3_points: (B, 1024, 1)
        
        _,_,N = l2_xyz.shape
        feat_all_re = l3_points.repeat(1, 1, N)   #  (B, 1024, 512)
        partial512_feat = torch.cat([l2_points, feat_all_re], dim = 1) # (B, 1024+512, 512)
        return l2_xyz, partial512_feat



# generate kp
class KP_3DGS(nn.Module):
    def __init__(self,k = 64):
        super(KP_3DGS,self).__init__()
        self.point_number = k # 64
        
        self.encoder_means = Encoder()
        self.encoder_covs = Encoder()
        # self.layer1 = conv1d(3,16)
        # self.layer2 = conv1d(16,64)
        # self.layer3 = conv1d(64,256)
        # self.layer4 = conv1d(256,1024)

        self.get_means1 = mlp(1024+512,512)
        self.get_means2 = mlp(512,256)
        self.get_means3 = mlp(256,self.point_number*3)
        
        self.get_cov1 = mlp(1024+512,512)
        self.get_cov2 = mlp(512,256)
        self.get_cov3 = mlp(256,self.point_number*7)
        # self.sample = Sample_with_clipping(num_samples = 30)

    def freeze_cov_layers(self):
        """Freeze the cov-related layers."""
        for param in self.encoder_covs.parameters():
            param.requires_grad = False
        for param in self.get_cov1.parameters():
            param.requires_grad = False
        for param in self.get_cov2.parameters():
            param.requires_grad = False
        for param in self.get_cov3.parameters():
            param.requires_grad = False
    
    def unfreeze_cov_layers(self):
        """Unfreeze the cov-related layers."""
        for param in self.encoder_covs.parameters():
            param.requires_grad = True
        for param in self.get_cov1.parameters():
            param.requires_grad = True
        for param in self.get_cov2.parameters():
            param.requires_grad = True
        for param in self.get_cov3.parameters():
            param.requires_grad = True
    
    def freeze_means_layers(self):
        """Freeze the cov-related layers."""
        for param in self.encoder_means.parameters():
            param.requires_grad = False
        for param in self.get_means1.parameters():
            param.requires_grad = False
        for param in self.get_means2.parameters():
            param.requires_grad = False
        for param in self.get_means3.parameters():
            param.requires_grad = False
    
    

    def forward(self,partial):
        """
        Args:
            partial: (B, 3, N)
        """
        # encoder
        partial_means, partial_feat_means = self.encoder_means(partial)
        partial_covs, partial_feat_covs = self.encoder_covs(partial)  # (B, 1024+512, 512)
        # decoder
        partial_feat_means = partial_feat_means.permute(0,2,1) # (B, 512, 1024+512)
        means_feat = self.get_means1(partial_feat_means) # (B, N, 512)
        means_feat = F.dropout(means_feat,p = 0.2)
        means_feat = self.get_means2(means_feat)  # (B, N, 256)
        means_feat = F.dropout(means_feat,p = 0.3)
        means = self.get_means3(means_feat)  # (B, N, 64*3)   mean_x mean_y mean_z σ1 σ2 σ3 q0 q1 q2 q3 q4
        means = torch.mean(means,dim = 1)  # (B, 64*3)
        means = means.reshape((-1, self.point_number, 3)) # (B, 64, 3)   mean_x mean_y mean_z σ1 σ2 σ3 q0 q1 q2 q3 q4
        
        
        partial_feat_covs = partial_feat_covs.permute(0,2,1) # (B, 512, 1024+512)
        cov_feat = self.get_cov1(partial_feat_covs) # (B, N, 512)
        cov_feat = F.dropout(cov_feat,p = 0.05)
        cov_feat = self.get_cov2(cov_feat)  # (B, N, 256)
        cov_feat = F.dropout(cov_feat,p = 0.02)
        cov_feat = self.get_cov3(cov_feat)  # (B, N, 64*7)   σ1 σ2 σ3 q0 q1 q2 q3 q4
        cov_feat = torch.mean(cov_feat,dim = 1)  # (B, 64*7)
        var_q = cov_feat.reshape((-1, self.point_number, 7)) # (B, 64, 7)   σ1 σ2 σ3 q0 q1 q2 q3 q4
        
        # get cov_mat
        B, N, _ = means.shape
        
        var_element = var_q[:, :, :3] # (B, 64, 3) σ1 σ2 σ3
        epsilon = 1e-5
        var_element = torch.abs(var_element) + epsilon
        
        q = var_q[:, :, 3:]  # (B, 64, 4)  q0 q1 q2 q3 q4
        
        q_norm = normalize_quaternions(q) # (B, 64, 4)
        R = quaternion_2_rotation_matrix(q_norm)  # (B, N, 3, 3)
        
        var_mat = torch.zeros(B, N, 3, 3, device = var_element.device)
        # for b in range(B):
        #     for n in range(N):
        #         # 取方差向量并将其填充到对角线上
        #         diag = torch.diag(var_element[b, n])
        #         var_mat[b, n] = diag
        
        var_mat = torch.diag_embed(var_element)
        
        # var_mat_squared = torch.matmul(var_mat, var_mat)  # 方差矩阵平方
        cov_mat = torch.matmul(R, torch.matmul(var_mat, R.transpose(-2, -1)))  # (B, N, 3, 3)
        
        # sample
        sample_num = 30
        sample_points = sample_with_clipping(means, cov_mat, sample_num)  # # (B, N, sample_num, 3)
        

        # test
        # sample_points_re = sample_points.reshape(B, -1, 3)  # (B, 640, 3)
        # kp_3dgs = sample_points_re[0].unsqueeze(0) # (1,640,3)
        
        # kp_cut = np.squeeze(kp_3dgs)   # 去掉一个维度
        # tensor_cpu = kp_cut.cpu()      # 转换为 cpu 张量
        # kp_cpu = tensor_cpu.detach().numpy()   # 张量转 numpy
        # file_name = f'kp3dgs.npy'
        # base_path = '../test_kp/test_kp3dgs_cloud'
        # file_path = os.path.join(base_path, file_name)
        # np.save(file_path, kp_cpu)     # 保存所有
        # exit()
        
    

        return means, sample_points
        


def chamfer_sqrt(p1, p2):
    chamfer_dist = chamfer_3DDist()
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

class kp_3dgs_loss(nn.Module):
    def __init__(self):
        super(kp_3dgs_loss,self).__init__()
        self.chamfer_dist = chamfer_3DDist()
        
    def forward(self, means, sample_points, gt):
        """
        Args:
            means: (B, N, 3)
            sample_points: (B, N, sample_num, 3)
            gt: (B, 8192, 3)
        """
        B, N, sample_num, _ = sample_points.shape
        gt = gt.contiguous()
        CD = chamfer_sqrt
        k = 200
        idx = query_knn(k,  gt, means, include_self = False)  # (B, N, k)
        
        grouped_xyz = grouping_operation(gt.permute(0,2,1).contiguous(), idx)   # (B, 3, N, k)
        grouped_xyz = grouped_xyz.permute(0,2,3,1)  # (B, N, k, 3)
        
        sample_points_list = sample_points.unbind(dim=1) # N * (B, sample_num, 3)
        gt_list = grouped_xyz.unbind(dim=1) # N * (B, k, 3)
        
        cd2_list = [CD(sample_, gt_)
                     for sample_, gt_ in zip(sample_points_list, gt_list)]
        

        cd2 = torch.sum(torch.stack(cd2_list))

        
        
        B1,N1,_ = means.shape
        # gt1 = fps_subsample(gt,N1)
        cd1 = CD(means, gt)
        

        # return cd1*1e3 + cd2*1e3, cd1*1e3 , cd2*1e3
        return cd2*1e3, cd1*1e3 , cd2*1e3


class means_cd_loss(nn.Module):
    def __init__(self):
        super(means_cd_loss,self).__init__()
        self.chamfer_dist = chamfer_3DDist()
        
    def forward(self, means, gt):
        """
        Args:
            means: (B, N, 3)
            gt: (B, M, 3)
        """
        CD = chamfer_sqrt
        B1,N1,_ = means.shape
        # gt1 = fps_subsample(gt,N1)
        cd = CD(means, gt)

        return cd*1e3


def KP_3DGS_640():
    model = KP_3DGS(k = 64)
    return model

if __name__=="__main__":
    #x = torch.randn((6,12,3)).cuda()
    #y = torch.randn((6,1024,3)).cuda()
    #ppro_cd_loss()(x,y)
    x = torch.randn((48,3,128))
    net = KP_3DGS(k=64)
    y = net(x)
    #y,yp = net(x)