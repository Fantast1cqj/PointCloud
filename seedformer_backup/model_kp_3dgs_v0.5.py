'''
==============================================================

key points generate(3D GS)


==============================================================

Author:
Date:
version: 0.5
note: 裁剪限制采样范围    先训均值再一起训有问题，训练完均值再采样裁切会导致矩阵不正定     同时训正常效果可以
      相比 v0.4 多了裁剪限制两倍方差，两个 loss 应该同时训，不能先训均值
      
      
      
result: train_kp3dgs_shapenet55_Log_2024_11_14_02_41_29   error
        train_kp3dgs_shapenet55_Log_2024_11_13_08_17_02   normal
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


# class Sample_with_clipping(nn.Module):
#     def __init__(self, num_samples):
#         super(Sample_with_clipping, self).__init__()
#         self.num_samples = num_samples
        
#     def forward(self, means, cov_matrices):
#         """
#         Args:
#             means: 均值 (B, N, 3)
#             cov_matrices: 协方差矩阵 (B, N, 3, 3)
#             num_samples: 采样数量
#             lower_bound: 每个坐标的最小值
#             upper_bound: 每个坐标的最大值

#         Returns:
#             sample_points: 裁剪后的样本 (B, N, num_samples, 3)
#         """
#         mvn = MultivariateNormal(means, cov_matrices)
#         sample_points = mvn.sample(torch.Size([self.num_samples])) 
        
        
#         stddev = torch.sqrt(torch.diagonal(cov_matrices, dim1=-2, dim2=-1))
#         lower_bound = means -  1.7 * stddev  # (B, N, 3)
#         upper_bound = means +  1.7 * stddev  # (B, N, 3)
#         samples_clip = sample_points.clone()  # 防止在原张量上做 inplace 修改
#         for i in range(3):  # 3 对应 x, y, z 轴
#             samples_clip[..., i] = torch.clamp(sample_points[..., i], min=lower_bound[..., i], max=upper_bound[..., i]) # torch.Size([60, 112, 64])
        
        
#         samples_clip = samples_clip.permute(1,2,0,3)
        
#         return samples_clip





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



# generate 
class KP_3DGS(nn.Module):
    def __init__(self,k = 64):
        super(KP_3DGS,self).__init__()
        self.point_number = k # 64
        
        self.encoder = Encoder()
        
        self.layer1 = conv1d(3,16)
        self.layer2 = conv1d(16,64)
        self.layer3 = conv1d(64,256)
        self.layer4 = conv1d(256,1024)

        self.ge_point1 = mlp(1024+512,512)
        self.ge_point2 = mlp(512,256)
        self.ge_point3 = mlp(256,self.point_number*9)
        # self.sample = Sample_with_clipping(num_samples = 30)

        # self.pp_point3 = mlp(256,self.point_number)
        
    
    def forward(self,partial):
        """
        Args:
            partial: (B, 3, N)
        """
        # encoder
        partial_, partial_feat = self.encoder(partial)
        partial_feat = partial_feat.permute(0,2,1) # (B, 512, 1024+512)
        
        kp_feat = self.ge_point1(partial_feat) # (B, N, 512)
        kp_feat = F.dropout(kp_feat,p = 0.05)
        kp_feat = self.ge_point2(kp_feat)  # (B, N, 256)
        kp_feat = F.dropout(kp_feat,p = 0.02)
        
        kp_gs = self.ge_point3(kp_feat)  # (B, N, 64*9)
        # pp = self.pp_point3(kp_feat)
        kp_gs = torch.mean(kp_gs,dim = 1)  # (B, 64*9)
        kp_gs = kp_gs.reshape((-1,self.point_number,9)) # (B, 64, 9)   x y z a b c ab ac bc
        
        B, N, _ = kp_gs.shape
        means = kp_gs[..., :3]  # (B, 64, 3)
        cov_elements = kp_gs[..., 3:]  # (B, 64, 6)

        
        # make sure > 0
        device = cov_elements.device
        cov_elements = torch.abs(cov_elements)
        cov_elements = torch.where(cov_elements == 0, torch.tensor(1e-6,device=device), cov_elements)
        
        
        cov_matrices = torch.zeros(B, N, 3, 3, device=cov_elements.device)
        # fill diagonal
        cov_matrices[..., 0, 0] = cov_elements[..., 0]
        cov_matrices[..., 1, 1] = cov_elements[..., 1]
        cov_matrices[..., 2, 2] = cov_elements[..., 2]
        # fill under diagonal
        cov_matrices[..., 0, 1] = 0.0
        cov_matrices[..., 1, 0] = cov_elements[..., 3]
        cov_matrices[..., 0, 2] = 0.0
        cov_matrices[..., 2, 0] = cov_elements[..., 4]
        cov_matrices[..., 1, 2] = 0.0
        cov_matrices[..., 2, 1] = cov_elements[..., 5]
        
        
        # 保证正定：
        cov_matrices = cov_matrices @ cov_matrices.transpose(-2, -1)
        cov_matrices = cov_matrices + torch.eye(3, device=cov_matrices.device) * 1e-6
        
        # Sample
        sample_num = 30
        sample_points = sample_with_clipping(means, cov_matrices, sample_num) # (B, N, sample_num, 3)
        
        
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

        cd1 = CD(means, gt)
        

        return cd1*1e3 + cd2*1e3, cd1*1e3 , cd2*1e3


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