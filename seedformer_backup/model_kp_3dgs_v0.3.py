'''
==============================================================

key points generate(3D GS)


==============================================================

Author:
Date:
version: 0.3
==============================================================
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor
from torch.distributions import MultivariateNormal

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


# class KPN(nn.Module):
#     def __init__(self,k = 128):
#         super(KPN,self).__init__()

#         self.point_number = k
#         self.layer1 = conv1d(3,16)
#         self.layer2 = conv1d(16,64)
#         self.layer3 = conv1d(64,256)
#         self.layer4 = conv1d(256,1024)

#         self.ge_point1 = mlp(1024+64,512)
#         self.ge_point2 = mlp(512,256)
#         self.ge_point3 = mlp(256,self.point_number*3)

#         self.pp_point3 = mlp(256,self.point_number)
        


#     def forward(self,x):
#         ##
#         # x : [N,C,L]
#         # out : [N,L,C]
#         x_feat = self.layer2(self.layer1(x))   # (B, 64, N)
#         go_feat = self.layer4(self.layer3(x_feat)) # (B, 1024, N)
#         go_feat = torch.max(go_feat,2,keepdim=True)[0] # (B, 1024, 1)
#         _,C,L = x_feat.shape
#         go_feat_r = go_feat.repeat(1,1,L)   # (B, 1024, N)
        
#         kp_feat = torch.cat([x_feat,go_feat_r], dim = 1).permute(0,2,1) # (B, N, 1024+64)
#         kp_feat = self.ge_point1(kp_feat) # (B, N, 512)
#         kp_feat = F.dropout(kp_feat,p = 0.5)
#         kp_feat = self.ge_point2(kp_feat)  # (B, N, 256)
#         kp_feat = F.dropout(kp_feat,p = 0.2)
        
#         kp = self.ge_point3(kp_feat)  # (B, N, 128*3)
#         pp = self.pp_point3(kp_feat)

#         kp = torch.mean(kp,dim = 1)  # (B, 128*3)
#         kp = kp.reshape((-1,self.point_number,3))

#         pp = torch.mean(pp,dim = 1)
#         pp = pp.reshape((-1,self.point_number))
#         pp = F.sigmoid(pp)
        
#         return kp,pp


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

        # self.pp_point3 = mlp(256,self.point_number)
        
    
    def forward(self,partial):
        """
        Args:
            partial: (B, 3, N)
        """
        # encoder
        
        
        
        
        partial_256 = fps_subsample(partial.permute(0,2,1), 256) # (B, N, 3)
        # x_feat = self.layer2(self.layer1(partial))   # (B, 64, N)
        # go_feat = self.layer4(self.layer3(x_feat)) # (B, 1024, N)
        # go_feat = torch.max(go_feat,2,keepdim=True)[0] # (B, 1024, 1)
        # _,C,L = x_feat.shape
        # go_feat_r = go_feat.repeat(1,1,L)   # (B, 1024, N)
        
        # kp_feat = torch.cat([x_feat,go_feat_r], dim = 1).permute(0,2,1) # (B, N, 1024+64)
        
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
        cov_matrices += torch.eye(3, device=cov_matrices.device) * 1e-6
        # cov_matrices = torch.matmul(cov_matrices, cov_matrices.transpose(-2, -1))
        # cov_matrices = torch.linalg.cholesky(cov_matrices)
        
        # Sample
        mvn = MultivariateNormal(means, cov_matrices)
        sample_num = 100
        sample_points = mvn.sample(torch.Size([sample_num]))  # (sample_num, B, N, 3)
        sample_points = sample_points.permute(1,0,2,3)    # (B, sample_num, N, 3)
        sample_points = sample_points.reshape(B, N * sample_num, 3)  # (B, sample_num*N, 3)
        # sample
        # num_samples = 10
        # sample_points = torch.normal(means.repeat(1, 1, num_samples), stds.repeat(1, 1, num_samples))
        # kp = sample_points.reshape(B, -1, 3)  # (B, 640, 3)
        kp_all = torch.cat((partial_256, sample_points), dim=1)

        return kp_all, means
        


def chamfer_sqrt(p1, p2):
    chamfer_dist = chamfer_3DDist()
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

class ppro_cd_loss(nn.Module):
    def __init__(self):
        super(ppro_cd_loss,self).__init__()
        self.chamfer_dist = chamfer_3DDist()
        
    def forward(self,p1, p2, gt):
        ###  d1 : p1->p2
        ###  d2 : p2->p1
        ###  p1 : [B,L1,C]
        ###  p2 : [B,L2,C]
        ###  d1 : [B,L1]
        ###  d2 : [B,L2]
        CD = chamfer_sqrt
        
        B1,N1,_ = p1.shape
        B2,N2,_ = p2.shape

        gt1 = fps_subsample(gt,N1)
        gt2 = fps_subsample(gt,N2)   # means
        
        cd1 = CD(p1, gt1)
        cd2 = CD(p2, gt2)
        return (cd1+cd2)*1e3, cd1*1e3 , cd2*1e3
 


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