'''
==============================================================

key points model


==============================================================

Author:
Date:

==============================================================
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist


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


class KPN(nn.Module):
    def __init__(self,k = 128):
        super(KPN,self).__init__()

        self.point_number = k
        self.layer1 = conv1d(3,16)
        self.layer2 = conv1d(16,64)
        self.layer3 = conv1d(64,256)
        self.layer4 = conv1d(256,1024)

        self.ge_point1 = mlp(1024+64,512)
        self.ge_point2 = mlp(512,256)
        self.ge_point3 = mlp(256,self.point_number*3)

        self.pp_point3 = mlp(256,self.point_number)
        


    def forward(self,x):
        ##
        # x : [N,C,L]
        # out : [N,L,C]
        x_feat = self.layer2(self.layer1(x))
        go_feat = self.layer4(self.layer3(x_feat))
        go_feat = torch.max(go_feat,2,keepdim=True)[0]
        _,C,L = x_feat.shape
        go_feat_r = go_feat.repeat(1,1,L)
        
        kp_feat = torch.cat([x_feat,go_feat_r], dim = 1).permute(0,2,1)
        kp_feat = self.ge_point1(kp_feat)
        kp_feat = F.dropout(kp_feat,p = 0.5)
        kp_feat = self.ge_point2(kp_feat)
        kp_feat = F.dropout(kp_feat,p = 0.2)
        
        kp = self.ge_point3(kp_feat)
        pp = self.pp_point3(kp_feat)

        kp = torch.mean(kp,dim = 1)
        kp = kp.reshape((-1,self.point_number,3))

        pp = torch.mean(pp,dim = 1)
        pp = pp.reshape((-1,self.point_number))
        pp = F.sigmoid(pp)
        
        return kp,pp



class ppro_cd_loss(nn.Module):
    def __init__(self):
        super(ppro_cd_loss,self).__init__()
        self.chamfer_dist = chamfer_3DDist()
    
    def forward(self,p1,p2):
        ###  d1 : p1->p2
        ###  d2 : p2->p1
        ###  p1 : [B,L1,C]
        ###  p2 : [B,L2,C]
        ###  d1 : [B,L1]
        ###  d2 : [B,L2]
        d1,d2,_,_ = self.chamfer_dist(p1,p2)
        return torch.mean(d1)+torch.mean(d2)
 
def kp_24():
    model = KPN(k = 24)
    return model

def kp_128():
    model = KPN(k = 128)
    return model

if __name__=="__main__":
    #x = torch.randn((6,12,3)).cuda()
    #y = torch.randn((6,1024,3)).cuda()
    #ppro_cd_loss()(x,y)
    x = torch.randn((48,3,128))
    net = KPN(k=12)
    y = net(x)
    #y,yp = net(x)