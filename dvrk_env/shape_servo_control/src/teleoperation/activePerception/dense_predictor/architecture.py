import torch.nn as nn
import torch
import torch.nn.functional as F
# from pointnet2_utils_groupnorm import PointNetSetAbstraction,PointNetFeaturePropagation
from pointconv_util_groupnorm_2 import PointConvDensitySetAbstraction,PointConvFeaturePropagation

class DensePredictor(nn.Module):
    
    """ 
    Archiecture of the dense predictor. Segment the query point cloud into classes.
    """
    
    def __init__(self, num_classes, normal_channel=False):
        super(DensePredictor, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)
        
        self.fp3 = PointConvFeaturePropagation(in_channel=128+256, mlp=[128], bandwidth = 0.4, linear_shape=128+3)
        self.fp2 = PointConvFeaturePropagation(in_channel=64+128, mlp=[64], bandwidth = 0.2, linear_shape=64+3)
        self.fp1 = PointConvFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64], bandwidth = 0.1, linear_shape=3)

        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.GroupNorm(1, 64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)


    def forward(self, xyz, xyz_query):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(f"l1_xyz, points: {l1_xyz.shape}, {l1_points.shape}")
        # print(f"l2_xyz, points: {l2_xyz.shape}, {l2_points.shape}")
        # print(f"l3_xyz, points: {l3_xyz.shape}, {l3_points.shape}")

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # print(f"l2_points fp: {l2_points.shape}")
        # print(f"l1_points fp: {l1_points.shape}")
        # print(f"l0_points fp: {l0_points.shape}")


        if self.normal_channel:
            l0_points_g = xyz_query
            l0_xyz_g = xyz_query[:,:3,:]
        else:
            l0_points_g = xyz_query
            l0_xyz_g = xyz_query
        l1_xyz_g, l1_points_g = self.sa1(l0_xyz_g, l0_points_g)
        l2_xyz_g, l2_points_g = self.sa2(l1_xyz_g, l1_points_g)
        l3_xyz_g, l3_points_g = self.sa3(l2_xyz_g, l2_points_g) 

        # print(f"g l1_xyz, points: {l1_xyz_g.shape}, {l1_points_g.shape}")
        # print(f"g l2_xyz, points: {l2_xyz_g.shape}, {l2_points_g.shape}")
        # print(f"g l3_xyz, points: {l3_xyz_g.shape}, {l3_points_g.shape}")

        
        l2_points_g = self.fp3(l2_xyz_g, l3_xyz_g, l2_points_g, l3_points_g)
        l1_points_g = self.fp2(l1_xyz_g, l2_xyz_g, l1_points_g, l2_points_g)
        l0_points_g = self.fp1(l0_xyz_g, l1_xyz_g, None, l1_points_g)

        # print(f"g l2_points fp: {l2_points_g.shape}")
        # print(f"g l1_points fp: {l1_points_g.shape}")
        # print(f"g l0_points fp: {l0_points_g.shape}")


        x = torch.cat([l0_points, l0_points_g], 1)
        #print("concat features x shape: ", x.shape)
        
        
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(x)))
        #print(f"feat after conv 1: {feat.shape}")
        x = self.drop1(feat)
        x = self.conv2(x)
        #print(f"feat after conv 2: {x.shape}")
        x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x #(batch, 2, num_query_points)



class SimpleDensePredictor(nn.Module):
    
    """ 
    Archiecture of the dense predictor. Predict attachment point using segmentation network. Does not take in query point cloud.
    """
    
    def __init__(self, num_classes, normal_channel=False):
        super(SimpleDensePredictor, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)
        
        self.fp3 = PointConvFeaturePropagation(in_channel=128+256, mlp=[128], bandwidth = 0.4, linear_shape=128+3)
        self.fp2 = PointConvFeaturePropagation(in_channel=64+128, mlp=[64], bandwidth = 0.2, linear_shape=64+3)
        self.fp1 = PointConvFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64], bandwidth = 0.1, linear_shape=3)

        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.GroupNorm(1, 32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, num_classes, 1)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = l0_points
        
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x #(batch, num_classes, num_points)


if __name__ == '__main__':

    num_classes = 2
    num_points = 1024
    point_dim=3
    batch = 8
    device = torch.device("cuda") # cuda
    pc = torch.randn((batch,point_dim,num_points)).float().to(device)
    pc_goal = torch.randn((batch,point_dim,num_points)).float().to(device)
    model = DensePredictor(num_classes).to(device)
    out = model(pc, pc_goal)
    
    # m = nn.LogSoftmax(dim=1)
    # input = torch.randn(2, 3)
    # output = m(input)

    # print(output)
    # print(torch.sum(torch.exp(output), dim=1))
    ########## output shape of each layer:
    # encode point cloud:
    # l1_xyz, points: torch.Size([8, 3, 512]), torch.Size([8, 64, 512])
    # l2_xyz, points: torch.Size([8, 3, 128]), torch.Size([8, 128, 128])
    # l3_xyz, points: torch.Size([8, 3, 1]), torch.Size([8, 256, 1])
    # l2_points fp: torch.Size([8, 128, 128])
    # l1_points fp: torch.Size([8, 64, 512])
    # l0_points fp: torch.Size([8, 64, 1024])

    # encode query point cloud:
    # g l1_xyz, points: torch.Size([8, 3, 512]), torch.Size([8, 64, 512])
    # g l2_xyz, points: torch.Size([8, 3, 128]), torch.Size([8, 128, 128])
    # g l3_xyz, points: torch.Size([8, 3, 1]), torch.Size([8, 256, 1])
    # g l2_points fp: torch.Size([8, 128, 128])
    # g l1_points fp: torch.Size([8, 64, 512])
    # g l0_points fp: torch.Size([8, 64, 1024])

    # after concatenation:
    # concat features x shape:  torch.Size([8, 128, 1024])
    # feat after conv 1: torch.Size([8, 64, 1024])
    # feat after conv 2: torch.Size([8, 2, 1024])
