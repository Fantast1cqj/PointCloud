# 测试 knn 能不能用
import torch
from models.utils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor
import open3d as o3d
import numpy as np
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
def chamfer_sqrt(p1, p2):
    chamfer_dist = chamfer_3DDist()
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


CD = chamfer_sqrt
filename = f'./knn_test/plane.npy'
point_cloud_data = np.load(filename)
tensor_data = torch.from_numpy(point_cloud_data)
tensor_data = tensor_data.to(torch.float32)
tensor_data = tensor_data.unsqueeze(0) # (1,8192,3)
tensor_data = tensor_data.to('cuda:0')


means = fps_subsample(tensor_data, 64) # (1,64,3)

idx = query_knn(100,  tensor_data, means, include_self = True)
grouped_xyz = grouping_operation(tensor_data.permute(0,2,1).contiguous(), idx)
grouped_xyz = grouped_xyz.permute(0,2,3,1)  # (B, N, k, 3)
reshaped_tensor = grouped_xyz.reshape(grouped_xyz.size(0), -1, 3)

cd = CD(means,tensor_data)
print(cd*1e3)

# 测试可视化输出 means
# kp_cut = np.squeeze(reshaped_tensor)   # 去掉一个维度
# tensor_cpu = kp_cut.cpu()      # 转换为 cpu 张量
# kp_cpu_np = tensor_cpu.numpy()    # 张量转 numpy
# print(kp_cpu_np.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(kp_cpu_np)
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Point Cloud Visualization", width=800, height=600, visible=True)
# vis.add_geometry(pcd)
# vis.run()





# filename = f'./test_kp3dgs_cloud/kp_30.npy'
# point_cloud_data = np.load(filename)
# print(point_cloud_data.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
# # path = f'./png_p4/{name}{i}{png}'
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Point Cloud Visualization", width=800, height=600, visible=True)

# vis.add_geometry(pcd)
# vis.run()