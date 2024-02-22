
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops.knn import knn_points
import torch

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
    loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(voxel_src), voxel_tgt)
    return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
    nearest_1, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K = 1)
    nearest_2, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K = 1)
    loss_chamfer = torch.sum(nearest_1) + torch.sum(nearest_2)
    return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
    loss_laplacian = mesh_laplacian_smoothing(mesh_src)
    return loss_laplacian
