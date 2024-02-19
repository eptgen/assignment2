import argparse
import os
import time

import dataset_location
import imageio
import losses
import numpy as np
from pytorch3d.ops import cubify, sample_points_from_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
import torch
from utils import get_device, get_mesh_renderer, load_cow_mesh

def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)
        
        color = [0.7, 0.7, 1]
        
        renderer = get_mesh_renderer(image_size=256)
        mesh1 = cubify(voxels_src, 0.5)
        mesh1_textures = torch.ones_like(mesh1.verts_list())
        mesh1_textures = mesh1_textures * color
        mesh1.textures = mesh1_textures
        mesh1 = mesh1.to(args.device)
        mesh2 = cubify(voxels_tgt, 0.5)
        mesh2_textures = torch.ones_like(mesh2.verts_list())
        mesh2_textures = mesh2_textures * color
        mesh2.textures = mesh2_textures
        mesh2 = mesh2.to(args.device)
        lights = PointLights(location=[[0, 0, -3]], device=args.device)
        
        num_povs = 15
        rends = []
        for i in range(num_povs):
            theta = 360 * (i / num_povs)
            R, T = look_at_view_transform(dist = 3., azim = theta)
            # Prepare the camera:
            cameras = FoVPerspectiveCameras(
                R=R, T=T, fov=60, device=args.device
            )

            rend1 = renderer(mesh1, cameras=cameras, lights=lights)
            rend1 = rend1.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            rends1.append((rend1 * 255).astype(np.uint8))

            rend2 = renderer(mesh2, cameras=cameras, lights=lights)
            rend2 = rend1.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            rends2.append((rend2 * 255).astype(np.uint8))
        imageio.mimsave("out/voxel_pred.gif", rends1, fps = 15, loop = 0)
        imageio.mimsave("out/voxel_gt.gif", rends2, fps = 15, loop = 0)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
