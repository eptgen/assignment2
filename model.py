from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch3d.utils import checkerboard, ico_sphere, torus
import pytorch3d

def get_shape(shape, device):
    if shape == "torus1":
        return torus(0.5, 1, 42, 61, device = device)
    if shape == "torus2":
        return torus(0.5, 1, 61, 42, device = device)
    if shape == "checkerboard":
        return checkerboard(25).to(device)
    return ico_sphere(4, device)
    
class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        b = args.batch_size
        
        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            
            self.conv0 = nn.ConvTranspose3d(512, 256, 4, 2, 1) # b x (256, 2 x 2 x 2)
            self.batch0 = nn.BatchNorm3d(256)
            self.conv1 = nn.ConvTranspose3d(256, 128, 4, 2, 1) # b x (128, 4 x 4 x 4)
            self.batch1 = nn.BatchNorm3d(128)
            self.conv2 = nn.ConvTranspose3d(128, 64, 4, 2, 1) # b x (64, 8 x 8 x 8)
            self.batch2 = nn.BatchNorm3d(64)
            self.conv3 = nn.ConvTranspose3d(64, 32, 4, 2, 1) # b x (32, 16 x 16 x 16)
            self.batch3 = nn.BatchNorm3d(32)
            self.conv4 = nn.ConvTranspose3d(32, 8, 4, 2, 1) # b x (8, 32 x 32 x 32)
            self.batch4 = nn.BatchNorm3d(8)
            self.conv5 = nn.ConvTranspose3d(8, 1, 1) # b x (1, 32 x 32 x 32)
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 2048)
            self.fc3 = nn.Linear(2048, 3 * self.n_point)
            self.gelu1 = nn.GELU()
            self.gelu2 = nn.GELU()
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = get_shape(args.start_shape, self.device)
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 2048)
            self.n_vertices = mesh_pred.verts_packed().shape[0]
            self.fc3 = nn.Linear(2048, self.n_vertices * 3)
            self.gelu1 = nn.GELU()
            self.gelu2 = nn.GELU()
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =             

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # print(encoded_feat.shape)
            voxels_pred = torch.reshape(encoded_feat, (B, 512, 1, 1, 1))
            voxels_pred = F.relu(self.batch0(self.conv0(voxels_pred)))
            voxels_pred = F.relu(self.batch1(self.conv1(voxels_pred)))
            voxels_pred = F.relu(self.batch2(self.conv2(voxels_pred)))
            voxels_pred = F.relu(self.batch3(self.conv3(voxels_pred)))
            voxels_pred = F.relu(self.batch4(self.conv4(voxels_pred)))
            voxels_pred = torch.sigmoid(self.conv5(voxels_pred))
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.gelu1(self.fc1(encoded_feat))
            pointclouds_pred = self.gelu2(self.fc2(pointclouds_pred))
            pointclouds_pred = self.fc3(pointclouds_pred)
            pointclouds_pred = torch.reshape(pointclouds_pred, (B, args.n_points, 3))
            # print(pointclouds_pred.shape)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =
            deform_vertices_pred = self.gelu1(self.fc1(encoded_feat))
            deform_vertices_pred = self.gelu2(self.fc2(deform_vertices_pred))
            deform_vertices_pred = self.fc3(deform_vertices_pred)
            deform_vertices_pred = torch.reshape(deform_vertices_pred, (B, self.n_vertices, 3))
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return mesh_pred

