
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg,
                 image_width, image,
                 MV_images,
                 MV_image_path,
                 image_height, image_path,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, landmark2d=np.zeros((1, 68)), lip_crop=None, data_device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T

        self.MV_image_path = MV_image_path        # must be dict{path}
        self.MV_images = MV_images
        if landmark2d is not None:
            if isinstance(landmark2d, np.ndarray):
                self.landmark2d = torch.tensor(landmark2d, dtype=torch.float32)
            elif isinstance(landmark2d, list):
                # only be used in teacher
                self.landmark2d = landmark2d
            elif isinstance(landmark2d, torch.Tensor):
                self.landmark2d = landmark2d
        else:
            self.landmark2d = landmark2d

        if lip_crop is not None:
            self.lip_crop = torch.tensor(lip_crop, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        else:
            self.lip_crop = None

        self.R_tensor = torch.tensor(R, dtype=torch.float32)
        self.T_tensor = torch.tensor(T, dtype=torch.float32)

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear,
                                                     zfar=self.zfar,
                                                     fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep
