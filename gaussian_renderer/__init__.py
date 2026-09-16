import sys

import torch
import math
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
import pickle

def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel],
           pipe, bg_color: torch.Tensor, device=torch.device('cuda:0'),
           scaling_modifier=1.0, override_color=None, deta_attr=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    binding = pc.binding_ori

    if deta_attr.shape[1] == 1:
        change_xyz = None
        change_scales = None
        change_opacity = deta_attr[:, :]
        change_rotation = None
    else:

        change_xyz = deta_attr[:, :3]
        change_scales = deta_attr[:, 8:]
        change_opacity = deta_attr[:, 3:4]
        change_rotation = deta_attr[:, 4:8]

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.to(device),
        projmatrix=viewpoint_camera.full_proj_transform.to(device),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.to(device),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if deta_attr is not None and change_xyz is not None:
        xyz_ = pc.get_xyz
        xyz_clone = xyz_.clone()
        xyz_clone[binding] = xyz_clone[binding] + change_xyz
        means3D = xyz_clone
    else:
        means3D = pc.get_xyz

    means2D = screenspace_points

    if deta_attr is not None and change_opacity is not None:
        opacity_ = pc.get_opacity
        opacity_clone = opacity_.clone()
        opacity_clone[binding] = opacity_clone[binding] + change_opacity
        opacity = opacity_clone
    else:
        opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None #torch.Tensor([]).to(device)  # None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if deta_attr is not None and change_scales is not None:
            scales_ = pc.get_scaling
            scales_clone = scales_.clone()
            scales_clone[binding] = scales_clone[binding] + change_scales
            scales = scales_clone
            if change_rotation is not None:
                rotation_ = pc.get_rotation
                rotation_clone = rotation_.clone()
                rotation_clone[binding] = rotation_clone[binding] + change_rotation
                rotations = rotation_clone
            else:
                rotations = pc.get_rotation
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None   # torch.Tensor([]).to(device)  # None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
