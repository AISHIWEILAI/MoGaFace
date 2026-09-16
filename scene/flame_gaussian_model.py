# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
from pathlib import Path
import numpy as np
import torch
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
import copy
import pickle
import glob
from utils.system_utils import mkdir_p
import os
from plyfile import PlyData, PlyElement
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, disable_flame_static_offset=False,
                 not_finetune_flame_params=False, n_shape=300, n_expr=100,
                 device='cuda', infer_flame_param=''):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params

        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape,
            n_expr,
            add_teeth=True,
        ).cuda()
        self.device = torch.device(device)

        with open(Path(__file__).resolve().parent.parent / 'flame_model/assets/flame/FLAME_masks.pkl', 'rb') as f:
            flame_masks = pickle.load(f, encoding='latin1')
        self.lips_mask = flame_masks['lips']  # idx list

        self.flame_param = None
        self.flame_param_orig = None
        self.flame_param_diff = None
        self.drive_model = None

        if infer_flame_param:
            flame_param_offset = dict(np.load(infer_flame_param, allow_pickle=True))
            self.flame_param_diff = {
                'shape': torch.from_numpy(flame_param_offset['shape']).cuda(),
                'expr': torch.from_numpy(flame_param_offset['expr']).cuda(),
                'rotation': torch.from_numpy(flame_param_offset['rotation']).cuda(),
                'neck_pose': torch.from_numpy(flame_param_offset['neck_pose']).cuda(),
                'jaw_pose': torch.from_numpy(flame_param_offset['jaw_pose']).cuda(),
                'eyes_pose': torch.from_numpy(flame_param_offset['eyes_pose']).cuda(),
                'translation': torch.from_numpy(flame_param_offset['translation']).cuda(),
            }

        self.offset_R = None
        self.offset_T = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).to(self.device)
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).to(self.device)
            self.binding_ori = torch.arange(len(self.flame_model.faces)).to(self.device)
            num_fill_pts = 1000
            self.binding_lip = torch.arange(self.binding.shape[0] - (len(self.flame_model.faces)-len(self.flame_model.faces_ori)),  self.binding.shape[0]).to(self.device)

        self.trainval_len = 0
        self.test_len = 0
        self.mapping_idx = {}

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:

            self.trainval_len = len(train_meshes)
            self.test_len = len(test_meshes)

            idx = self.trainval_len
            for i in range(self.test_len):
                self.mapping_idx[idx] = i
                idx += 1

            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes

            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])

            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().to(self.device)

            if tgt_meshes:
                T_tgt = max(tgt_meshes) + 1  # required by viewers
                self.flame_param_tgt = {
                    'shape': torch.from_numpy(tgt_meshes[0]['shape']),
                    'expr': torch.zeros([T_tgt, tgt_meshes[0]['expr'].shape[1]]),
                    'rotation': torch.zeros([T_tgt, 3]),
                    'neck_pose': torch.zeros([T_tgt, 3]),
                    'jaw_pose': torch.zeros([T_tgt, 3]),
                    'eyes_pose': torch.zeros([T_tgt, 6]),
                    'translation': torch.zeros([T_tgt, 3]),
                    'static_offset': static_offset,
                    'dynamic_offset': torch.zeros([T_tgt, num_verts, 3]),
                }

                for i, mesh in tgt_meshes.items():
                    self.flame_param_tgt['expr'][i] = torch.from_numpy(mesh['expr'])
                    self.flame_param_tgt['rotation'][i] = torch.from_numpy(mesh['rotation'])
                    self.flame_param_tgt['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                    self.flame_param_tgt['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                    self.flame_param_tgt['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                    self.flame_param_tgt['translation'][i] = torch.from_numpy(mesh['translation'])

                for k, v in self.flame_param_tgt.items():
                    self.flame_param_tgt[k] = v.float().to(self.device)

            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}

            if tgt_meshes:
                exp_mean = torch.mean(self.flame_param_tgt['expr'], dim=0)
                self.scales = (10 ** torch.floor(torch.log10(
                    exp_mean.abs() + 1e-8)) * 20).unsqueeze(0).cuda()

                rotation_mean = torch.mean(self.flame_param_tgt['rotation'], dim=0)
                self.scales_rotation = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param_tgt['rotation']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                neck_pose_mean = torch.mean(self.flame_param_tgt['neck_pose'], dim=0)
                self.scales_neck_pose = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param_tgt['neck_pose']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                jaw_pose_mean = torch.mean(self.flame_param_tgt['jaw_pose'], dim=0)
                self.scales_jaw_pose = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param_tgt['jaw_pose']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                eyes_pose_mean = torch.mean(self.flame_param_tgt['eyes_pose'], dim=0)
                self.scales_eyes_pose = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param_tgt['eyes_pose']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                self.scales_translation_diff = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param_tgt['translation']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                self.scales_translation = 100 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param_tgt['translation']), dim=0) + 1e-8))).unsqueeze(0).cuda()

            else:
                exp_mean = torch.mean(self.flame_param['expr'], dim=0)
                self.scales = (10 ** torch.floor(torch.log10(
                    exp_mean.abs() + 1e-8)) * 20).unsqueeze(0).cuda()

                rotation_mean = torch.mean(self.flame_param['rotation'], dim=0)
                self.scales_rotation = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param['rotation']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                neck_pose_mean = torch.mean(self.flame_param['neck_pose'], dim=0)
                self.scales_neck_pose = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param['neck_pose']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                jaw_pose_mean = torch.mean(self.flame_param['jaw_pose'], dim=0)
                self.scales_jaw_pose = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param['jaw_pose']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                eyes_pose_mean = torch.mean(self.flame_param['eyes_pose'], dim=0)
                self.scales_eyes_pose = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param['eyes_pose']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                self.scales_translation_diff = 20 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param['translation']), dim=0) + 1e-8))).unsqueeze(0).cuda()

                self.scales_translation = 100 * (10 ** torch.floor(torch.log10(
                    torch.mean(torch.abs(self.flame_param['translation']), dim=0) + 1e-8))).unsqueeze(0).cuda()

        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass

    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].to(self.device),
            flame_param['rotation'].to(self.device),
            flame_param['neck'].to(self.device),
            flame_param['jaw'].to(self.device),
            flame_param['eyes'].to(self.device),
            flame_param['translation'].to(self.device),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, offset_flame_params=None):
        self.timestep = timestep
        flame_param = self.flame_param

        if offset_flame_params is not None:
            verts, verts_cano = self.flame_model(
                flame_param['shape'][None, ...],
                flame_param['expr'][[timestep]] + offset_flame_params['expr'],
                flame_param['rotation'][[timestep]],
                flame_param['neck_pose'][[timestep]],
                flame_param['jaw_pose'][[timestep]],
                flame_param['eyes_pose'][[timestep]],
                flame_param['translation'][[timestep]],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'],
                dynamic_offset=flame_param['dynamic_offset'][[timestep]],
            )
            self.update_mesh_properties(verts, verts_cano)
            return verts

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_properties(self, verts, verts_cano):
        """"""

        faces = self.flame_model.faces
        triangles = verts[:, faces]

        self.face_center = triangles.mean(dim=-2).squeeze(0)

        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0),
                                                                          faces.squeeze(0),
                                                                          return_scale=True)
        self.face_orien_mat = self.face_orien_mat.to(self.device)
        self.face_scaling = self.face_scaling.to(self.device)

        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano

    def compute_dynamic_offset_loss(self):
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()

    def compute_laplacian_loss(self):
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()

    def training_setup(self, training_args, device,
                       drive_model=None, verts_model=None):
        super().training_setup(training_args, device)
        if drive_model is not None:
            print("################ exist drive_model #################")
            self.drive_model = drive_model
            param_drive_model = {'params': drive_model.parameters(), 'lr': training_args.flame_pose_lr, "name": "drive_model"}
            self.optimizer.add_param_group(param_drive_model)

            self.verts_model = verts_model
            param_verts_model = {'params': verts_model.parameters(), 'lr': training_args.flame_pose_lr, "name": "lip_model"}
            self.optimizer.add_param_group(param_verts_model)

        if drive_model is None:
            self.offset_R = torch.randn((62007, 6), device=device, dtype=torch.float32).to(self.device)
            self.offset_T = torch.randn((62007, 3), device=device, dtype=torch.float32).to(self.device)
            self.offset_R.requires_grad = True
            self.offset_T.requires_grad = True
            param_offset_R = {'params': [self.offset_R], 'lr': training_args.flame_pose_lr, "name": "offset_R"}
            self.optimizer.add_param_group(param_offset_R)
            param_offset_T = {'params': [self.offset_T], 'lr': training_args.flame_pose_lr, "name": "offset_T"}
            self.optimizer.add_param_group(param_offset_T)

        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr,
                       "name": "trans"}
        self.optimizer.add_param_group(param_trans)

        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        if self.not_finetune_flame_params:
            print('################## only change cameron, not finetune flame #################')
            return

        self.flame_param_diff = copy.deepcopy(self.flame_param)
        # pose
        self.flame_param_diff['rotation'].requires_grad = True
        self.flame_param_diff['neck_pose'].requires_grad = True
        self.flame_param_diff['jaw_pose'].requires_grad = True
        self.flame_param_diff['eyes_pose'].requires_grad = True
        params = [
            self.flame_param_diff['rotation'],
            self.flame_param_diff['neck_pose'],
            self.flame_param_diff['jaw_pose'],
            self.flame_param_diff['eyes_pose'],
        ]
        param_pose_diff = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose_diff)

        # translation
        self.flame_param_diff['translation'].requires_grad = True
        param_trans_diff = {'params': [self.flame_param_diff['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans_diff)

        # expression
        self.flame_param_diff['expr'].requires_grad = True
        param_expr_diff = {'params': [self.flame_param_diff['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr_diff)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

        if self.flame_param_diff is not None:
            offset_npz_path = Path(path).parent / "offset_param.npz"
            flame_param_diff = {k: v.cpu().numpy() for k, v in self.flame_param_diff.items()}
            np.savez(str(offset_npz_path), **flame_param_diff)
        if self.offset_R is not None and self.offset_T is not None:
            offset_npz_cameron_path = Path(path).parent / "offset_cameron_param.npz"
            cameron_params = {"offset_R": self.offset_R.cpu().numpy(),
                              "offset_T": self.offset_T.cpu().numpy()}
            np.savez(str(offset_npz_cameron_path), **cameron_params)

    def save_ply_tradition_method(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        if self.binding is not None:
            binding = self.binding.detach().cpu().numpy()
            attributes = np.concatenate((attributes, binding[:, None]), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).to(self.device) for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers

        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).to(self.device) for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers

        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]

    def load_all_meshes(self, root_dir, num_fill_pts=1000):
        materials = os.listdir(root_dir)
        self.all_meshes = {}
        for material in materials:
            mesh_dir = os.path.join(root_dir, material, 'flame_param')
            material_name = material.split('_')[2]
            flame_paths = sorted(glob.glob(os.path.join(mesh_dir, '*.npz')))
            flame_meshes = {}
            for idx, flame_path in enumerate(flame_paths):
                flame_param = dict(np.load(flame_path, allow_pickle=True))
                flame_meshes[idx] = flame_param

            meshes = {**flame_meshes}
            if len(meshes) == 0:
                continue
            num_verts = self.flame_model.v_template.shape[0]
            pose_meshes = meshes
            T = max(pose_meshes) + 1  # required by viewers

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset,
                                                            (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            flame_param_material = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                flame_param_material['expr'][i] = torch.from_numpy(mesh['expr'])
                flame_param_material['rotation'][i] = torch.from_numpy(mesh['rotation'])
                flame_param_material['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                flame_param_material['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                flame_param_material['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                flame_param_material['translation'][i] = torch.from_numpy(mesh['translation'])

            for k, v in flame_param_material.items():
                flame_param_material[k] = v.float().to(self.device)
            self.all_meshes[material_name] = flame_param_material
        self.lip_verts_dict = self.save_all_lip_vert()
        self.lip_pts_filled(self.lip_verts_dict, num_fill_pts=num_fill_pts)

    def lip_pts_filled(self, lip_verts_dict, num_fill_pts=1000):
        """"""

        lip_verts_list = []
        for material in lip_verts_dict:
            lip_verts_list.append(lip_verts_dict[material])  # (N, 254, 3)

        lip_verts = torch.cat(lip_verts_list, dim=0)  # (N_all, 254, 3)
        N_all = lip_verts.shape[0]

        all_lip_pts = lip_verts.reshape(-1, 3)  # (N_all * 254, 3)

        mouth_center = torch.mean(all_lip_pts, dim=0, keepdim=True)  # (1, 3)

        mouth_center[:, 1] = all_lip_pts[:, 1].min()

        max_dist_per_point = (all_lip_pts - mouth_center).abs()  # (N_all * 254, 3)
        max_dist = max_dist_per_point.max(dim=0)[0]

        rand_unit = (torch.rand(num_fill_pts, 3, device=max_dist.device) - 0.5) * 1.6
        self.filled_lip_pts = rand_unit * max_dist + mouth_center  # (num_fill_pts, 3)

    def save_all_lip_vert(self):
        all_meshes = self.all_meshes
        all_verts = {}
        for name in all_meshes.keys():
            v = all_meshes[name]
            num_frames = v['expr'].shape[0]
            lip_verts_list = []
            for i in range(num_frames):
                verts, verts_cano = self.flame_model(
                    v['shape'][None, ...],
                    v['expr'][[i]],
                    v['rotation'][[i]],
                    v['neck_pose'][[i]],
                    v['jaw_pose'][[i]],
                    v['eyes_pose'][[i]],
                    v['translation'][[i]],
                    zero_centered_at_root_node=False,
                    return_landmarks=False,
                    return_verts_cano=True,
                    static_offset=v['static_offset'],
                    dynamic_offset=v['dynamic_offset'][[i]],
                )

                lip_verts = verts.squeeze(0)[self.lips_mask]  # 254,3
                lip_verts_list.append(lip_verts)
            all_verts[name] = torch.stack(lip_verts_list).cuda()
        return all_verts
