import re
import subprocess
import sys
import pickle
from utils.loss_utils import l1_loss
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
from gaussian_renderer import render, network_gui
from mogface.mgcg import MGCGModule
from mogface.lta import LatentTextureAttention

from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.loss_utils import ssim

mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset: ModelParams, name, iteration,
               views, gaussians, pipeline,
               background, render_mesh, pretrain_features=None,
               lta_module=None, mgcg_module=None, dataset_name=None):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []

    flame_params = {}
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):

        image_name = view.image_name
        current_img_idx, MV_imgs = next(iter(view.MV_images.items()))

        if dataset_name == 'test':
            cam_idx = int(image_name.split('_')[1])
            if cam_idx >= 8:
                cam_idx -= 1

            flame_params, _, _, img_features = mgcg_module(MV_imgs.cuda())
            expr_ = pretrain_features[view.timestep]
            expr_[cam_idx, :] = flame_params['expr'][0, :]

            flame_params = {
                'expr': torch.mean(expr_, dim=0).unsqueeze(0),
            }
        elif dataset_name == 'val':
            _, _, _, img_features = mgcg_module(MV_imgs.cuda())
            expr_ = pretrain_features[view.timestep]
            flame_params = {
                'expr': torch.mean(expr_, dim=0).unsqueeze(0),
            }
        else:
            raise NotImplementedError

        verts = gaussians.select_mesh_by_timestep(view.timestep, offset_flame_params=flame_params)

        deta_attr = lta_module(verts.squeeze(0), img_features)

        rendering = render(view, gaussians, pipeline, background, deta_attr=deta_attr)["render"]

        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + gt.to(rgb_mesh) * (
                        alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        image = torch.clamp(rendering, 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += ssim(image, gt_image).mean().double()
        lpips_test += lpips(image, gt_image).mean().double()

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader) - 1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []

    try:
        os.system(
            f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        command = [
            'ffmpeg',
            '-r', '25',
            '-f', 'image2',
            '-i', f'{render_path}/%05d.png',
            '-pix_fmt', 'yuv420p',
            '-q:v', '0',
            '-q:a', '0',
            f"{iter_path}/high_renders.mp4"
        ]
        if not os.path.exists(f"{iter_path}/high_renders.mp4"):
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.system(
            f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        command = [
            'ffmpeg',
            '-r', '25',
            '-f', 'image2',
            '-i', f'{gts_path}/%05d.png',
            '-pix_fmt', 'yuv420p',
            '-q:v', '0',
            '-q:a', '0',
            f"{iter_path}/high_gt.mp4"
        ]
        if not os.path.exists(f"{iter_path}/high_gt.mp4"):
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        result_str = "[ITER {}]: PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(
            iteration, psnr_test / len(views_loader), ssim_test / len(views_loader), lpips_test/ len(views_loader))

        filename = f"{iter_path}/infer_results.txt"
        with open(filename, 'a') as f:
            f.write(result_str + '\n')

        if render_mesh:
            os.system(
                f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_val: bool,
                skip_test: bool, render_mesh: bool, check_info: str, dataset_name: str):
    project_root = Path(__file__).resolve().parent
    dataset.source_path = str(project_root / f'data/{check_info[0]}_20material_all_views/UNION20_{check_info[0]}_EMO1234EXP234589_v16_DS4_whiteBg_staticOffset_maskBelowLine')

    dataset.disable_flame_static_offset = False

    mgcg_module = MGCGModule().cuda()
    lta_module = LatentTextureAttention().cuda()

    check_path = os.path.join(dataset.model_path, f'chkpnt{check_info[1]}.pth')
    _, mgcg_module_model_dict_, lta_module_state_dict, pretrain_features, _ = torch.load(check_path)
    pretrain_features = {k: v.cuda() for k, v in pretrain_features.items()}

    mgcg_module.load_state_dict(mgcg_module_model_dict_, strict=True)
    mgcg_module.eval()

    lta_module.load_state_dict(lta_module_state_dict, strict=True)
    lta_module.eval()

    gaussians = FlameGaussianModel(dataset.sh_degree,
                                   dataset.disable_flame_static_offset)

    scene = Scene(dataset, gaussians, load_iteration=iteration,
                  shuffle=False, check_info=check_info[1])

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        if dataset.target_path != "":
            name = os.path.basename(os.path.normpath(dataset.target_path))
            render_set(dataset, f'{name}', scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, render_mesh, pretrain_features, lta_module, mgcg_module)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter,
                           scene.getTrainCameras(), gaussians,
                           pipeline, background, render_mesh, pretrain_features, lta_module, mgcg_module, dataset_name)

            if not skip_val:
                dataset.select_camera_id = 8
                render_set(dataset, "val", scene.loaded_iter,
                           scene.getValCameras(), gaussians,
                           pipeline, background, render_mesh, pretrain_features, lta_module, mgcg_module, dataset_name)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter,
                           scene.getTestCameras(), gaussians,
                           pipeline, background, render_mesh, pretrain_features, lta_module, mgcg_module, dataset_name)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--hum_id", default='057')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    def read_checkpoint(model_path, dataset_name, hum_id):
        with open(os.path.join(model_path, f"{dataset_name}_results.txt"), 'r') as file:
            lines = file.readlines()

        data = [hum_id]
        for line in lines:
            if '#' in line:
                numbers = re.findall(r'\b\d+\b', line)
                if numbers:
                    iteration = numbers[0]
                    print(f"Found line: {line.strip()}")
                    print(f"Extracted numbers: {numbers}")
                    data.append(iteration)
                    return data

    if (args.skip_val is False and args.skip_train is True) or (args.skip_val is True and args.skip_train is False):
        dataset_name = 'val'
        check_info = read_checkpoint(args.model_path, dataset_name, args.hum_id)
    elif args.skip_test is False:
        dataset_name = 'test'
        check_info = read_checkpoint(args.model_path, dataset_name, args.hum_id)
    else:
        raise NotImplementedError

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_val, args.skip_test, args.render_mesh, check_info, dataset_name)
