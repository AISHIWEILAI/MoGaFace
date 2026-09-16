# MoGaFace: Momentum-Guided and Texture-Aware Gaussian Avatars for Consistent Facial Geometry [PRCV 2026]

<p align="center">
  <a href="https://mogaface.github.io/">Project Page</a> |
  <a href="https://arxiv.org/abs/2508.01218">Paper (arXiv)</a>
</p>

---

## Installation & Dependencies

### Linux / Ubuntu

Tested with **Python 3.8**, **PyTorch 2.4.1 + CUDA 12.1**. Environment setup follows common 3D Gaussian Splatting + PyTorch3D workflows (see `setup_env.sh`).

```bash
git clone https://github.com/AISHIWEILAI/MoGaFace.git --recursive
cd MoGaFace

# If already cloned without --recursive:
# git submodule update --init --recursive

# Recommended: conda environment
conda create -n mogaface python=3.8
conda activate mogaface

# One-shot setup (Tsinghua pip mirror + CUDA extensions)
bash setup_env.sh
```

| Component | Version |
|-----------|---------|
| PyTorch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| pytorch3d | 0.7.8 |
| diff-gaussian-rasterization / simple-knn | `submodules/` (git submodules) |
| gridencoder | `gridencoder/` → `submodules/torch-ngp/gridencoder` (symlink) |

> **Note**: PyTorch3D may require building from source. See `setup_env.sh` for compiler flags (`gcc`, `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`).

---

## FLAME Assets Preparation

MoGaFace is built on **FLAME 2023**. Due to the [FLAME license](https://flame.is.tue.mpg.de/modellicense.html), model files are **not** included in this repository. You must register and download them from the official site:

**[https://flame.is.tue.mpg.de/download.php](https://flame.is.tue.mpg.de/download.php)**

After registration, download the following resources and place them as shown (same convention as [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)):

| Resource | Official download | Target path |
|----------|-------------------|-------------|
| FLAME 2023 (w/ jaw rotation) | FLAME 2023 model | `flame_model/assets/flame/flame2023.pkl` |
| FLAME Vertex Masks | FLAME masks | `flame_model/assets/flame/FLAME_masks.pkl` |
| Landmark embedding (w/ eyes) | FLAME landmark embedding | `flame_model/assets/flame/landmark_embedding_with_eyes.npy` |
| Head template mesh | FLAME template / geometry resources | `flame_model/assets/flame/head_template_mesh.obj` |
| Mean texture (optional) | FLAME texture resources | `flame_model/assets/flame/tex_mean_painted.png` |
| MediaPipe landmark embedding | FLAME MediaPipe resource | `flame_model/assets/mediapipe/mediapipe_landmark_embedding.npz` |

Expected layout:

```text
flame_model/
└── assets/
    ├── flame/
    │   ├── flame2023.pkl
    │   ├── FLAME_masks.pkl
    │   ├── landmark_embedding_with_eyes.npy
    │   ├── head_template_mesh.obj
    │   └── tex_mean_painted.png          # optional
    └── mediapipe/
        └── mediapipe_landmark_embedding.npz
```

> **Important**
> - You need to **sign up** on the FLAME website and agree to the license before downloading.
> - FLAME assets are for **non-commercial research** only; see the [model license](https://flame.is.tue.mpg.de/modellicense.html).
> - Validation inference does **not** require `face_mask.pth`.

---

## Usage

All scripts should be run from the **project root**.

**Data & checkpoints are not included in this GitHub repo.** Download them from Baidu Netdisk (see below) and extract to `data/` and `output/` before running inference.

| Script | Usage |
|--------|-------|
| `setup_env.sh` | Install conda/pip dependencies and build CUDA extensions |
| `infer_val.sh` | Batch validation inference for subjects **306** and **074** |
| `render.py` | Single-subject inference with custom flags |

### 1. Data & checkpoints

> **Note:** Due to size limits, **multi-view data** and **pretrained checkpoints** are hosted on Baidu Netdisk and are **not** synced with this GitHub repository. After downloading, extract the archives into the project root so that paths match the layout below.

| Resource | Baidu Netdisk | Extract code |
|----------|---------------|--------------|
| NeRSemble data (subjects 306 & 074) | [Download data](https://pan.baidu.com/s/PLACEHOLDER_DATA) | `xxxx` |
| Pretrained checkpoints (306 & 074) | [Download checkpoints](https://pan.baidu.com/s/PLACEHOLDER_CKPT) | `xxxx` |

**After extraction, the expected layout is:**

```text
data/
├── 306_20material_all_views/
│   └── UNION20_306_EMO1234EXP234589_v16_DS4_whiteBg_staticOffset_maskBelowLine/
└── 074_20material_all_views/
    └── UNION20_074_EMO1234EXP234589_v16_DS4_whiteBg_staticOffset_maskBelowLine/

output/nersemble/
├── 306_20material_allviews_expemo/
└── 074_20material_allviews_expemo/
```

| Subject | Data path (`-s`, resolved in `render.py`) | Model path (`-m`) |
|---------|-------------------------------------------|-------------------|
| 306 | `data/306_20material_all_views/UNION20_306_EMO1234EXP234589_v16_DS4_whiteBg_staticOffset_maskBelowLine` | `output/nersemble/306_20material_allviews_expemo` |
| 074 | `data/074_20material_all_views/UNION20_074_EMO1234EXP234589_v16_DS4_whiteBg_staticOffset_maskBelowLine` | `output/nersemble/074_20material_allviews_expemo` |

Replace `PLACEHOLDER_DATA` / `PLACEHOLDER_CKPT` and extract codes with the actual Baidu Netdisk links before publishing.

### 2. Validation inference

**Both subjects:**

```bash
bash infer_val.sh
```

**Single subject:**

```bash
python render.py \
  -m output/nersemble/306_20material_allviews_expemo \
  --hum_id 306 \
  --skip_train \
  --skip_test
```

| Flag | Meaning |
|------|---------|
| `-m` | Checkpoint directory |
| `--hum_id` | Subject ID (`306` or `074`) |
| `--skip_train` | Skip training cameras (run validation only) |
| `--skip_test` | Skip test cameras |

**Outputs** (example subject 306):

```text
{model_path}/val_8/ours_1000000/
├── renders/           # rendered images
├── gt/                # ground-truth images
├── infer_results.txt  # PSNR / SSIM / LPIPS
├── renders.mp4
└── high_renders.mp4
```

---

## Citation

Please cite the following paper if you use this method, model, or conduct derivative research based on this project:

```bibtex
@inproceedings{liu2026mogaface,
  title={MoGaFace: Momentum-Guided and Texture-Aware Gaussian Avatars for Consistent Facial Geometry},
  author={Liu, Yujian and Cao, Linlang and Chen, Chuang and Geng, Fanyu and Shen, Dongxu and Cao, Peng and Xu, Shidang and Liu, Xiaoli},
  booktitle={Proceedings of the Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  year={2026}
}
```

---

## Acknowledgements

This project is built upon or inspired by the following open-source projects:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
- [TensorialGaussianAvatar](https://github.com/ant-research/TensorialGaussianAvatar)
- [FLAME](https://flame.is.tue.mpg.de/)
- [SyncAnimation](https://github.com/AISHIWEILAI/syncanimation)

We sincerely thank the authors of these projects for their contributions to the open-source community.

---

## Disclaimer

By using this project, you agree to comply with all applicable laws and regulations.
You must not use it to generate or disseminate harmful content.
FLAME and third-party CUDA extensions are subject to their respective licenses.
The developers assume no responsibility for any direct, indirect, or consequential damages arising from the use or misuse of this software.
