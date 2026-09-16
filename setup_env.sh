#!/bin/bash
set -euo pipefail

PROJECT=$(cd "$(dirname "$0")" && pwd)
PY="${PYTHON:-python3}"
TSINGHUA=https://pypi.tuna.tsinghua.edu.cn/simple
PYTORCH3D_SRC="${PYTORCH3D_SRC:-to/path/pytorch3d-main}"

$PY -m pip config set global.index-url $TSINGHUA
$PY -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

$PY -m pip install "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
  --index-url https://download.pytorch.org/whl/cu121

$PY -m pip install tqdm numpy==1.22.3 matplotlib scipy chumpy tensorboard fvcore iopath \
  plyfile tyro dearpygui roma lpips -i $TSINGHUA

export PATH=/usr/local/cuda/bin:/usr/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CXXFLAGS="-I/usr/include/x86_64-linux-gnu"
export TORCH_CUDA_ARCH_LIST=8.0
PY_INCLUDE=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
cp -f /usr/include/crypt.h "$PY_INCLUDE/crypt.h"

cd $PROJECT/submodules/simple-knn && rm -rf build && $PY -m pip install -e .
cd $PROJECT/submodules/diff-gaussian-rasterization && rm -rf build && $PY -m pip install -e .

cd "$PYTORCH3D_SRC" && rm -rf build && $PY -m pip install -e . -i $TSINGHUA

$PY -c "import torch, pytorch3d; from simple_knn._C import distCUDA2; print('OK', torch.__version__, pytorch3d.__version__)"

cd $PROJECT/gridencoder && rm -rf build && $PY -m pip install -e .

$PY -c "from gridencoder import GridEncoder; print('gridencoder OK')"

cd $PROJECT
$PY -c "from scene import Scene, FlameGaussianModel; from mogface import MGCGModule, LatentTextureAttention; from gaussian_renderer import render; print('MoGaFace imports OK')"
