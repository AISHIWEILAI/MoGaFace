#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
SUBJECTS=("306" "074")

export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib/python3.8/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH=/usr/local/cuda/bin:/usr/bin:${PATH}
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.0

cd "${PROJECT_ROOT}"

for SUBJECT in "${SUBJECTS[@]}"; do
  echo "========== Running validation inference for subject ${SUBJECT} =========="
  "$PYTHON" render.py \
    -m "${PROJECT_ROOT}/output/nersemble/${SUBJECT}_20material_allviews_expemo" \
    --hum_id "${SUBJECT}" \
    --skip_train \
    --skip_test
done

echo "========== Validation inference completed =========="
