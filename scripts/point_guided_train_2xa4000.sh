#!/usr/bin/env bash
set -euo pipefail

: "${BRATS21_ROOT:?Set BRATS21_ROOT to the BraTS2021 full-volume root}"
: "${MEDICALNET_CKPT:?Set MEDICALNET_CKPT to a local MedicalNet ResNet10 checkpoint}"
: "${MEDICALNET_SHA256:?Set MEDICALNET_SHA256 to the supplied checkpoint SHA-256 digest}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the run-artifact root}"
: "${POINT_GUIDED_PYTHON:=/home/aidev/miniconda3/envs/smagm-a4000/bin/python}"

[[ -d "$BRATS21_ROOT" ]] || { echo "BRATS21_ROOT is not a directory: $BRATS21_ROOT" >&2; exit 2; }
[[ -f "$MEDICALNET_CKPT" ]] || { echo "MEDICALNET_CKPT is not a file: $MEDICALNET_CKPT" >&2; exit 2; }
[[ -x "$POINT_GUIDED_PYTHON" ]] || { echo "POINT_GUIDED_PYTHON is not executable: $POINT_GUIDED_PYTHON" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

echo "git HEAD: $(git rev-parse HEAD)"
echo "hostname: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Python executable: $POINT_GUIDED_PYTHON"
"$POINT_GUIDED_PYTHON" - <<'PY'
import sys
import torch
print("Python version:", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
PY
echo "Chosen config: configs/training/point_guided_brats21_2xa4000.json"
echo "Launcher: torch.distributed.run DDP; one model and one sample per GPU"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHONPATH=src "$POINT_GUIDED_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m smagm.cli.point_guided_train \
  --config configs/training/point_guided_brats21_2xa4000.json \
  --data-root "$BRATS21_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --medicalnet-checkpoint "$MEDICALNET_CKPT" \
  --medicalnet-sha256 "$MEDICALNET_SHA256"
