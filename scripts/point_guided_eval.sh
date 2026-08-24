#!/usr/bin/env bash
set -euo pipefail

: "${BRATS21_ROOT:?Set BRATS21_ROOT to the BraTS2021 full-volume root}"
: "${MEDICALNET_CKPT:?Set MEDICALNET_CKPT to a local MedicalNet ResNet10 checkpoint}"
: "${MEDICALNET_SHA256:?Set MEDICALNET_SHA256 to the supplied checkpoint SHA-256 digest}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the run-artifact root}"
: "${POINT_GUIDED_PYTHON:=/home/aidev/miniconda3/envs/smagm-a4000/bin/python}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CHECKPOINT [EVAL_OUTPUT_DIR]" >&2
  exit 2
fi
CHECKPOINT="$1"
EVAL_OUTPUT_DIR="${2:-$OUTPUT_ROOT/eval-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
SPLIT_FILE="$RUN_DIR/split.json"

[[ -f "$CHECKPOINT" ]] || { echo "checkpoint is not a file: $CHECKPOINT" >&2; exit 2; }
[[ -f "$SPLIT_FILE" ]] || { echo "training split is not a file: $SPLIT_FILE" >&2; exit 2; }
[[ -d "$BRATS21_ROOT" ]] || { echo "BRATS21_ROOT is not a directory: $BRATS21_ROOT" >&2; exit 2; }
[[ -f "$MEDICALNET_CKPT" ]] || { echo "MEDICALNET_CKPT is not a file: $MEDICALNET_CKPT" >&2; exit 2; }
[[ -x "$POINT_GUIDED_PYTHON" ]] || { echo "POINT_GUIDED_PYTHON is not executable: $POINT_GUIDED_PYTHON" >&2; exit 2; }
run_eval() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH=src "$POINT_GUIDED_PYTHON" -m smagm.cli.point_guided_eval \
    "$CHECKPOINT" \
    --config configs/evaluation/point_guided_brats21_eval.json \
    --data-root "$BRATS21_ROOT" \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --split-file "$SPLIT_FILE" \
    --split test \
    --device cuda \
    --medicalnet-checkpoint "$MEDICALNET_CKPT" \
    --medicalnet-sha256 "$MEDICALNET_SHA256" \
    "$@"
}

echo "git HEAD: $(git rev-parse HEAD)"
echo "hostname: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Python executable: $POINT_GUIDED_PYTHON"
"$POINT_GUIDED_PYTHON" - <<'PY'
import sys
import torch
print("Python version:", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
PY
echo "Chosen config: configs/evaluation/point_guided_brats21_eval.json"
echo "Checkpoint: $CHECKPOINT"
echo "Training split: $SPLIT_FILE"
echo "Evaluation output: $EVAL_OUTPUT_DIR"

if [[ "${POINT_GUIDED_REUSE_OUTPUT:-0}" == "1" ]]; then
  run_eval --reuse-output
else
  run_eval
fi
