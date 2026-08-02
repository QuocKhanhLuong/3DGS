#!/usr/bin/env bash
set -euo pipefail

# The repository has one product launch path.  Re-run this script with the
# same RUN_DIR to resume an interrupted run; a successful run is rejected by
# the controller rather than overwritten.
cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-experiments/runs/brats21-product-full-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"

if [[ -f "${RUN_DIR}/run_complete.json" ]]; then
  echo "refusing to overwrite completed run: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if ! SELECTED_BRATS_GPU="$(python scripts/select_free_gpu.py 2>&1)"; then
    printf '%s\n' "${SELECTED_BRATS_GPU}" | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${SELECTED_BRATS_GPU}"
fi

export PYTHONUNBUFFERED=1
WANDB_RUN_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_RUN_MODE}" != "online" && "${WANDB_RUN_MODE}" != "offline" && "${WANDB_RUN_MODE}" != "disabled" ]]; then
  echo "WANDB_MODE must be online, offline, or disabled; got ${WANDB_RUN_MODE}" >&2
  exit 2
fi
PYTHONPATH=src python scripts/train_brats21.py \
  --config configs/experiments/brats21_product_full.json \
  --output-dir "${RUN_DIR}" \
  --stage full \
  --resume auto \
  --wandb-mode "${WANDB_RUN_MODE}" \
  2>&1 | tee -a "${LOG_FILE}"
