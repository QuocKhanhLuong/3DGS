# Point-guided BraTS21 server run

This document operates the existing locked point-guided frontend:
`T1/T2/FLAIR -> frozen MedicalNet ResNet10 -> S_coarse -> points -> sparse
PoU -> B -> fixed SWT-Haar A -> f_spec -> bounded Gate-C trajectory -> final-Z
Gate-D decoder`. It does not introduce a new architecture or claim a trained
checkpoint. The commands below are ready for a server checkout of `main`.

## Evidence boundary

Gate F F1/F2 software and synthetic engineering checks are complete, and Gate G
G1-G4 target-free inference/evaluation software is present. F3/F4 experiment
execution remains pending actual server evidence; no real trained checkpoint,
held-out evaluation, GPU/server evidence, reconstruction result, or clinical
claim is available from local software checks. This runbook records executable
commands and does not turn software readiness into experimental evidence.

The full-volume adapter is separate from the legacy sparse-plane
`smagm.data.brats21` contract. The model input is always exactly three
modalities in `[3,D,H,W]` tensor order: T1, T2, FLAIR. T1ce and segmentation
are loaded separately and are introduced only after target-free inference or
context creation.

## 1. Install

From the repository root, use the Python environment intended for the GPU:

```bash
export POINT_GUIDED_PYTHON=/path/to/gpu-env/bin/python
"$POINT_GUIDED_PYTHON" -m pip install -e '.[test,real-data,wandb]'
```

The normal trainer does not require W&B. The `wandb` extra is only needed when
`--wandb` is selected.

## 2. Dataset and checkpoint variables

Set paths without changing the repository scripts:

```bash
export BRATS21_ROOT=/path/to/BraTS2021_TrainingData
export MEDICALNET_CKPT=/path/to/resnet_10_23dataset.pth
export MEDICALNET_SHA256=<actual_sha256>
export OUTPUT_ROOT=/path/to/point_guided_runs
```

All supported point-guided launchers use `POINT_GUIDED_PYTHON` for their
PyTorch probe and module entrypoint. The 2xA4000 launcher also starts DDP with
`-m torch.distributed.run` from that same interpreter; it does not resolve a
separate ambient `python` or `torchrun` executable.
The wrappers retain the existing `/home/aidev/miniconda3/envs/smagm-a4000/bin/python`
server fallback for compatibility; on another host, override it explicitly.
They fail closed when the selected interpreter is not executable.

Do not replace `<actual_sha256>` with a guessed value. The loader never
downloads weights. It validates the local tensor checkpoint and records its
digest. `--require-pretrained-backbone` additionally requires a separately
reviewed digest in the repository's strict MedicalNet allowlist; no digest is
invented or treated as official by this checkout. A smoke/debug run may omit a
checkpoint only when invoking the Python API/config explicitly; the server
scripts require the checkpoint variables and report
`pretrained_backbone_verified` honestly.

The expected root is a directory of subject directories:

```text
BRATS21_ROOT/
  BraTS2021_00000/
    BraTS2021_00000_t1.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_seg.nii.gz
  BraTS2021_00001/
    ...
```

The adapter validates matching 3-D shapes, qform/sform and affine agreement,
finite positive spacing, finite tensors, and segmentation labels in
`{0,1,2,4}`. NIfTI `[X,Y,Z]` is explicitly copied to tensor `[D=Z,H=Y,W=X]`;
the original affine remains the physical `[x,y,z] == [w,h,d]` mapping.
No crop or resize is performed. MAIN preprocessing independently computes the
1st and 99th percentiles inside the raw input-derived union mask, clips, and
maps each modality to `[0,1]` with `masked_robust_01`. These percentile values
are tunable preprocessing settings and are recorded per modality. T1ce uses
the same input-derived mask but its own target-only percentile statistics;
those statistics never affect observations, points, or routing. The recorded
intensity-space name is `masked_robust_01_[0,1]`. The legacy
`masked_zscore` policy remains available for debugging, but requires an
explicit metric data range rather than silently assuming `1.0`.

## 3. Preflight

Inspect the GPU first, then validate the dataset discovery and checkpoint
digest:

```bash
nvidia-smi
bash scripts/point_guided_preflight.sh
```

The preflight is read-only with respect to the dataset and checkpoint. It
does not download weights or start training.

## 4. Tiny overfit on an RTX 4070

This is an ENGINEERING / DEBUG PROFILE. It uses the declared one-subject
overfit cohort, reduced `K_max`, reduced counterfactual candidates, and a
smaller decoder chunk. It is not held-out validation and does not auto-declare
a quality pass/fail.

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/point_guided_overfit_4070.sh
```

Inspect `train.jsonl`, `metrics.csv`, and `summary.json` to see whether the
reconstruction loss decreases on the declared overfit subjects. The trainer
does not fabricate convergence or clinical claims.

## 5. Single-GPU 4070 baseline

The MAIN architecture remains 2048 points, 4-mm support, at most 2-mm point
displacement, 64 route steps, 32 counterfactual candidates, fixed SWT-Haar,
168-d `f_spec`, 32-channel dynamic state planes, and the `96 -> 64 -> 32 -> 1`
decoder. The configured chunk size is an engineering memory knob.

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/point_guided_train_4070.sh
```

Optional CLI overrides are explicit and recorded in `config.json`; they do
not silently change architecture constants. For example, adding
`--gradient-accumulation 2` changes optimization scheduling only.

## 6. Two A4000 GPUs with true DDP

The dual-GPU launcher uses `torch.distributed`/`DistributedDataParallel`,
NCCL, one process per GPU, `torch.cuda.set_device(LOCAL_RANK)`, and a
`DistributedSampler` for training. It does not use `DataParallel` and does
not pool GPU memory. Each A4000 must fit one model/sample independently.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/point_guided_train_2xa4000.sh
```

Only rank 0 writes checkpoints, normal logs, W&B records, and the summary.
Validation indices are sharded without padding, so a subject is not counted
twice merely to equalize rank lengths. Validation uses the raw local model
module rather than the DDP wrapper, and only final validation statistics are
all-reduced. The process group is destroyed on normal exit and failure
cleanup.

## 7. Resume

The trainer writes a resumable checkpoint after each completed epoch:

```bash
PYTHONPATH=src "$POINT_GUIDED_PYTHON" -m smagm.cli.point_guided_train \
  --config configs/training/point_guided_brats21_4070.json \
  --data-root "$BRATS21_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --split-file "$OUTPUT_ROOT/<run>/split.json" \
  --medicalnet-checkpoint "$MEDICALNET_CKPT" \
  --medicalnet-sha256 "$MEDICALNET_SHA256" \
  --resume "$OUTPUT_ROOT/<run>/checkpoints/last_train.pt"
```

Resume checks the schema, exact model state, optimizer state, optional
GradScaler state, split hash, counters, and RNG state. A mismatched split is
rejected rather than silently continuing on different subjects.

New training runs reserve their directory with an exclusive filesystem create;
an existing `--run-name` fails rather than mixing logs or checkpoints. Reuse
is allowed only through the explicit `--resume` checkpoint path, which holds a
run lock for the duration of the resumed writer. A stale
`.point-guided.lock` must be investigated and removed manually before reuse.

## 8. Evaluation

Evaluation loads the clean checkpoint through the strict existing validated
loader and automatically requires the exact `<run>/split.json` next to the
checkpoint. An explicit `--split-file` may override the path, but no split is
silently regenerated for a trained checkpoint. It calls only the target-free
baseline inference API first; T1ce is read afterward for MAE/PSNR/SSIM, and
segmentation is read afterward for semantic Dice diagnostics. MAIN
PSNR/SSIM uses `data_range=1.0`, matching the robust `[0,1]` preprocessing.

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/point_guided_eval.sh \
  "$OUTPUT_ROOT/<run>/checkpoints/best_model.pt"
```

The default evaluation split is `test`. It is never used for checkpoint
selection or hyperparameter tuning. To evaluate another split directly:

```bash
PYTHONPATH=src "$POINT_GUIDED_PYTHON" -m smagm.cli.point_guided_eval \
  "$OUTPUT_ROOT/<run>/checkpoints/best_model.pt" \
  --config configs/evaluation/point_guided_brats21_eval.json \
  --data-root "$BRATS21_ROOT" \
  --output-dir "$OUTPUT_ROOT/<run>/evaluation-test" \
  --split test \
  --device cuda \
  --medicalnet-checkpoint "$MEDICALNET_CKPT" \
  --medicalnet-sha256 "$MEDICALNET_SHA256" \
  --save-predictions
```

Evaluation reserves `--output-dir` before inference and rejects an existing
directory by default. To intentionally replace artifacts in an existing
directory, pass `--reuse-output` (or set `POINT_GUIDED_REUSE_OUTPUT=1` for the
shell wrapper); this never clears stale files and cannot run concurrently with
another writer. JSON, checkpoint, debug-prediction, and NIfTI artifacts use
unique sibling temporary files followed by atomic replacement.

If predictions are saved, they are transposed from tensor `[D,H,W]` back to
NIfTI `[X,Y,Z]` and written with the source affine. Output metrics explicitly
identify normalized intensity space; they are not clinical quality claims.

## 9. Artifacts

Each run directory contains:

```text
<run>/
  config.json
  split.json
  environment.json
  train.jsonl
  metrics.csv
  summary.json
  checkpoints/
    last_train.pt
    best_model.pt
```

`last_train.pt` is the resume format and includes optimizer/scaler/RNG state.
`best_model.pt` is intentionally the exact clean inference payload:

```python
{"metadata": baseline_checkpoint_metadata(model), "state_dict": model.state_dict()}
```

The best checkpoint is selected by lowest validation reconstruction loss.
Evaluation adds `per_subject_metrics.json`, `aggregate_metrics.json`,
`trajectory_diagnostics.json`, `evaluation_metadata.json`, and optionally a
`predictions/` directory.

`evaluation_metadata.json` records the exact split file, split hash, training
run directory, and normalization-space label used for metrics.

## 10. OOM knobs

If a CUDA OOM occurs, the trainer fails with the same scientific settings and
suggests these engineering knobs:

```text
decoder_chunk_size
counterfactual_candidates
K_max
gradient_accumulation
batch size
```

Reduce them explicitly in a copied config or CLI override and record the
change. The code does not automatically resize anatomy, reduce 2048 points,
change 4-mm support, or alter the 2-mm displacement bound. DDP does not pool
memory across A4000 devices.

## 11. Sending logs for diagnosis

Send the run's `config.json`, `split.json`, `environment.json`,
`train.jsonl`, `metrics.csv`, `summary.json`, and the exact command. Include
the first complete traceback and `nvidia-smi` output captured near the
failure. Do not send the MedicalNet checkpoint itself or secrets. Report
whether the run was the debug overfit, single-GPU MAIN, or DDP MAIN profile.

The next scientific decision should wait for actual server evidence: tiny
overfit logs, training curves, route diagnostics, reconstruction metrics, and
GPU profiling.
