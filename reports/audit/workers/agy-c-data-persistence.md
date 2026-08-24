# AGY-C Audit Report: Data, Split Provenance, Checkpoint, and Artifact Persistence

- **Audit Target**: `main` at frozen commit `0efeb94af72ffa067769e19afcd19ad358feefd2`
- **Audit Lane**: `AGY-C` (Data, split provenance, checkpoint, and artifact persistence)
- **Codegraph Task**: `server_pipeline`
- **Auditor**: Dispatched Worker `AGY-C`
- **Date**: 2026-08-23

---

## 1. Executive Summary

This report delivers an exhaustive, evidence-backed audit of the full-volume BraTS21 data pipeline, NIfTI geometry handling, normalization, deterministic subject splitting, batch collation, checkpoint serialization, and evaluation artifact provenance at frozen `main` HEAD `0efeb94af72ffa067769e19afcd19ad358feefd2`.

### Key Conclusions:
1. **Target Decoupling & Scientific Boundary**: The observation input tensor (`[B, 3, D, H, W]` in `(T1, T2, FLAIR)` order), input-derived brain mask, deterministic initial points, point refinement, static base planes ($B$), stationary Haar wavelet spectral anchor ($A$), 168-d point spectral evidence ($f_{\text{spec}}$), dynamic tri-plane trajectory ($Z$), and final-Z implicit decoder remain strictly target-free. Target $T1\text{ce}$ and segmentation labels enter exclusively post-inference in the training-only Gate-E/auxiliary objective. In evaluation, the model runs purely target-free.
2. **NIfTI Geometry & Affine Integrity**: Source arrays in `[X, Y, Z]` are mapped to tensor order `[D, H, W] == [Z, Y, X]` via contiguous transpose. Affines are strictly validated (qform/sform consistency, homogeneity, non-singularity, positive spacing) and preserved as physical `[w, h, d] -> [x, y, z]` RAS mappings. Evaluation prediction export reconstructs NIfTI arrays in `[X, Y, Z]` with identical affine matrices.
3. **Split Determinism & Held-Out Payload Boundary**: The pre-split structural inventory uses filesystem metadata exclusively without decoding NIfTI payloads. Subject splitting is ranked by $\text{SHA256}(\text{seed}:\text{subject\_id})$ with exact largest-remainder apportionment and persists a 64-character `split_hash`. Preflight and training validate NIfTI payloads exclusively for active train/val cohorts; held-out test subject payloads remain untouched until evaluation.
4. **Checkpoint & Artifact Atomicity**: Checkpoint saves (`last_train.pt`, `best_model.pt`) and artifact JSON files are written through temporary files in the destination directory before atomic POSIX rename, preventing corrupted writes. Resuming strictly verifies split hash match and restores model, optimizer, scaler, and full RNG states (Python, NumPy, PyTorch CPU/CUDA).

---

## 2. System Architecture & Component Mapping

| Component | Source File | Contract & Responsibility |
| :--- | :--- | :--- |
| **Full-Volume BraTS21 Adapter** | `src/smagm/data/brats21_point_guided.py` | Discovers NIfTI subjects, structural inventory, NIfTI loading, affine validation, observation-only brain mask, masked normalization, sample creation, batch collation, deterministic splitting. |
| **Atomic Checkpoint Serialization** | `src/smagm/features/point_guided/baseline_checkpoint.py` | Atomic temporary-file saves, RNG state capture/restore, training resume checkpoint (`point-guided-training-resume-v1`), clean baseline inference checkpoint (`point-guided-gate-f-baseline-v1`). |
| **Training Pipeline & Orchestration** | `src/smagm/training/point_guided.py` | Configuration validation, DDP initialization, preflight, metric range verification, epoch execution, log/CSV/JSON persistence, resume coordination. |
| **Training CLI** | `src/smagm/cli/point_guided_train.py` | CLI parsing, config overrides, preflight invocation, training launch. |
| **Held-Out Evaluation CLI** | `src/smagm/cli/point_guided_eval.py` | Split resolution from checkpoint provenance, target-free inference, metric computation, prediction NIfTI export, aggregate report generation. |
| **Baseline Inference Policy** | `src/smagm/features/point_guided/baseline_inference.py` | Target-free inference execution, exact-no-revisit routing, checkpoint metadata definition, strict checkpoint validation. |
| **Baseline Metrics** | `src/smagm/features/point_guided/baseline_metrics.py` | Target-after-inference MAE, PSNR, 3-D SSIM, and 3-class coarse semantic Dice metrics. |

---

## 3. Cohort Discovery, Filesystem Boundary, and Structural Eligibility

### 3.1 Subject ID & Directory Naming
- Subject directories must strictly match `BRATS21_POINT_GUIDED_SUBJECT_PATTERN = re.compile(r"^BraTS2021_(?P<number>\d{5})$")`.
- `discover_point_guided_subjects` rejects non-conforming directories fail-closed with `BraTS21PointGuidedValidationError`.
- `discover_point_guided_subject` inspects `.nii` / `.nii.gz` filenames, requires exact single occurrences of `t1`, `t2`, `flair`, at most single occurrences of `t1ce` and `seg`, and records unrecognized NIfTIs in `unknown_nifti_files`.

### 3.2 Pre-Split Structural Eligibility
- `structural_inventory_point_guided_subjects` executes before subject splitting.
- **Payload Safety**: It never calls `nibabel`, opens file handles for reading image headers, or memory-maps arrays. It performs checks only with:
  - `path.exists()`
  - `path.is_file()`
  - `path.stat().st_size > 0`
- Classifies structural failures into explicit exclusion reasons: `missing_file`, `empty_file`, `duplicate_file`, `not_regular_file`, `stat_failed`.
- Emits a structural inventory with SHA-256 `inventory_hash` over canonical JSON.

---

## 4. Deterministic Subject Splitting, Provenance Hashing, and Held-Out Boundary

### 4.1 Split Generation Algorithm
- Function: `deterministic_subject_split(subjects, *, split_fractions=(0.8, 0.1, 0.1), seed=20260813, ...)`
- **Ranking**: Subjects are sorted by `(hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).hexdigest(), subject_id)`.
- **Allocation**: Largest-remainder method (Hare-Niemeyer) allocates integer counts matching split proportions exactly, with fractional tie-breaking.
- **Caps**: Per-split caps (`max_train_subjects`, `max_val_subjects`, `max_test_subjects`) retain the top-ranked members; remainder are routed to `excluded_subject_ids`.
- **Split Hash**: Computed over canonical JSON payload:
  ```json
  {
    "all_subject_ids": ["..."],
    "assignments": {"BraTS2021_00000": "train", "...": "..."},
    "caps": {"train": null, "val": null, "test": null},
    "fractions": [0.8, 0.1, 0.1],
    "seed": 20260813,
    "version": "brats21_point_guided_split_v1"
  }
  ```

### 4.2 Held-Out Test Payload Isolation
- In training and preflight (`_prepare_structurally_eligible_split`):
  - Structural inventory identifies eligible IDs.
  - Split is created / loaded.
  - Active subject cohort: `active_subject_ids = tuple(dict.fromkeys((*train_ids, *val_ids)))`.
  - `inventory_point_guided_subjects(..., subject_ids=active_subject_ids)` validates NIfTI payloads only for train and validation cohorts.
  - **Test cohort NIfTIs are never opened during preflight or training**, logged as `test_payload_validation: "not_performed"`.
- In evaluation (`point_guided_eval.py`):
  - `resolve_split_file(checkpoint, split_file)` resolves the exact `split.json` from `<checkpoint_dir>/../../split.json` (or explicit CLI argument).
  - `_load_split` verifies complete, disjoint partition coverage of all discovered subjects and checks `split_hash`.

---

## 5. NIfTI Loading, Geometry, Orientation, and Tensor Transpose

### 5.1 Affine and Orientation Validation (`_read_nifti`)
- Reads NIfTI via `nibabel.load(str(path), mmap=True)`.
- Validates 3-D non-empty dimensions, numeric finite data (`np.isfinite(data).all()`).
- Resolves affine with strict qform/sform validation:
  - If both `qcode > 0` and `scode > 0`, verifies `np.allclose(qform, sform, atol=1e-4)`. Ambiguities raise `BraTS21PointGuidedValidationError`.
  - Verifies homogeneous bottom row `[0, 0, 0, 1]`.
  - Verifies non-singularity: $|\det(A_{:3,:3})| > 10^{-8}$.
  - Derives positive column norms for voxel spacing.
  - Validates orientation codes from `{"R", "L", "A", "P", "S", "I"}` via `nib.aff2axcodes`.

### 5.2 Axis Transpose Semantics
- Source array order in NIfTI: `[X, Y, Z]`.
- Tensor index order in PyTorch: `[D, H, W] == [Z, Y, X]`.
- `nifti_xyz_to_dhw` performs `np.ascontiguousarray(np.transpose(array, (2, 1, 0)))`.
- Voxel affine mapping: Affine maps index $(x, y, z) = (w, h, d)$ to physical RAS mm $(X, Y, Z)$.
- Multi-modality consistency (`_same_geometry`): T1 reference geometry is compared against T2, FLAIR, T1ce, and segmentation; shape, affine (`atol=1e-4`), and spacing (`atol=1e-5`) must match identically.

### 5.3 Prediction Export to NIfTI
- In `point_guided_eval.py` (`_save_nifti`):
  - Model prediction `[D, H, W]` is transposed back: `array_xyz = prediction.detach().cpu().numpy().transpose(2, 1, 0)`.
  - Saved via `nib.Nifti1Image(array_xyz, affine)`.
  - Affine and spatial grid coordinates align exactly with original source inputs.

---

## 6. Normalization, Input Brain Mask, and Target Boundary

### 6.1 Input-Derived Brain Mask (`derive_input_brain_mask`)
- Union mask derived solely from raw observation values:
  $$\text{mask}_{\text{xyz}} = \bigcup_{m \in \{\text{T1}, \text{T2}, \text{FLAIR}\}} (|V_m| > \text{threshold})$$
- Default threshold: `0.0` (BraTS background is exact 0).
- Rejects all-zero/empty masks.
- Target $T1\text{ce}$ and segmentation are never touched during mask derivation.

### 6.2 Normalization Policies
1. **`masked_robust_01` (MAIN)**:
   - Evaluated on voxels inside `brain_mask`.
   - Computes $p_{\text{low}}$ (1.0%) and $p_{\text{high}}$ (99.0%).
   - Clips to $[p_{\text{low}}, p_{\text{high}}]$ and scales linearly to $[0.0, 1.0]$.
   - Background outside mask is set to exact `0.0`.
   - Modality normalization metadata records clip bounds, percentiles, voxel count, and SHA-256 hash.
2. **`masked_zscore` (Retained Ablation)**:
   - Inside mask: $(x - \mu) / \max(\sigma, \epsilon)$.
   - Background set to `0.0`.

### 6.3 Metric Data Range Validation
- `validate_metric_data_range(normalization_config, supervision)`:
  - For `masked_robust_01`: Verifies `supervision.ssim_data_range == 1.0`.
  - For `masked_zscore`: Prohibits silent default of 1.0; requires explicit `data.normalization.metric_data_range` matching `supervision.ssim_data_range`.
  - Declares metadata `normalization_space` as `"masked_robust_01_[0,1]"` or `"masked_zscore_explicit_metric_range"`.

---

## 7. Batch Collation and Distributed Sharding

### 7.1 Homogeneous Geometry Collation (`collate_point_guided_samples`)
- Rejects heterogeneous batches:
  - Samples must share identical `shape_dhw`.
  - Samples must share identical `affine_xyz_to_ras_mm` (`atol=1e-5`).
  - Samples must share identical segmentation presence.
- Stacks batch tensors:
  - `observations`: `[B, 3, D, H, W]` (float32)
  - `target_t1ce`: `[B, 1, D, H, W]` (float32)
  - `segmentation`: `[B, D, H, W]` (int64) or `None`
  - `brain_mask`: `[B, 1, D, H, W]` (bool)
  - `spacing_xyz_mm`: `[B, 3]` (float32)
  - `voxel_to_ras_mm`: `[B, 4, 4]` (float32)
  - `subject_ids`: `tuple[str, ...]` (len $B$, unique)
  - `normalization_metadata`: `tuple[Mapping, ...]` (len $B$)

### 7.2 Distributed Evaluation Sampler (`DistributedEvalSampler`)
- Standard `DistributedSampler` pads cohorts to equal batch counts per rank, which would cause duplicate counting in validation/evaluation metrics.
- `DistributedEvalSampler` implements strided non-padded sharding:
  $$\text{indices}(\text{rank}) = \{ i \mid 0 \le i < N, \, i \equiv \text{rank} \pmod{\text{world\_size}} \}$$
- Partitions the validation cohort disjointly across ranks without duplication or omission.

---

## 8. Checkpoint Serialization, Atomicity, and Strict Load Validation

### 8.1 Atomic Save Operation (`_atomic_torch_save` / `_atomic_json`)
- Checkpoints (`.pt`) are written using `tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False)`.
- Replaced onto destination path via `temporary.replace(destination)`.
- Cleanup handled via `try ... finally: temporary.unlink(missing_ok=True)`.
- JSON files use `path.with_name(f".{path.name}.tmp")` before rename.
- Atomic filesystem renames on POSIX guarantee zero partial or corrupted checkpoint files.

### 8.2 Training Resume Checkpoint (`save_training_resume_checkpoint`)
- **Schema**: `point-guided-training-resume-v1`
- **Payload**:
  - `epoch`: `int`
  - `global_step`: `int`
  - `model_state_dict`: `dict`
  - `optimizer_state_dict`: `dict`
  - `scaler_state_dict`: `dict | None`
  - `best_validation_reconstruction_loss`: `float`
  - `training_config`: `dict`
  - `split_hash`: `str` (64-char SHA-256)
  - `rng_state`: captured Python, NumPy, PyTorch CPU, and CUDA RNG states
  - `metadata`: run metadata dictionary
- **Load Verification (`load_training_resume_checkpoint`)**:
  - Validates schema and exact required key set.
  - Verifies `payload["split_hash"] == expected_split_hash`; mismatch raises `ValueError`.
  - Restores model (`strict=True`), optimizer, scaler, and full RNG states.

### 8.3 Clean Baseline Inference Checkpoint (`save_clean_inference_checkpoint`)
- **Schema**: `point-guided-gate-f-baseline-v1`
- **Payload**:
  - `metadata`: `baseline_checkpoint_metadata(model)` including `model_config`, `trajectory_config`, `decoder_architecture: "96->64->32->1"`, `gate_e_architecture: "target-after-inference objective"`.
  - `state_dict`: `model.state_dict()`.
- **Load Validation (`load_validated_baseline_checkpoint`)**:
  - Loads with `torch.load(..., weights_only=True)`.
  - Strictly compares stored metadata with `baseline_checkpoint_metadata(model)`.
  - Loads weights with `model.load_state_dict(state_dict, strict=True)`.

---

## 9. Artifact Persistence and Traceability

### 9.1 Training Run Directory Structure
```text
<output_root>/<run_name>/
├── config.json                     # Resolved configuration, settings, supervision
├── split.json                      # Full split assignments, seed, caps, fractions, split_hash
├── structural_inventory.json       # Pre-split filesystem discovery & exclusions
├── active_payload_inventory.json   # Validated payloads for train/val cohorts
├── subject_inventory.json          # Composite inventory
├── environment.json                # Hostname, git HEAD, Python/PyTorch/CUDA versions
├── train.jsonl                     # Line-delimited JSON metrics per epoch
├── metrics.csv                     # Tabular metrics CSV with header
├── summary.json                    # Status, best validation loss, last metrics, split_hash
└── checkpoints/
    ├── best_model.pt               # Clean baseline inference checkpoint
    └── last_train.pt               # Full resumable training checkpoint
```

### 9.2 Evaluation Artifacts
```text
<output_dir>/
├── per_subject_metrics.json        # Per-subject MAE, PSNR, SSIM, Dice, trajectory diagnostics
├── aggregate_metrics.json          # Mean metrics, semantic Dice, stop reason histogram
├── trajectory_diagnostics.json     # Detailed routing diagnostics per subject
├── evaluation_metadata.json        # Checkpoint path, split file, split_hash, training run dir
└── predictions/
    └── BraTS2021_XXXXX_t1ce_pred.nii.gz # NIfTI volume with original affine
```

---

## 10. Audit Verification Matrix

| Verification Aspect | Audit Result | Evidence / Implementation Reference |
| :--- | :--- | :--- |
| **Target Isolation** | **VERIFIED PASS** | `derive_input_brain_mask` uses only observation stack; model forward methods take only observations and mask; T1ce/seg enter only in post-inference objectives. |
| **Geometry Consistency** | **VERIFIED PASS** | `nifti_xyz_to_dhw` and `_save_nifti` strictly maintain $[X,Y,Z] \leftrightarrow [D,H,W]$ transpose; affines validated for qform/sform consistency, homogeneity, non-singularity. |
| **Split Determinism** | **VERIFIED PASS** | $\text{SHA256}(\text{seed}:\text{subject\_id})$ ranking with largest-remainder allocation; 64-char `split_hash` tracked throughout training, resume, and evaluation. |
| **Held-Out Protection** | **VERIFIED PASS** | Structural inventory is pre-split metadata only; test payloads remain unread during training and preflight (`test_payload_validation: "not_performed"`). |
| **Batch Homogeneity** | **VERIFIED PASS** | `collate_point_guided_samples` enforces identical volume shapes and affine transformations across batch items. |
| **Atomic Checkpoints** | **VERIFIED PASS** | `_atomic_torch_save` and `_atomic_json` use temporary files in target directory before atomic POSIX rename. |
| **Strict Resume** | **VERIFIED PASS** | Resume enforces exact `split_hash` match, full RNG state restoration, and strict parameter loading. |
| **Metric Range Alignment** | **VERIFIED PASS** | `validate_metric_data_range` enforces range compatibility between normalization policy (`masked_robust_01` vs. `masked_zscore`) and supervision SSIM data range. |
| **Production Mutation** | **VERIFIED ZERO MUTATION** | Audit was executed in read-only mode; zero production code, tests, configs, or plans were modified. |

---

## 11. Conclusion & Certification

The data persistence, NIfTI geometry, normalization, split provenance, batch collation, checkpoint serialization, and evaluation artifact architecture at frozen `main` commit `0efeb94af72ffa067769e19afcd19ad358feefd2` are **fully verified, mathematically consistent, fail-closed, and compliant with repository authorities**. Zero defects, data leaks, or provenance violations were identified.
