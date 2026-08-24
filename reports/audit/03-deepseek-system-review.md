# System-Wide Architectural & Engineering Audit Review: Point-Guided MRI Pipeline

## 1. Audit Metadata & Role Disclosure

- **Audit Target Commit (HEAD):** `0efeb94af72ffa067769e19afcd19ad358feefd2` (`main`)
- **Upstream Target:** `origin/main` at `0efeb94af72ffa067769e19afcd19ad358feefd2`
- **Role Disclosure:** Independent System-Wide Auditor (Phase-3 Reviewer). DeepSeek is unavailable in this local runtime environment; this review is executed by the explicitly disclosed **GPT-OSS 120B** substitute acting under the locked audit authority defined in `reports/audit/01-sol-plan.md`.
- **Pre-existing Dirty Files Preserved:**
  - Modified: `.DS_Store`, `src/.DS_Store`, `tests/.DS_Store`
  - Untracked: `configs/.DS_Store`, `docs/.DS_Store`, `docs/architecture/point_guided_reward_cost_trajectory.html`, `scripts/.DS_Store`
- **Mutation Boundary:** Written strictly to `reports/audit/03-deepseek-system-review.md`. All production code, unit tests, configurations, plans, and existing documentation remain 100% read-only.
- **Authority Hierarchy:**
  $$\text{AGENTS.md} \succ \text{PLAN\_GATE\_F\_G.md} / \text{PLAN\_GATE\_C\_D\_E.md} \succ \text{PLAN.md} \succ \text{CODEGRAPH.json} / \text{CODEBASE.md} \succ \text{README.md} / \text{docs/} / \text{quality/}$$

---

## 2. Executive Synthesis & High-Level System Verdict

An exhaustive, cross-subsystem audit was conducted across all 7 worker reports (`agy-a` through `agy-g`) and the complete frozen codebase at commit `0efeb94`. 

### 2.1 Core Architectural Assessment
1. **Scientific Decoupling & Target Isolation (VERIFIED PASS):**
   The core point-guided tensor transformations strictly enforce the scientific boundary. The input observation stack ($T_1, T_2, \text{FLAIR}$ ordered $[B, 3, D, H, W]$), input-derived union brain mask, deterministic quasi-uniform point placement, observation-only bounded displacement ($\le 2\,\text{mm}$), compact 4-mm sparse Partition of Unity (PoU), static base planes ($B$), stationary 2-level Haar wavelet spectral anchor ($A$), 168-d point spectral evidence ($f_{\text{spec}}$), dynamic tri-plane trajectory ($Z_0 \to Z_K$), and final-$Z$ implicit decoder remain strictly target-free. Target $T_{1\text{ce}}$ and ground-truth segmentation enter exclusively after inference in the training-only Gate-E composite loss and auxiliary semantic cross-entropy loss. Generic `forward()` remains fail-closed.
2. **Legacy Isolation & Reward Revert Integrity (VERIFIED PASS):**
   Legacy 3DGS packages (`src/smagm/anchors/`, `fields/`, `memory/`, `routing/`, `reconstruction/`, `render/`, `gaussians.py`, `state.py`, `brats21.py`, `trainer.py`) are completely decoupled and unimported by the active point-guided pipeline. The reward revert in commit `39a39d3` cleanly restored the 126-d descriptor ($96 + 3 + 24 + 3$) and the 8,193-parameter `RewardNet` with zero lingering references to the failed 222-d paired candidate architecture.
3. **Operational & Concurrency Readiness (FAILED / DEFECTIVE):**
   While the mathematical and scientific frontend is coherent, the system possesses critical operational defects across distributed synchronization, runtime persistence, configuration consistency, and launcher scripts. In particular:
   - **Deadlock Risk in DDP:** Distributed failure teardown invokes an unconditional barrier in `destroy_distributed()`, which deadlocks surviving ranks when any single rank raises an exception.
   - **Run Collisions & Race Conditions:** Training and evaluation run directories rely on 1-second timestamp resolution or reusable static names without filesystem locks, while atomic JSON helpers share fixed sibling temporary names (`.<name>.tmp`), causing `FileNotFoundError` and silent file clobbering under concurrent or repeated execution.
   - **Parameter Count & Configuration Drift:** `AGENTS.md` and `PLAN.md` specify a locked 1,419-parameter offset predictor (`offset_hidden_channels=12`), but all server profile configs specify `offset_hidden_channels=128` (15,107 parameters, expanding total trainables from 69,723 to 83,411).
   - **Provenance & Integrity Gaps:** Split hashes are accepted as opaque 64-character strings without recalculating digests; structural inventory silently drops malformed directory names before eligibility filtering; clean inference checkpoints do not persist split hashes; evaluation metadata hardcodes `git_head: null`.
   - **Shell Environment Incoherence:** Launchers mix `POINT_GUIDED_PYTHON`, bare `python`, and bare `torchrun` with inconsistent conda defaults, preventing reliable execution across diverse host environments.

---

## 3. End-to-End Coherent Product Trace

```text
[BraTS2021 Directory]
   │
   ├─► 1. Structural Inventory: checks file existence/size without decoding NIfTI payloads (brats21_point_guided.py)
   │
   ├─► 2. Deterministic Split: SHA256(seed:subject_id) ranking + Hare-Niemeyer allocation -> split.json
   │
   ├─► 3. Data Loading & Geometry: NIfTI [X,Y,Z] -> Tensor [D,H,W], qform/sform validation, affine preservation
   │
   ├─► 4. Preprocessing: Union Brain Mask (T1|T2|FLAIR > 0) + masked_robust_01 [0.0, 1.0] intensity normalization
   │
   ▼
[PointGuidedMRIModel: Shared Frontend Traversal]
   │
   ├─► 5. Backbone: 1 frozen MedicalNet ResNet10 traversal -> intermediate features (medicalnet_resnet10.py)
   │
   ├─► 6. Semantic Prior: 3 coarse classes (normal, edema, core candidate) (semantic_prior.py)
   │
   ├─► 7. Points & Refinement: Deterministic 2048 initial points -> bounded displacement <= 2mm (refinement.py)
   │
   ├─► 8. Partition of Unity: Sparse 4-mm spheres, quadratic spatial kernel, L1 semantic affinity (pou.py)
   │
   ├─► 9. Base Planes B: Axis-conditioned projection -> Bxy [B,C,H,W], Bxz [B,C,D,W], Byz [B,C,D,H]
   │
   ├─► 10. Spectral Anchor A: Fixed 2-level SWT-Haar + shared 1x1 Conv (64->8) -> Axy, Axz, Ayz (56 channels each)
   │
   └─► 11. Point Spectral Evidence: Geometry-aware bilinear query -> q (24-d) -> reliability alpha -> f_spec (168-d)
   │
   ▼
[Gate-C Adaptive Trajectory & Gate-D Implicit Decoding]
   │
   ├─► 12. Dynamic Initialization: Z_0 = B (state_init.py)
   │
   ├─► 13. Route Selection: 126-d RewardNet + Travel/Overlap/Step Costs + ExactNoRevisitPolicy -> Greedy argmax
   │
   ├─► 14. Dynamic Update: UpdateNet (270->96) + 4-mm physical Gaussian write-back -> Z_K (updater.py, writeback.py)
   │
   └─► 15. Gate-D Decoder: Geometry-aware chunked query of Z_K -> 96->64->32->1 SiLU MLP -> Prediction [B,1,D,H,W]
   │
   ▼
[Execution Branch: Training vs. Evaluation]
   │
   ├─► [TRAINING BRANCH]
   │     ├─► Gate-E Supervision: Charbonnier + 3D SSIM + grad + counterfactual SmoothL1 + monotonic/delta regularizers
   │     ├─► Semantic Auxiliary: CE loss against BraTS segmentation (training only)
   │     ├─► Optimizer: AdamW over 8 authorized modules (MedicalNet backbone & SWT filters strictly frozen)
   │     └─► Persistence: Atomic last_train.pt (resume) & best_model.pt (clean inference)
   │
   └─► [EVALUATION BRANCH]
         ├─► Baseline Inference: eval/no-grad, exact-no-revisit, single final-Z decode (baseline_inference.py)
         ├─► Post-Inference Metrics: MAE, PSNR, 3D SSIM over brain mask; Coarse Semantic Dice
         ├─► NIfTI Prediction Export: Transpose [D,H,W] -> [X,Y,Z], attach original 4x4 affine
         └─► Summary Artifacts: per_subject_metrics.json, aggregate_metrics.json, trajectory_diagnostics.json, evaluation_metadata.json
```

---

## 4. Cross-Subsystem Assumption Disagreements & Root Cause Analysis

### 4.1 Disagreement 1: Offset-Predictor Parameter Count & Width Contract
- **Subsystems in Conflict:** Governance Authority (`AGENTS.md`, `PLAN.md`) vs. Optimizer Unit Tests (`test_baseline_training.py`) vs. Production Server Configs (`configs/training/point_guided_brats21_*.json`).
- **Code Locations:**
  - `AGENTS.md:200-207` & `PLAN.md:1476-1483`: Locks `point_refiner.offset_predictor` to **1,419 parameters**.
  - `src/smagm/features/point_guided/baseline_training.py:118`: Annotates `"human Gate-F resolution: trainable; 1,419 parameters; bounds unchanged"`.
  - `tests/features/point_guided/test_baseline_training.py:28, 57`: Constructs test model with `offset_hidden_channels=12` and asserts `1,419` parameters.
  - `configs/training/point_guided_brats21_4070.json:16`, `2xa4000.json:16`, `overfit.json:16`: Specifies `"offset_hidden_channels": 128`.
- **Root Cause & Impact:**
  With `offset_hidden_channels=128`, the offset predictor contains $15,107$ parameters, expanding total trainable parameters from $69,723$ to $83,411$. `resolve_parameter_ownership()` checks module identity but not parameter count or channel width. Consequently, a server training job using checked-in profiles will silently optimize a model ten times larger in its refinement head than the locked Gate-F specification.

### 4.2 Disagreement 2: Split Hash Verification vs. Opaque String Check
- **Subsystems in Conflict:** Dataset Provenance Hashing (`brats21_point_guided.py`) vs. Split Loading in Evaluation (`point_guided_eval.py`) & Training (`point_guided.py`).
- **Code Locations:**
  - `src/smagm/data/brats21_point_guided.py:930-947`: Computes a deterministic SHA-256 digest over canonical JSON containing all subject assignments, seed, fractions, and caps.
  - `src/smagm/cli/point_guided_eval.py:73-75`:
    ```python
    split_hash = payload.get("split_hash")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        raise ValueError("split file must contain a 64-character split_hash")
    ```
  - `src/smagm/training/point_guided.py:1082-1084`: Implements the identical length-only check.
  - `src/smagm/features/point_guided/baseline_inference.py:269-282`: `baseline_checkpoint_metadata()` does not record `split_hash`.
- **Root Cause & Impact:**
  `_load_split()` and `_resolve_split()` accept arbitrary 64-character dummy strings (e.g. `"a" * 64`) without re-hashing the partition. Furthermore, clean baseline checkpoints have no split binding in their metadata, allowing a checkpoint trained on split A to be evaluated against split B without detection.

### 4.3 Disagreement 3: Structural Inventory Filtering vs. Discovery Failure Logging
- **Subsystems in Conflict:** Pre-split Structural Inventory vs. Legacy/Active Inventory & Error Classification.
- **Code Locations:**
  - `src/smagm/data/brats21_point_guided.py:729-733`:
    ```python
    directories = tuple(
        directory
        for directory in sorted(item for item in source_root.iterdir() if item.is_dir())
        if BRATS21_POINT_GUIDED_SUBJECT_PATTERN.fullmatch(directory.name) is not None
    )
    ```
  - `src/smagm/data/brats21_point_guided.py:859-865`: Explicitly flags non-matching directories as `OTHER_INVALID`.
- **Root Cause & Impact:**
  `structural_inventory_point_guided_subjects()` filters directories with regex before constructing the inventory. Malformed folder names (e.g. `BraTS21_00001`, `BraTS2021_00001_corrupt`) are silently omitted from `structural_inventory.json` rather than recorded in `excluded_subjects`, masking filesystem anomalies during audit.

### 4.4 Disagreement 4: Strict Checkpoint Load Transactionality
- **Subsystems in Conflict:** Checkpoint Verification Contract vs. PyTorch In-Place `load_state_dict`.
- **Code Locations:**
  - `src/smagm/features/point_guided/baseline_inference.py:285-304`: Validates top-level keys and metadata, then directly executes `model.load_state_dict(state_dict, strict=True)`.
- **Root Cause & Impact:**
  `torch.nn.Module.load_state_dict` mutates parameters sequentially in-place. If a later tensor has a shape mismatch or corruption, the exception is raised after earlier tensors (e.g. RewardNet bias) have already been overwritten in the live model. A caller catching the exception is left with a corrupted hybrid model.

### 4.5 Disagreement 5: Gate-E Context Ownership & Cross-Model Leakage
- **Subsystems in Conflict:** Model Forward API (`model.py`) vs. Objective Computation (`training_objective.py`).
- **Code Locations:**
  - `src/smagm/features/point_guided/model.py:419-438`: `compute_training_objective()` verifies that `self` has a trajectory and decoder, but forwards `context` directly to `_compute_training_objective(context, ...)`.
  - `src/smagm/features/point_guided/training_objective.py:305-308`: Extracts `trajectory = context._trajectory` and `decoder = context._decoder`.
- **Root Cause & Impact:**
  `GateESupervisionContext` does not bind an instance identifier of its producing model. If context from Model A is passed to Model B's `compute_training_objective()`, the loss and backward pass optimize Model A's modules while Model B receives no gradients, violating module ownership boundaries.

### 4.6 Disagreement 6: Distributed Failure Teardown Barrier Deadlock
- **Subsystems in Conflict:** DDP Process Group Teardown vs. Rank-Divergent Error Handling.
- **Code Locations:**
  - `src/smagm/training/point_guided.py:268-272`:
    ```python
    def destroy_distributed(context: DistributedContext) -> None:
        if context.is_distributed and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    ```
  - `src/smagm/training/point_guided.py:1605-1606`: Calls `destroy_distributed(context)` in an unconditional `finally:` block.
- **Root Cause & Impact:**
  If rank 1 crashes (e.g. CUDA OOM or data corruption), it enters `finally:` and blocks indefinitely inside `torch.distributed.barrier()`. Rank 0 may still be waiting inside an epoch all-reduce, stop broadcast, or logging block. The job hangs until the distributed timeout expires rather than failing immediately.

### 4.7 Disagreement 7: Run Directory Collisions & Unsafe Temporary JSON Filenames
- **Subsystems in Conflict:** Training/Evaluation Output Management vs. File Persistence.
- **Code Locations:**
  - `src/smagm/training/point_guided.py:1289-1295`: Run directory timestamps use 1-second resolution (`%Y%m%dT%H%M%SZ`); `mkdir(exist_ok=True)` performs no exclusive lock.
  - `scripts/point_guided_overfit_4070.sh:8, 29-36`: Reuses fixed default `RUN_NAME=point-guided-overfit-4070`.
  - `src/smagm/training/point_guided.py:217-221` & `src/smagm/cli/point_guided_eval.py:28-33`: `_atomic_json()` creates temporary files using fixed names `path.with_name(f".{path.name}.tmp")`.
- **Root Cause & Impact:**
  Concurrent jobs targeting the same output directory or started in the same second race on `.<name>.tmp`, producing `FileNotFoundError` during `replace` and clobbering logs, checkpoints, and summary JSONs.

### 4.8 Disagreement 8: Launch Wrapper Interpreter Incoherence
- **Subsystems in Conflict:** Shell Launch Scripts (`scripts/point_guided_*.sh`) vs. Host Python Environment.
- **Code Locations:**
  - `scripts/point_guided_train_4070.sh:8`: Sets `: "${POINT_GUIDED_PYTHON:=/home/aidev/miniconda3/envs/smagm-a4000/bin/python}"` and uses `$POINT_GUIDED_PYTHON`.
  - `scripts/point_guided_preflight.sh:16, 28`: Uses bare `python`.
  - `scripts/point_guided_train_2xa4000.sh:16, 26`: Uses bare `python` for probe and bare `torchrun` for launch.
  - `scripts/point_guided_eval.sh:27, 39`: Uses bare `python`.
- **Root Cause & Impact:**
  On multi-python server environments where `python` defaults to a system interpreter without PyTorch/CUDA, preflight, multi-GPU training, and evaluation fail during pre-launch probes. Hardcoded conda paths fail on systems with alternative install locations.

### 4.9 Disagreement 9: DDP Resume RNG State & Early Stopping Patience Reset
- **Subsystems in Conflict:** Checkpoint Persistence (`baseline_checkpoint.py`) vs. Trainer Resume (`training/point_guided.py`).
- **Code Locations:**
  - `src/smagm/training/point_guided.py:1533-1565`: Only rank 0 captures RNG state (`save_training_resume_checkpoint`).
  - `src/smagm/training/point_guided.py:1380-1392`: On resume, all ranks restore rank 0's RNG state, collapsing rank-specific seeds.
  - `src/smagm/training/point_guided.py:1408-1412`: `patience_count = 0` is reset upon resume; patience state is not saved in checkpoint.
  - `src/smagm/training/point_guided.py:1296-1333`: Overwrites run directory `config.json` with current CLI arguments without checking against saved `training_config`.
- **Root Cause & Impact:**
  Resuming a distributed run compromises RNG stream independence across ranks, extends early stopping beyond the configured patience budget, and permits silent drift in training hyperparameters.

### 4.10 Disagreement 10: Evaluation Metadata Provenance Incompleteness
- **Subsystems in Conflict:** Training Provenance Logging vs. Evaluation Metadata Generation.
- **Code Locations:**
  - `src/smagm/cli/point_guided_eval.py:230`: Persists `"git_head": None` as a hardcoded literal in `evaluation_metadata.json`.
  - `src/smagm/training/point_guided.py:224-228`: Contains working `_git_head()` implementation.
- **Root Cause & Impact:**
  Evaluation summary JSON loses the git commit provenance link, forcing operators to inspect transient console logs.

---

## 5. Worker Findings Validation, Rejection, and Deduplication Matrix

Every finding from worker reports `agy-a` through `agy-g` was independently validated against the frozen codebase.

| Original Finding ID | Worker Source | Severity | System Review Status | Unified Categorization & Actionable Summary |
| :--- | :--- | :---: | :---: | :--- |
| **`AGY-A-FIND-001`** | AGY-A | P2 | **VALIDATED** | **`SYS-DEF-001` (Hygiene):** `CODEGRAPH.json` task `baseline_training` lists 4 non-existent drafting files (`baseline_data.py`, `scripts/point_guided_baseline.py`, etc.). Clean up task paths. |
| **`AGY-A-FIND-002`** | AGY-A | P3 | **VALIDATED (DUP)** | **`SYS-DEF-002` (Deduplicated with AGY-E-002):** Orphaned duplicate `point_guided_metrics.py` unreferenced by production modules. Consolidate on `baseline_metrics.py`. |
| **`AGY-A-FIND-003`** | AGY-A | P3 | **VALIDATED** | **`SYS-DOC-001` (Exports):** `features/point_guided/__init__.py` exports only Phase 1–5 symbols and omits completed Gates A–G. |
| **`AGY-A-FIND-004`** | AGY-A | DOC | **VALIDATED** | **`SYS-DOC-002` (Doc Drift):** `README.md`, `POINT_GUIDED_FRONTEND.md`, and `quality/` describe Gates F/G as inactive; harmonize with `AGENTS.md`. |
| **`AGY-A-FIND-005`** | AGY-A | INFO | **VALIDATED** | **`SYS-INFO-001` (Dead Code):** `interfaces.py` contains unused abstract base classes from early drafting. |
| **`AGY-A-FIND-006`** | AGY-A | INFO | **VALIDATED** | **`SYS-INFO-002` (Architecture):** Top-level `__init__.py` files export only legacy symbols; intended modular isolation. |
| **`AGY-B-FIND-001`** | AGY-B | P1 | **VALIDATED** | **`SYS-DEF-003` (Integrity):** `load_validated_baseline_checkpoint` mutates live model in-place before strict shape check exceptions. Preflight state dict or load into fresh instance. |
| **`AGY-B-FIND-002`** | AGY-B | P2 | **VALIDATED** | **`SYS-DEF-004` (Ownership):** `compute_training_objective` accepts contexts produced by foreign model instances. Bind instance ID to context. |
| **`AGY-B-FIND-003`** | AGY-B | P2 | **VALIDATED** | **`SYS-DEF-005` (State):** Gate-G wrapper `self.train(was_training)` resets fine-grained child submodule training modes. Record per-module mode map. |
| **`AGY-B-FIND-004`** | AGY-B | P3 | **VALIDATED** | **`SYS-DEF-006` (Typing):** Float64 physical coordinates query Float32 dynamic state producing raw PyTorch matmul error in decoder MLP. Enforce dtype check. |
| **`AGY-C-CERT`** | AGY-C | PASS | **VALIDATED (QUALIFIED)**| AGY-C verified NIfTI affine transpose $[X,Y,Z] \leftrightarrow [D,H,W]$, target decoupling, and atomic file replacement. Note that split hash checking was opaque string validation (see AGY-G-001). |
| **`AGY-D-FIND-001`** | AGY-D | P1 | **VALIDATED** | **`SYS-DEF-007` (Distributed Deadlock):** `destroy_distributed()` unconditionally calls `barrier()` in `finally:`, causing surviving ranks to hang on single-rank failure. |
| **`AGY-D-FIND-002`** | AGY-D | P1 | **VALIDATED** | **`SYS-DEF-008` (Concurrency):** Run directories lack exclusive reservation (1-sec timestamp); `_atomic_json()` fixed temporary names cause collision `FileNotFoundError`. |
| **`AGY-D-FIND-003`** | AGY-D | P2 | **VALIDATED** | **`SYS-DEF-009` (Reproducibility):** DDP resume restores rank 0 RNG state to all ranks, collapsing rank-specific seeds. |
| **`AGY-D-FIND-004`** | AGY-D | P2 | **VALIDATED** | **`SYS-DEF-010` (Provenance):** Resume resets early stopping patience to 0, does not validate saved training settings, and overwrites run `config.json`. |
| **`AGY-D-FIND-005`** | AGY-D | P2 | **VALIDATED** | **`SYS-DEF-011` (DDP Sampler):** `DistributedSampler` default padding duplicates subjects in non-divisible training splits. Document or configure `drop_last`. |
| **`AGY-D-FIND-006`** | AGY-D | P2 | **VALIDATED** | **`SYS-DEF-012` (Resource Cleanup):** Exceptions bypass `wandb_run.finish()`, leave un-cleared optimizer gradients, and do not shut down persistent DataLoader workers. |
| **`AGY-D-FIND-007`** | AGY-D | P1 | **VALIDATED** | **`SYS-DEF-013` (Parameter Contract):** Server profile configs set `offset_hidden_channels=128` ($15,107$ params) violating locked 1,419-param contract ($h=12$). |
| **`AGY-E-FIND-001`** | AGY-E | P3 | **VALIDATED** | **`SYS-DEF-014` (Provenance):** `evaluation_metadata.json` hardcodes `"git_head": None`. Connect to `_git_head()`. |
| **`AGY-E-FIND-002`** | AGY-E | P3 | **VALIDATED (DUP)** | **`SYS-DEF-002` (Deduplicated with AGY-A-002):** Orphaned `point_guided_metrics.py`. |
| **`AGY-E-FIND-003`** | AGY-E | INFO | **VALIDATED** | **`SYS-INFO-003` (Operational):** Evaluation executes single-pass subject loop without incremental per-subject `.jsonl` resume. |
| **`AGY-E-FIND-004`** | AGY-E | INFO | **VALIDATED** | **`SYS-INFO-004` (Convention):** `semantic_dice()` evaluates to 1.0 when both predicted and target class masks are empty (standard medical imaging convention). |
| **`AGY-F-FIND-001`** | AGY-F | P2 | **VALIDATED** | **`SYS-OPS-001` (CI Gaps):** CI only tests CPU subset without `nibabel`, skipping 19 data tests and all server-pipeline operational checks. |
| **`AGY-F-FIND-002`** | AGY-F | P1 | **VALIDATED** | **`SYS-OPS-002` (Launchers):** Launch wrappers inconsistently bind Python interpreter (`POINT_GUIDED_PYTHON` vs bare `python` vs bare `torchrun`). |
| **`AGY-F-FIND-003`** | AGY-F | P2 | **VALIDATED (COLLAPSED)**| **`SYS-DEF-008` (Subsumed into Concurrency):** Evaluation output persistence lacks run directory exclusivity and atomic prediction writes. |
| **`AGY-F-FIND-004`** | AGY-F | P2 | **VALIDATED** | **`SYS-OPS-003` (Dependency Pinning):** CI pins differ from broad ranges in `pyproject.toml` and unpinned server install instructions. |
| **`AGY-G-FIND-001`** | AGY-G | P1 | **VALIDATED** | **`SYS-DEF-015` (Provenance):** Split hashes are validated only for 64-char length without digest recomputation; checkpoint metadata omits split hash binding. |
| **`AGY-G-FIND-002`** | AGY-G | P2 | **VALIDATED** | **`SYS-DEF-016` (Structural Data):** `structural_inventory_point_guided_subjects` regex filters directory list first, silently omitting malformed folders. |
| **`AGY-G-FIND-003`** | AGY-G | P2 | **VALIDATED** | **`SYS-DEF-017` (Artifact Schema):** Resuming training with expanded metric columns appends mismatched rows under existing `metrics.csv` headers. |

---

## 6. Consolidated Defect Catalog by Severity

### 6.1 Critical Severity (P1)

1. **`[SYS-DEF-013]` Server Profile Parameter Count Violation (15,107 vs. 1,419)**
   - **File:** `configs/training/point_guided_brats21_4070.json:16`, `2xa4000.json:16`, `overfit.json:16`
   - **Evidence:** Authority locks `point_refiner.offset_predictor` to 1,419 parameters (`offset_hidden_channels=12`). Server profiles set `offset_hidden_channels=128`, resulting in 15,107 parameters (83,411 total trainables vs. 69,723).
   - **Remediation:** Update server configuration presets to `"offset_hidden_channels": 12` or explicitly obtain authorized governance approval to revise Gate-F locked parameter contracts.

2. **`[SYS-DEF-007]` DDP Process Group Teardown Barrier Deadlock on Failure**
   - **File:** `src/smagm/training/point_guided.py:268-272, 1605-1606`
   - **Evidence:** `destroy_distributed()` executes `torch.distributed.barrier()` inside `finally:`. If rank 1 fails and enters teardown while rank 0 is waiting in an epoch collective, the job hangs indefinitely.
   - **Remediation:** Guard teardown barrier so it only executes when all ranks exit cleanly, or wrap in try/except with a short timeout and call `destroy_process_group()` directly upon error.

3. **`[SYS-DEF-008]` Run Directory Collision & Shared Temporary JSON Name Collisions**
   - **File:** `src/smagm/training/point_guided.py:217-221, 1289-1298`, `src/smagm/cli/point_guided_eval.py:28-33`
   - **Evidence:** 1-second timestamp resolution without `O_EXCL` lock permits duplicate run dirs. `_atomic_json` uses fixed temporary name `path.with_name(f".{path.name}.tmp")`, causing multithreaded/multiprocess collisions (`FileNotFoundError`).
   - **Remediation:** Use `tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")` for JSON writes and enforce UUID/exclusive directory creation.

4. **`[SYS-DEF-015]` Opaque Split Hash Acceptance & Unbound Checkpoint Metadata**
   - **File:** `src/smagm/cli/point_guided_eval.py:73-75`, `src/smagm/training/point_guided.py:1082-1084`, `src/smagm/features/point_guided/baseline_inference.py:269-282`
   - **Evidence:** Split loaders check only `len(split_hash) == 64` without re-hashing JSON contents. Checkpoint metadata omits split hash, permitting unverified cross-cohort evaluation.
   - **Remediation:** Recompute canonical split digest during loading, reject mismatches, and bind `expected_split_hash` into checkpoint metadata.

5. **`[SYS-OPS-002]` Launch Scripts Python & Torchrun Environment Incoherence**
   - **File:** `scripts/point_guided_*.sh`
   - **Evidence:** `preflight.sh`, `2xa4000.sh`, `overfit.sh`, and `eval.sh` call bare `python` and `torchrun` rather than respecting `POINT_GUIDED_PYTHON`. `train_4070.sh` hardcodes a non-portable miniconda path.
   - **Remediation:** Standardize all shell scripts to resolve `PYTHON_BIN="${POINT_GUIDED_PYTHON:-$(which python3)}"`, derive `TORCHRUN_BIN`, verify executable permissions, and probe PyTorch imports uniformly.

6. **`[SYS-DEF-003]` Checkpoint Load In-Place Mutation on Strict Failure**
   - **File:** `src/smagm/features/point_guided/baseline_inference.py:285-304`
   - **Evidence:** Direct call to `model.load_state_dict(..., strict=True)` mutates earlier tensors before raising exceptions on shape/key mismatches.
   - **Remediation:** Pre-validate state dictionary keys, shapes, and dtypes against `model.state_dict()` before calling `load_state_dict`.

---

### 6.2 Major Severity (P2)

7. **`[SYS-DEF-004]` Cross-Model Context Ownership Leakage in Training Objective**
   - **File:** `src/smagm/features/point_guided/model.py:419-438`, `training_objective.py:305-308`
   - **Evidence:** `compute_training_objective` consumes `context._trajectory` and `context._decoder` without verifying `context._trajectory is self.trajectory`.
   - **Remediation:** Add explicit assertions in `compute_training_objective` that context-owned modules match `self`.

8. **`[SYS-DEF-005]` Gate-G Inference Wrapper Silently Resets Submodule Eval Modes**
   - **File:** `src/smagm/features/point_guided/model.py:311-336`
   - **Evidence:** Wrapper records `was_training = self.training` and executes `self.train(was_training)` in `finally:`, turning deliberately frozen/eval submodules back into training mode.
   - **Remediation:** Snapshot and restore module-by-module `training` state dictionary.

9. **`[SYS-DEF-009]` DDP Resume Collapses Per-Rank RNG Streams**
   - **File:** `src/smagm/training/point_guided.py:1380-1392`, `baseline_checkpoint.py:33-56`
   - **Evidence:** Rank 0's RNG state is saved and restored to all ranks upon resume, collapsing distinct rank seeds.
   - **Remediation:** Persist rank-specific RNG payloads or reseed ranks deterministically post-resume (`seed + epoch * world_size + rank`).

10. **`[SYS-DEF-010]` Resume Resets Early Stopping Patience & Overwrites Configuration**
    - **File:** `src/smagm/training/point_guided.py:1296-1333, 1408-1412`
    - **Evidence:** `patience_count` is reset to 0 upon resume; run `config.json` is unconditionally overwritten with current invocation arguments.
    - **Remediation:** Save and restore `patience_count` in resume checkpoints; compare current config against saved `training_config` and warn/reject on incompatible drift.

11. **`[SYS-DEF-011]` Training `DistributedSampler` Pads Non-Divisible Cohorts**
    - **File:** `src/smagm/training/point_guided.py:1004-1020`
    - **Evidence:** `DistributedSampler` without `drop_last=True` pads training cohorts, duplicating subjects on uneven splits (e.g. 965 subjects across 2 GPUs duplicates 1 subject/epoch).
    - **Remediation:** Document sample padding behavior and report effective vs unique sample counts in training logs.

12. **`[SYS-DEF-012]` Unhandled Exception Resource & Logger Teardown Gaps**
    - **File:** `src/smagm/training/point_guided.py:840-847, 1413-1424, 1600-1601`
    - **Evidence:** Exceptions bypass `wandb_run.finish()`, leave accumulated gradients in optimizer, and do not close persistent DataLoader worker pools.
    - **Remediation:** Wrap training loop in `try ... except ... finally:` ensuring `wandb.finish()` and optimizer `zero_grad()` on abort.

13. **`[SYS-DEF-016]` Structural Pre-Split Inventory Silently Drops Malformed Directories**
    - **File:** `src/smagm/data/brats21_point_guided.py:729-733`
    - **Evidence:** Regex pre-filtering omits non-conforming folder names before building discovered/excluded lists.
    - **Remediation:** Enumerate all directories first, recording non-matching folder names as `OTHER_INVALID` exclusions.

14. **`[SYS-DEF-017]` Metrics CSV Header Mismatch Risk on Resume**
    - **File:** `src/smagm/training/point_guided.py:1177-1187, 1406-1409`
    - **Evidence:** Trainer assumes pre-existing `metrics.csv` has matching columns, appending new fields without verifying headers.
    - **Remediation:** Validate existing CSV header columns against current log fields before appending.

15. **`[SYS-OPS-001]` CI Pipeline Skips Real-Data NIfTI & CUDA Execution**
    - **File:** `.github/workflows/ci.yml:18-25`
    - **Evidence:** CI runs CPU pytest without optional `nibabel` extra, skipping 19 data tests and all server-pipeline operational checks.
    - **Remediation:** Add a dedicated CI workflow job testing `.[real-data]` with synthetic NIfTI generation.

16. **`[SYS-OPS-003]` Unpinned Dependency Specifications Across CI and Server Docs**
    - **File:** `pyproject.toml:10-22`, `docs/POINT_GUIDED_SERVER_RUN.md:15-24`
    - **Evidence:** Wide dependency ranges permit unexpected PyTorch/CUDA API changes on server environments.
    - **Remediation:** Generate and maintain a locked server environment lockfile (e.g. `requirements-server-lock.txt`).

---

### 6.3 Minor & Documentation Hygiene (P3 / DOC)

17. **`[SYS-DEF-006]` Float64 Point Coordinates Raise Unchecked Matmul Error in Decoder**
    - **File:** `src/smagm/features/point_guided/reward.py:115-117`, `decoder.py:136-144`
    - **Remediation:** Enforce point and dynamic state dtype equality with clear `TypeError` before decoder MLP forward.

18. **`[SYS-DEF-014]` Evaluation Metadata Hardcodes `"git_head": None`**
    - **File:** `src/smagm/cli/point_guided_eval.py:230`
    - **Remediation:** Connect `git_head` field to `_git_head()` utility.

19. **`[SYS-DEF-001]` `CODEGRAPH.json` Task `baseline_training` References Phantom Files**
    - **File:** `CODEGRAPH.json`
    - **Remediation:** Remove non-existent references (`baseline_data.py`, etc.) to align with `server_pipeline`.

20. **`[SYS-DEF-002]` Orphaned Metric Duplicate `point_guided_metrics.py`**
    - **File:** `src/smagm/features/point_guided/point_guided_metrics.py`
    - **Remediation:** Deprecate and remove duplicate metric file in favor of `baseline_metrics.py`.

21. **`[SYS-DOC-001]` Incomplete Public Exports in `features/point_guided/__init__.py`**
    - **File:** `src/smagm/features/point_guided/__init__.py:14-23`
    - **Remediation:** Export completed Gate A–G symbols in `__all__`.

22. **`[SYS-DOC-002]` Documentation Status Drift in `README.md` and Quality Catalogs**
    - **File:** `README.md:20-23`, `docs/architecture/POINT_GUIDED_FRONTEND.md:13-14`, `quality/README.md:15-17`
    - **Remediation:** Update documentation to reflect implemented software status for Gates F and G while clearly noting pending server experimental execution.

---

## 7. Actionable Server Pre-Flight Remediation Roadmap

To ensure successful F3/F4 training and Gate-G evaluation on server hardware, remediations should be sequenced as follows:

```mermaid
flowchart TD
    subgraph Phase 1: Critical Contract & Configuration Alignments
        A1["Reconcile offset_hidden_channels (12 vs 128) across server configs"]
        A2["Fix DDP teardown barrier deadlock in destroy_distributed()"]
        A3["Implement unique temp naming in _atomic_json() & run dir locking"]
        A4["Standardize shell scripts to use unified PYTHON_BIN / TORCHRUN_BIN"]
    end

    subgraph Phase 2: Provenance & Checkpoint Transactionality
        B1["Recompute & verify canonical split_hash in _load_split()"]
        B2["Add split_hash to baseline_checkpoint_metadata()"]
        B3["Pre-validate state_dict before in-place mutation in load_checkpoint"]
        B4["Enforce model instance ownership check in compute_training_objective()"]
    end

    subgraph Phase 3: Runtime State & Resource Robustness
        C1["Preserve & restore per-rank RNG state across DDP resume"]
        C2["Persist patience_count in resume checkpoints"]
        C3["Add try/finally cleanup for wandb.finish() and optimizer gradients"]
        C4["Fix structural inventory regex filtering to log OTHER_INVALID"]
        C5["Connect git_head dynamically in evaluation_metadata.json"]
    end

    subgraph Phase 4: Code Hygiene & Documentation Harmonization
        D1["Remove orphaned point_guided_metrics.py & clean CODEGRAPH.json"]
        D2["Update features/point_guided/__init__.py public exports"]
        D3["Harmonize README.md & quality/ checklists with AGENTS.md"]
    end

    Phase 1 --> Phase 2 --> Phase 3 --> Phase 4
```

---

## 8. Verification & Audit Certification

- **Frozen Target Commit:** `0efeb94af72ffa067769e19afcd19ad358feefd2` (`main`)
- **Read-Only Verification:** `git status --short` confirms zero modifications to production code, unit tests, configurations, or plans.
- **Syntax & Whitespace Verification:** `compileall` and `git diff --check` passed with 0 errors.
- **Scientific Conclusion:** The point-guided reconstruction architecture is mathematically sound, decoupled from legacy systems, and strictly preserves target isolation. Remediation of the operational, distributed, and configuration defects identified above will make the codebase fully robust for server execution.

*Signed,*  
**Independent System Auditor (GPT-OSS 120B Substitute)**  
*Dispatched under `reports/audit/01-sol-plan.md`*
