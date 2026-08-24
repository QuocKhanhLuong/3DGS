# AGY-A — Architecture, Wiring, and Documentation Authority Audit Report

## 1. Audit Metadata and Target Context

- **Audit Target Commit (HEAD):** `0efeb94af72ffa067769e19afcd19ad358feefd2` (`main`)
- **Base / Upstream:** `origin/main` at `0efeb94af72ffa067769e19afcd19ad358feefd2`
- **Pre-existing Dirty Files Preserved:**
  - Modified: `.DS_Store`, `src/.DS_Store`, `tests/.DS_Store`
  - Untracked: `configs/.DS_Store`, `docs/.DS_Store`, `docs/architecture/point_guided_reward_cost_trajectory.html`, `scripts/.DS_Store`
- **Mutation Scope:** `reports/audit/workers/agy-a-architecture.md` only. Production code, tests, configs, plans, and existing documentation are strictly read-only.
- **Auditor Role:** AGY-A (Architecture, wiring, legacy reachability, export consistency, and documentation authority).

---

## 2. Executive Summary

1. **Active Wiring & Component Ownership:**
   The active point-guided reconstruction pipeline is fully wired end-to-end from operator entrypoints (`src/smagm/cli/point_guided_train.py`, `src/smagm/cli/point_guided_eval.py`, and `scripts/point_guided_*.sh`) through the server trainer/evaluator (`src/smagm/training/point_guided.py`), the full-volume data adapter (`src/smagm/data/brats21_point_guided.py`), and the core model (`src/smagm/features/point_guided/model.py`). The architecture executes one shared MedicalNet ResNet10 traversal, extracts coarse semantics, places deterministic initial points, applies bounded displacement (<2 mm), computes compact semantic-aware PoU, projects static base planes ($B_{xy}, B_{xz}, B_{yz}$), constructs fixed 2-level SWT-Haar spectral anchor planes ($A_{xy}, A_{xz}, A_{yz}$), derives geometry-aware 168-d point spectral evidence ($f_{spec}$), runs a bounded Gate-C reward-cost trajectory producing dynamic state $Z_K$, decodes $Z_K$ via a final-$Z$-only Gate-D implicit decoder, and applies target-after-inference Gate-E supervision (in training) or deterministic Gate-G inference (in evaluation).

2. **Legacy Code Reachability & Isolation:**
   AST-level import and call-graph analysis proves that legacy packages (`src/smagm/anchors/`, `src/smagm/fields/`, `src/smagm/memory/`, `src/smagm/routing/`, `src/smagm/reconstruction/`, `src/smagm/render/`, `src/smagm/gaussians.py`, `src/smagm/state.py`, `src/smagm/data/brats21.py`, `src/smagm/training/trainer.py`, and `src/smagm/cli/train.py`) are **completely unreachable** from any active point-guided module. The point-guided frontend has zero runtime dependencies on legacy 3DGS code. The only cross-package reference is `src/smagm/data/__init__.py`, which re-exports both legacy and point-guided data classes, and the canonical coordinate contract in `src/smagm/contracts/coordinates.py`.

3. **Public Exports and API Inconsistencies:**
   - Package-level `__init__.py` exports (`src/smagm/__init__.py`, `src/smagm/features/__init__.py`, and `src/smagm/training/__init__.py`) currently expose only legacy 3DGS symbols and omit all point-guided symbols.
   - The module export file `src/smagm/features/point_guided/__init__.py` only exposes Phase 1–5 symbols (`PointGuidedConfig`, `PointGuidedMRIModel`, `EmptySparseSupportError`, `FrontendOutput`, `PointField`, `PointGuidedGeometryError`, `SparsePoU`, `VolumeGeometry`). It omits all completed Gate A/B/C/D/E/F/G symbols (`BaseTriPlanes`, `SpectralAnchor`, `PointSpectralEvidence`, `DynamicTriPlanes`, `TrajectoryConfig`, `BaselineTrainingConfig`, `BaselineInferenceResult`, `GateGInferenceConfig`, `ReconstructionMetrics`, etc.).
   - Its module docstring still reads: `"""Modular point-guided MRI frontend with deliberate future-only interfaces."""`

4. **Reward Revert Verification:**
   Commit `39a39d3` (`revert(point-guided): restore 126-d reward descriptor after failed paired gate`) was audited in detail. The revert was clean and complete:
   - `REWARD_DESCRIPTOR_CHANNELS = 126` (96 dynamic state + 3 point semantic + 24 reliability-weighted $\\bar{q}$ + 3 reliability = 126) is cleanly restored.
   - `RewardNet` parameter count is exactly 8,193 (`Linear(126, 64)` + `Linear(64, 1)`).
   - Removed experimental classes (`CandidateCorrections`, `forward_candidates`, `ACTION_DESCRIPTOR_CHANNELS`) have zero dangling references in production code, tests, or configurations.

5. **Dead, Duplicate, and Phantom Paths:**
   - `src/smagm/features/point_guided/point_guided_metrics.py`: Orphaned duplicate metrics module (global 1D/2D SSIM) unreferenced by production runners (`training/point_guided.py` and `cli/point_guided_eval.py` use `baseline_metrics.py`). Not tracked in `CODEGRAPH.json`.
   - `src/smagm/features/point_guided/interfaces.py`: Obsolete pre-Gate-C abstract placeholder classes (`SpectralAnchorBase`, `InitialDynamicTriPlaneBase`, `TrajectorySelectorBase`, `LocalTrajectoryUpdaterBase`, etc.) that are not implemented or used by production modules.
   - `CODEGRAPH.json` phantom paths: Task `baseline_training` lists phantom file paths (`src/smagm/features/point_guided/baseline_data.py`, `scripts/point_guided_baseline.py`, `configs/training/point_guided_baseline.json`, `tests/features/point_guided/test_baseline_data.py`) that do not exist on the filesystem and were superseded by `server_pipeline` assets.

6. **Documentation & Authority Hierarchy Reconciliation:**
   The active repository authority hierarchy is:
   `AGENTS.md` > `PLAN_GATE_F_G.md` / `PLAN_GATE_C_D_E.md` > `PLAN.md` > `CODEGRAPH.json` / `CODEBASE.md` > `README.md` / `docs/architecture/POINT_GUIDED_FRONTEND.md` / `quality/README.md` / `quality/checklists.json`.
   Several non-authoritative documentation files (`README.md`, `POINT_GUIDED_FRONTEND.md`, `quality/README.md`, `quality/checklists.json`) contain outdated status prose claiming Gate F and Gate G are "inactive" or "default-deny", whereas Gate F F1/F2 and server pipeline software (F3/F4 ready) and Gate G G1–G4 software are fully implemented in code.

---

## 3. Detailed Architecture and Wiring Trace

### 3.1 Operator Entrypoints

| Entrypoint | Type | Invocation Command | Target Module |
|---|---|---|---|
| `scripts/point_guided_train_4070.sh` | Shell | `bash scripts/point_guided_train_4070.sh` | `smagm.cli.point_guided_train` with `configs/training/point_guided_brats21_4070.json` |
| `scripts/point_guided_train_2xa4000.sh` | Shell | `bash scripts/point_guided_train_2xa4000.sh` | `torchrun --nproc_per_node=2 -m smagm.cli.point_guided_train` with `configs/training/point_guided_brats21_2xa4000.json` |
| `scripts/point_guided_overfit_4070.sh` | Shell | `bash scripts/point_guided_overfit_4070.sh` | `smagm.cli.point_guided_train` with `configs/training/point_guided_brats21_overfit.json --overfit` |
| `scripts/point_guided_preflight.sh` | Shell | `bash scripts/point_guided_preflight.sh` | `smagm.cli.point_guided_train --preflight` |
| `scripts/point_guided_eval.sh` | Shell | `bash scripts/point_guided_eval.sh <ckpt>` | `smagm.cli.point_guided_eval` with `configs/evaluation/point_guided_brats21_eval.json` |
| `src/smagm/cli/point_guided_train.py` | Python CLI | `python -m smagm.cli.point_guided_train --config <path> --data-root <path>` | `smagm.training.point_guided.run_training()` or `preflight()` |
| `src/smagm/cli/point_guided_eval.py` | Python CLI | `python -m smagm.cli.point_guided_eval <ckpt> --config <path> --data-root <path> --output-dir <path>` | `smagm.cli.point_guided_eval.evaluate()` |

### 3.2 Component Ownership and Call Graph

```text
[Operator Entry: point_guided_train.py / point_guided_eval.py]
  │
  ├─► [src/smagm/training/point_guided.py] (Training & Validation Runtime)
  │     │
  │     ├─► [src/smagm/data/brats21_point_guided.py] (Full-Volume BraTS21 Data Adapter)
  │     │     └─► Validates geometry, [X,Y,Z]->[D,H,W] tensors, input-derived brain mask, robust 0-1 norm
  │     │
  │     ├─► [src/smagm/features/point_guided/baseline_training.py] (Optimizer Builder)
  │     │     └─► Resolves exact optimizer parameters (103,425 trainable params with offset_predictor)
  │     │
  │     ├─► [src/smagm/features/point_guided/baseline_checkpoint.py] (Checkpoint Persistence)
  │     │     └─► Atomic save/load of resume checkpoints & clean inference checkpoints
  │     │
  │     └─► [src/smagm/features/point_guided/semantic_supervision.py] (Auxiliary Grounding)
  │           └─► CE loss on coarse classes from BraTS segmentation (training only)
  │
  └─► [src/smagm/features/point_guided/model.py: PointGuidedMRIModel]
        │
        ├─► [medicalnet_resnet10.py & semantic_prior.py] (Phase 1-3)
        │     └─► Frozen 3-channel MedicalNet ResNet10 backbone -> S_coarse (3 classes: normal, edema, core)
        │
        ├─► [points.py, refinement.py, sampling.py, directional.py, offset_predictor.py] (Phase 1-3)
        │     └─► Deterministic initial points -> directional sampling -> bounded offset (<2 mm) -> refined points
        │
        ├─► [pou.py, semantic_affinity.py, spatial_affinity.py] (Phase 1-3)
        │     └─► Sparse semantic-aware partition of unity on fixed 4 mm spheres
        │
        ├─► [triplane_projection.py] (Phase 4-5)
        │     └─► Axis-conditioned projection to static base planes Bxy [B,C,H,W], Bxz [B,C,D,W], Byz [B,C,D,H]
        │
        ├─► [swt_haar.py & spectral_anchor.py] (Phase 6 / Gate A)
        │     └─► Fixed 2-level SWT-Haar + shared 1x1 Conv (64->8) -> static anchor Axy, Axz, Ayz (56-ch each)
        │
        ├─► [spectral_query.py & cross_plane_consistency.py] (Phase 7 / Gate B)
        │     └─► Geometry-aware bilinear query -> descriptor q=[LL2,E1,E2] (24-d) -> reliability alpha -> f_spec (168-d)
        │
        ├─► [state_init.py, reward.py, trajectory_cost.py, trajectory_solver.py, updater.py, writeback.py, trajectory.py] (Gate C)
        │     └─► Dynamic state Z0 -> RewardNet (126->64->1) -> route costs -> greedy solver -> UpdateNet (270->96) -> 4mm writeback -> Z_K
        │
        ├─► [decoder.py] (Gate D)
        │     └─► Geometry-aware chunked query of Z_K -> shared 96->64->32->1 SiLU MLP -> absolute T1ce prediction
        │
        ├─► [losses.py, reward_supervision.py, training_objective.py] (Gate E - Training only)
        │     └─► Charbonnier + 3D SSIM + gradient loss + counterfactual reward SmoothL1 loss + trajectory regularization
        │
        └─► [baseline_inference.py & baseline_metrics.py] (Gate G - Evaluation only)
              └─► Deterministic exact-no-revisit inference -> post-inference MAE, PSNR, 3D SSIM, and semantic Dice
```

### 3.3 Active Point-Guided File Inventory and Module Status

| File Path | Gate / Phase Ownership | Status | Wiring and Role |
|---|---|---|---|
| `config.py` | Phase 1–7 | Active | `PointGuidedConfig` configuration dataclass |
| `contracts.py` | Phase 1–7 | Active | Typed data structures: `VolumeGeometry`, `PointField`, `SparsePoU`, `PointSpectralEvidence`, `FrontendOutput` |
| `medicalnet_resnet10.py` | Phase 1 | Active | MedicalNet ResNet10 3D backbone with deterministic 1->3 ch stem |
| `semantic_prior.py` | Phase 2–3 | Active | Semantic prior head returning 3 coarse semantic classes |
| `points.py` | Phase 1–3 | Active | Deterministic quasi-uniform point placement |
| `directional.py` | Phase 1–3 | Active | Directional coordinate offsets ($\\pm 1, \\pm 2, \\pm 3$ mm) |
| `sampling.py` | Phase 1–3 | Active | Feature-grid and volume 3D trilinear sampling helpers |
| `offset_predictor.py` | Phase 1–3 / Gate F | Active | Trainable MLP offset predictor (1,419 parameters) |
| `refinement.py` | Phase 1–3 | Active | Refines initial points bounded by $\\le 2$ mm |
| `spatial_affinity.py` | Phase 1–3 | Active | Quadratic compact spatial affinity kernel ($r = 4$ mm) |
| `semantic_affinity.py` | Phase 1–3 | Active | Exact L1 semantic affinity metric |
| `pou.py` | Phase 1–3 | Active | Sparse partition-of-unity edge construction |
| `triplane_projection.py` | Phase 4–5 | Active | Axis-conditioned shallow feature projection to $B_{xy}, B_{xz}, B_{yz}$ |
| `swt_haar.py` | Phase 6 / Gate A | Active | Fixed 2D stationary wavelet transform (Haar filters) |
| `spectral_anchor.py` | Phase 6 / Gate A | Active | Static spectral anchor builder $A_{xy}, A_{xz}, A_{yz}$ (56 channels) |
| `spectral_query.py` | Phase 7 / Gate B | Active | Full-affine feature-grid geometry mapping and bilinear query |
| `cross_plane_consistency.py` | Phase 7 / Gate B | Active | Pairwise cosine agreement, softmax reliability $\\alpha$, 168-d $f_{spec}$ |
| `state_init.py` | Gate C (C1) | Active | Shared $B \\to Z_0$ dynamic triplane initialization |
| `reward.py` | Gate C (C2) | Active | Dynamic state query, 126-d descriptor, shared `RewardNet` |
| `trajectory_cost.py` | Gate C (C3) | Active | Travel, overlap, and step costs |
| `trajectory_solver.py` | Gate C (C4) | Active | Adaptive greedy point selection solver |
| `updater.py` | Gate C (C5) | Active | Shared `UpdateNet` ($270 \\to 96$) |
| `writeback.py` | Gate C (C6) | Active | Compact 4-mm physical write-back onto dynamic tri-planes |
| `trajectory.py` | Gate C (C7) | Active | Composite trajectory execution and diagnostic output |
| `availability.py` | Gate C / G | Active | Route availability policy & exact-no-revisit tracking |
| `decoder.py` | Gate D (D1) | Active | Chunked geometry-aware implicit decoder ($96 \\to 64 \\to 32 \\to 1$) |
| `losses.py` | Gate E (E1–E4) | Active | Charbonnier, 3D SSIM, and DHW gradient reconstruction losses |
| `reward_supervision.py` | Gate E (E5–E8) | Active | Counterfactual reward measurement & SmoothL1 supervision |
| `training_objective.py` | Gate E (E9) | Active | Composite training loss calculation |
| `semantic_supervision.py` | Gate F | Active | Auxiliary coarse semantic CE loss on BraTS labels |
| `baseline_training.py` | Gate F | Active | Trainable parameter extraction & AdamW optimizer configuration |
| `baseline_checkpoint.py` | Gate F / G | Active | Checkpoint persistence (resume & clean inference) |
| `baseline_inference.py` | Gate G | Active | Deterministic target-free Gate-G inference execution |
| `baseline_metrics.py` | Gate G | Active | Post-inference reconstruction metrics (MAE, PSNR, 3D SSIM) & semantic Dice |
| `model.py` | Phases 1–7, Gates C–G | Active | Central `PointGuidedMRIModel` composition |
| `interfaces.py` | Pre-Gate C | **Obsolete** | Unused abstract placeholder classes (`SpectralAnchorBase`, etc.) |
| `point_guided_metrics.py` | Gate G drafting | **Dead / Duplicate** | Orphaned metrics helper superseded by `baseline_metrics.py` |

---

## 4. Legacy Reachability Analysis

### 4.1 Static Dependency Boundary
An exhaustive AST inspection of all 38 files in `src/smagm/features/point_guided/`, as well as `src/smagm/data/brats21_point_guided.py`, `src/smagm/training/point_guided.py`, and `src/smagm/cli/point_guided_*.py`, was conducted.

**Findings:**
1. **Zero Point-Guided to Legacy Imports:**
   No point-guided module imports from `src/smagm/anchors/`, `src/smagm/fields/`, `src/smagm/memory/`, `src/smagm/routing/`, `src/smagm/reconstruction/`, `src/smagm/render/`, `src/smagm/gaussians.py`, `src/smagm/state.py`, `src/smagm/data/brats21.py`, or `src/smagm/training/trainer.py`.
2. **Shared Canonical Geometry:**
   The only shared contract is `src/smagm/contracts/coordinates.py` (which defines RAS-mm coordinate systems and affine utilities).
3. **Data Package Re-export:**
   `src/smagm/data/__init__.py` re-exports symbols from both `brats21.py` (legacy sparse-plane protocol) and `brats21_point_guided.py` (point-guided full-volume adapter). The two implementations do not import each other.
4. **Conclusion:**
   Legacy code is completely decoupled and inactive during point-guided execution.

---

## 5. Public Export and Interface Audit

### 5.1 Package `__init__.py` Summary

- **`src/smagm/__init__.py`:**
  Exports 78 legacy symbols (`FixedGaussianHead`, `sample_fixed_supports`, `EpisodeController`, `T1CTrainer`, `bootstrap_anchors`, `GlobalStructuralField`, `GaussianMemory`, etc.). Contains **zero** point-guided exports.
- **`src/smagm/features/__init__.py`:**
  Exports 12 legacy T1-A symbols (`EvidenceEncoder`, `analytic_feature_bank`, `CachedFeatureMaps`, etc.). Does not re-export `point_guided`.
- **`src/smagm/training/__init__.py`:**
  Exports 30 legacy T1-C training symbols (`T1CTrainer`, `build_legal_episode_step`, etc.). Does not export `run_training` or `preflight` from `point_guided.py`.
- **`src/smagm/features/point_guided/__init__.py`:**
  Exports only 8 symbols from Phases 1–5:
  `EmptySparseSupportError`, `FrontendOutput`, `PointField`, `PointGuidedConfig`, `PointGuidedGeometryError`, `PointGuidedMRIModel`, `SparsePoU`, `VolumeGeometry`.
  All completed Gate A–G symbols (`BaseTriPlanes`, `SpectralAnchor`, `PointSpectralEvidence`, `DynamicTriPlanes`, `TrajectoryConfig`, `BaselineTrainingConfig`, `BaselineInferenceResult`, `GateGInferenceConfig`, `ReconstructionMetrics`, `compute_reconstruction_metrics`, etc.) are missing from `__all__`.

---

## 6. Reward Revert Verification (Commit `39a39d3`)

### 6.1 Context & Commit Trace
Commit `39a39d3aabf568cfa3a5ccb6c6d7ad373f54a06a` (`revert(point-guided): restore 126-d reward descriptor after failed paired gate`) reverted an experimental 222-d descriptor that improperly passed candidate updates into RewardNet.

### 6.2 Audit Verification Points
1. **Descriptor Dimension:**
   `REWARD_DESCRIPTOR_CHANNELS` in `src/smagm/features/point_guided/reward.py` (line 17) is defined as:
   `STATE_QUERY_CHANNELS (96) + 3 + CONSISTENCY_DESCRIPTOR_CHANNELS (24) + 3 = 126`.
2. **RewardNet Architecture:**
   `RewardNet` is an MLP: `nn.Linear(126, 64) -> nn.SiLU() -> nn.Linear(64, 1) -> nn.Sigmoid()`.
   Parameter count: $(126 \\times 64 + 64) + (64 \\times 1 + 1) = 8,064 + 64 + 64 + 1 = 8,193$.
   Verified in `tests/features/point_guided/test_reward.py` and `tests/features/point_guided/test_baseline_training.py`.
3. **No Dangling Artifacts:**
   Searches for `222`, `ACTION_DESCRIPTOR_CHANNELS`, `CandidateCorrections`, and `forward_candidates` across the entire codebase return zero hits in production code, tests, configs, and active plans.
4. **Conclusion:**
   The revert is 100% clean and consistent with `PLAN_GATE_C_D_E.md`.

---

## 7. Documentation and Authority Hierarchy Reconciliation

### 7.1 Authority Hierarchy

```text
1. AGENTS.md (Top Authority: active status, locked rules, scope restrictions)
   └─► 2. PLAN_GATE_F_G.md & PLAN_GATE_C_D_E.md (Gate C-G engineering plans)
         └─► 3. PLAN.md (Root frontend plan for Phases 1-7)
               └─► 4. CODEGRAPH.json & CODEBASE.md (Navigation access-control & ownership)
                     └─► 5. README.md & POINT_GUIDED_FRONTEND.md (Informational documentation)
                           └─► 6. quality/README.md & quality/checklists.json (Quality gate specs)
```

### 7.2 Documentation Drift Inventory

| Document | Stale Statement / Location | Actual Code Reality |
|---|---|---|
| `README.md` | Lines 20–23: *"Gate F is next/inactive; Gate G remains inactive."* | Gate F F1/F2 and server pipeline software (F3/F4 ready) and Gate G G1–G4 software are fully implemented in code. Only server GPU execution and trained-checkpoint evidence are pending. |
| `docs/architecture/POINT_GUIDED_FRONTEND.md` | Lines 13–14: *"Next / inactive: Gate F. Blocked / default-deny: Gate G and the final inference policy."* | `baseline_training.py`, `baseline_checkpoint.py`, `baseline_inference.py`, `baseline_metrics.py`, `point_guided_train.py`, and `point_guided_eval.py` are implemented. |
| `quality/README.md` | Lines 15–17: *"Gate F is next/inactive and Gate G remains default-deny under AGENTS.md and CODEGRAPH.json."* | Gate F baseline software and Gate G inference are implemented in the server pipeline. |
| `quality/checklists.json` | Line 37: *"Spectral anchors, dynamic tri-planes, selection, updates, history, stopping, decoding, and reconstruction losses remain unresolved interfaces."* | All named components are implemented (Phases 6–7, Gates C, D, E). |
| `src/smagm/features/point_guided/__init__.py` | Line 1: *"Modular point-guided MRI frontend with deliberate future-only interfaces."* | Interfaces are implemented, not future-only. |

---

## 8. Findings Catalog

### [AGY-A-FIND-001] (Severity: P2) Phantom and stale file paths declared in `CODEGRAPH.json` task `baseline_training`

- **Component:** `CODEGRAPH.json`
- **Description:**
  Task `baseline_training` in `CODEGRAPH.json` declares four paths that do not exist on the filesystem:
  1. `src/smagm/features/point_guided/baseline_data.py` (listed under `entrypoints`, `read_paths`, `write_paths`)
  2. `scripts/point_guided_baseline.py` (listed under `entrypoints`, `read_paths`, `write_paths`)
  3. `configs/training/point_guided_baseline.json` (listed under `write_paths`)
  4. `tests/features/point_guided/test_baseline_data.py` (listed under `read_paths`, `write_paths`)
- **Code Evidence:**
  Filesystem scan confirms these files do not exist. They were early drafting artifacts superseded by `src/smagm/data/brats21_point_guided.py`, `src/smagm/cli/point_guided_train.py`, `configs/training/point_guided_brats21_*.json`, and `tests/data/test_brats21_point_guided.py` under the `server_pipeline` task.
- **Impact:**
  Invoking `scripts/codegraph.py --task baseline_training --check <path>` or navigating via `baseline_training` references phantom files.
- **Recommendation:**
  Update `CODEGRAPH.json` task `baseline_training` to remove the non-existent file references and align with the `server_pipeline` task.

---

### [AGY-A-FIND-002] (Severity: P3) Orphaned duplicate metric helper `point_guided_metrics.py`

- **Component:** `src/smagm/features/point_guided/point_guided_metrics.py`
- **Description:**
  `point_guided_metrics.py` implements `PointGuidedMetrics` and `compute_point_guided_metrics` (global 1D/2D SSIM). It is completely unreferenced by production code (`src/smagm/training/point_guided.py` and `src/smagm/cli/point_guided_eval.py` import `compute_reconstruction_metrics` from `baseline_metrics.py`). It is only imported by `tests/features/point_guided/test_point_guided_metrics.py` and is omitted from `CODEGRAPH.json`.
- **Code Evidence:**
  - `src/smagm/features/point_guided/point_guided_metrics.py` (136 lines)
  - `src/smagm/cli/point_guided_eval.py:18` imports `from ..features.point_guided.baseline_metrics import compute_reconstruction_metrics, semantic_dice`
  - `src/smagm/training/point_guided.py:40` imports `from ..features.point_guided.baseline_metrics import compute_reconstruction_metrics, semantic_dice`
- **Impact:**
  Redundant code creates maintenance confusion regarding which metric implementation is authoritative.
- **Recommendation:**
  Deprecate or remove `point_guided_metrics.py` and its test, standardizing entirely on `baseline_metrics.py`.

---

### [AGY-A-FIND-003] (Severity: P3) Incomplete public export surface in `src/smagm/features/point_guided/__init__.py`

- **Component:** `src/smagm/features/point_guided/__init__.py`
- **Description:**
  `__init__.py` only exposes Phase 1–5 symbols (`PointGuidedConfig`, `PointGuidedMRIModel`, `EmptySparseSupportError`, `FrontendOutput`, `PointField`, `PointGuidedGeometryError`, `SparsePoU`, `VolumeGeometry`). All completed Gate A–G symbols (`BaseTriPlanes`, `SpectralAnchor`, `PointSpectralEvidence`, `DynamicTriPlanes`, `TrajectoryConfig`, `BaselineTrainingConfig`, `BaselineInferenceResult`, `GateGInferenceConfig`, `ReconstructionMetrics`, etc.) are missing from `__all__`.
- **Code Evidence:**
  `src/smagm/features/point_guided/__init__.py` lines 14–23.
- **Impact:**
  External callers cannot import Gate A–G components directly from `smagm.features.point_guided` and must import from internal submodules.
- **Recommendation:**
  Update `src/smagm/features/point_guided/__init__.py` to export the public contracts and configurations of Gates A–G.

---

### [AGY-A-FIND-004] (Severity: DOC / P3) Documentation drift regarding Gate F and Gate G implementation status

- **Component:** `README.md`, `docs/architecture/POINT_GUIDED_FRONTEND.md`, `quality/README.md`, `quality/checklists.json`
- **Description:**
  Public documentation states that Gate F is "next/inactive" and Gate G is "inactive" or "default-deny", whereas Gate F baseline training software and Gate G inference software are already fully implemented in `src/smagm/features/point_guided/` and `src/smagm/training/point_guided.py`.
- **Code Evidence:**
  - `README.md` lines 20–23
  - `docs/architecture/POINT_GUIDED_FRONTEND.md` lines 13–14
  - `quality/README.md` lines 15–17
  - `quality/checklists.json` line 37
- **Impact:**
  Readers and operators are misinformed about the actual software capabilities and implemented gate status of the repository.
- **Recommendation:**
  Harmonize documentation prose across `README.md`, `POINT_GUIDED_FRONTEND.md`, and `quality/` to match `AGENTS.md` and `PLAN_GATE_F_G.md` (distinguishing implemented software from pending server experimental evidence).

---

### [AGY-A-FIND-005] (Severity: INFO / P3) Obsolete abstract placeholder module `interfaces.py`

- **Component:** `src/smagm/features/point_guided/interfaces.py`
- **Description:**
  `interfaces.py` defines placeholder classes (`SpectralAnchorBase`, `InitialDynamicTriPlaneBase`, `TrajectoryHistory`, `TrajectorySelectorBase`, `LocalTrajectoryUpdaterBase`, `StoppingPolicyBase`, `FinalTriPlaneDecoderBase`, `ReconstructionLossConfig`) created during Phase 1 for unstarted gates. Production classes in Gates C–E were implemented without subclassing these abstract classes.
- **Code Evidence:**
  `src/smagm/features/point_guided/interfaces.py` (104 lines). Only `tests/features/point_guided/test_frontend_boundaries.py` imports this file.
- **Impact:**
  Dead placeholder types remain in the codebase without participating in the runtime type system.
- **Recommendation:**
  Retire or clean up `interfaces.py` once boundary tests are updated.

---

### [AGY-A-FIND-006] (Severity: INFO) Package roots re-export only legacy 3DGS symbols

- **Component:** `src/smagm/__init__.py`, `src/smagm/features/__init__.py`, `src/smagm/training/__init__.py`
- **Description:**
  Top-level package `__init__.py` files export only legacy 3DGS symbols and zero point-guided symbols.
- **Code Evidence:**
  - `src/smagm/__init__.py` (158 lines, 78 legacy exports)
  - `src/smagm/features/__init__.py` (25 lines, 12 legacy exports)
  - `src/smagm/training/__init__.py` (63 lines, 30 legacy exports)
- **Impact:**
  Point-guided components must be imported via deep submodule paths (`smagm.features.point_guided`, `smagm.data.brats21_point_guided`, `smagm.training.point_guided`).
- **Recommendation:**
  Maintain this separation until legacy code retirement is officially scheduled under repository governance.

---

## 9. Verification Summary

- **HEAD Verification:** `git rev-parse HEAD` confirmed `0efeb94af72ffa067769e19afcd19ad358feefd2` before and after audit.
- **Dirty State Verification:** `git status --short` confirmed only the pre-existing dirty files (`.DS_Store`, etc.) plus this assigned report.
- **Compilation Check:** `python -m compileall -q src tests configs scripts` passed with 0 errors.
- **Whitespace Check:** `git diff --check` passed with 0 errors.
- **Codegraph Scope:** Verified with `python scripts/codegraph.py --task frontend`, `--task trajectory`, `--task server_pipeline`.
