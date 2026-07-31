# CODEBASE — Final Research Software Blueprint

## 1. Purpose

This document is the software blueprint for the complete sparse multi-sequence
MRI Gaussian reconstruction research project.

The theoretical architecture is defined under `docs/reconstruction/`.
`CODEBASE.md` maps that theory into stable packages, files, interfaces,
dependency rules, execution flows, and tests.

Implementation phases such as T0, T0.5, T1, T2, T3, T4, and T5 exist only to:

- divide implementation into bounded context windows;
- define authorization and Human Gates;
- make verification and rollback manageable;
- track which subset of the final architecture currently exists.

Phases do **not** define the final software architecture. Production research
modules should be named by responsibility, not by phase. A phase may add or
complete files described here, but should not create a parallel architecture
such as `t2_model.py`, `t3_memory.py`, or `t4_router.py` when the stable
responsibility belongs in `anchors/`, `fields/`, `memory/`, or `routing/`.

## 2. Authority and reading order

Use the following order when implementing or reviewing code:

1. `docs/strategies/2026-07-29-isbi-realignment.md` and its current execution-
   status addendum for the thesis, claims, authorization, Human Gates, and stop
   decisions.
2. `docs/reconstruction/` for the full theoretical method and cross-phase
   invariants.
3. `CODEBASE.md` for the final software architecture and file ownership.
4. `docs/codex/README.md` and the current phase handoff for the subset that is
   currently executable or authorized.
5. The nearest tests and implementation files for exact runtime behavior.

When documents conflict, strategy controls authorization, reconstruction docs
control the intended method, and this file controls where that method belongs
in code.

## 3. Theory-to-code map

| Theoretical specification | Stable software ownership |
|---|---|
| `modules/EVIDENCE_ENCODER.md` | `src/smagm/features/`, structural losses in `src/smagm/losses/` |
| `modules/ANCHOR_LOCAL_FIELD.md` | `src/smagm/anchors/` and `src/smagm/fields/` |
| `modules/SDF_GAUSSIAN_MEMORY.md` | low-level `src/smagm/gaussians.py` plus `src/smagm/memory/` |
| `modules/TRAJECTORY_ROUTER.md` | `src/smagm/routing/` |
| `modules/PLANE_RENDERER_RECONSTRUCTOR.md` | low-level `src/smagm/renderer.py` plus `src/smagm/reconstruction/` |
| `phases/01_DIRECT_SPARSE_TRAINING.md` | `src/smagm/training/`, `configs/`, `experiments/` |
| `phases/02_INITIAL_ANCHOR_BOOTSTRAP.md` | `anchors/`, `fields/`, memory initialization, patient-state construction |
| `phases/03_ACTIVE_TRAJECTORY_UPDATE.md` | routing, assimilation, topology update, stopping |
| `phases/04_FINAL_RECONSTRUCTION.md` | reconstruction, export, isolated evaluation |

## 4. End-to-end final flow

```text
permanently sparse manifest
→ patient-level split validation
→ immutable episode or acquisition assignment
→ legal context/committed observation loading
→ normalization and physical-plane binding
→ analytic differential scaffold
→ shared teacher-free evidence encoder
→ versioned compact evidence cache
→ physical candidate generation and anchor bootstrap
→ geometry-aware evidence aggregation
→ shared tiny anchor-local structural field
→ blended patient structural field
→ structural and volumetric Gaussian memory
→ physical-plane rendering
→ sparse target loss or render-before-update residual
→ local state assimilation and topology operations
→ legal trajectory selection and stopping
→ chunked full-volume reconstruction
→ serialized reconstruction package
→ isolated audit evaluation
```

The complete path must preserve two separations:

1. **global parameters versus patient state**;
2. **reconstruction generation versus audit-ground-truth evaluation**.

## 5. Status legend

- **IMPLEMENTED** — executable code and focused tests exist on `main`.
- **PARTIAL** — an interface or subset exists, but the final responsibility is
  not complete.
- **PLANNED** — required by the final method but not yet implemented.
- **COMPATIBILITY** — retained only to preserve an existing import path.
- **DIAGNOSTIC** — development or synthetic entry point, not a final user-facing
  command and not a scientific result.

The status annotations below describe the repository at the time this document
was introduced. `docs/codex/README.md` remains the live executable-status index.

## 6. Expected final repository structure

```text
.
├── AGENTS.md
├── CODEBASE.md
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── routing/
│   ├── reconstruction/
│   └── experiments/
├── docs/
│   ├── reconstruction/        # theoretical method backbone
│   ├── strategies/            # thesis, claims, gates, stop decisions
│   ├── designs/               # bounded technical decisions
│   ├── plans/                 # implementation sequencing only
│   ├── codex/                 # executable handoffs and current status
│   ├── protocols/             # data legality and cohort protocols
│   ├── experiments/           # experiment definitions and results
│   ├── reproducibility/       # environments, hashes, audit records
│   └── reviews/               # independent scientific and code reviews
├── experiments/
│   ├── manifests/
│   ├── sweeps/
│   └── reports/
├── quality/                    # machine-readable phase-gate evidence catalog
├── tests/quality/              # quality catalog and runner contract tests
├── docs/checklists/            # human-readable phase-gate documentation
├── scripts/
│   ├── check_phase.py
│   ├── train.py
│   ├── reconstruct.py
│   ├── evaluate.py
│   └── audit.py
├── src/smagm/
│   ├── __init__.py
│   ├── contracts/
│   ├── data/
│   ├── features/
│   ├── losses/
│   ├── anchors/
│   ├── fields/
│   ├── gaussians.py
│   ├── memory/
│   ├── renderer.py
│   ├── state/
│   ├── training/
│   ├── routing/
│   ├── reconstruction/
│   ├── evaluation/
│   ├── baselines/
│   └── cli/
└── tests/
    ├── contracts/
    ├── data/
    ├── features/
    ├── losses/
    ├── anchors/
    ├── fields/
    ├── memory/
    ├── render/
    ├── state/
    ├── training/
    ├── routing/
    ├── reconstruction/
    ├── evaluation/
    ├── baselines/
    └── integration/
```

Future directories in this tree are architectural destinations, not permission
to add empty packages or placeholder APIs before their phase is authorized.

## 7. Package responsibilities

### 7.1 `src/smagm/contracts/`

Owns immutable, low-level scientific records used across the repository. This
package must not import model, training, routing, or evaluation code.

| File | Status | Responsibility |
|---|---|---|
| `coordinates.py` | IMPLEMENTED | Canonical RAS-mm coordinates, source-affine provenance, physical planes, grids, hashes, and geometric validation. |
| `observation.py` | IMPLEMENTED | Role-free sparse availability metadata, manifests, split registry, and legal observation provenance. |
| `observations.py` | COMPATIBILITY | Stable compatibility exports for older observation import paths. No new semantics belong here. |
| `episode.py` | IMPLEMENTED | Immutable episode assignments, context opening, target commit, frozen state, render evidence, receipt registration, reveal ordering, and deployment-cost contracts. |
| `gaussians.py` | COMPATIBILITY | Compatibility exports for low-level Gaussian tensor contracts. |
| `state.py` | PLANNED | Cross-package immutable identifiers and summaries for patient-state versioning, execution status, and serialization schema. It must not own update algorithms. |
| `outputs.py` | PLANNED | Typed slice, volume, uncertainty, trajectory, and final-package output records. |

Key rule: contracts define data meaning and validation, not orchestration.

### 7.2 `src/smagm/data/`

Owns legal access, decoding, preprocessing records, and episode construction.
It must never contain model architecture.

| File | Status | Responsibility |
|---|---|---|
| `manifest.py` | IMPLEMENTED | Focused exports and helpers for sparse manifests. |
| `io.py` | PLANNED | Decode legally opened observation payloads into tensors while preserving source metadata. No direct arbitrary-path reads. |
| `normalization.py` | PLANNED | Context-only normalization, inverse-normalization records, modality policies, and hashable preprocessing configuration. |
| `registration.py` | PLANNED | Registration metadata and confidence validation; no registration model hidden inside loaders. |
| `episodes.py` | PLANNED | Deterministic sampling of legal `EpisodeAssignment` objects from a fixed manifest. It returns IDs and metadata, not target pixels. |
| `cohorts.py` | PLANNED | Training, lesion-validation, and sealed final-audit cohort declarations and isolation checks. |

Data code may depend on `contracts/`; it must not depend on `features/`,
`training/`, or `evaluation/`.

### 7.3 `src/smagm/features/`

Implements the teacher-free 2D evidence path. It processes only legal observed
slices and returns spatially aligned, cacheable maps.

| File | Status | Responsibility |
|---|---|---|
| `analytic.py` | IMPLEMENTED | Fixed differentiable intensity, derivatives, gradient, Laplacian, local-contrast, and validity channels with physical spacing. |
| `contracts.py` | IMPLEMENTED | `EncoderFeatureMaps`, feature-grid-to-plane transforms, valid topology, modality identity, and sampling invariants. |
| `encoder.py` | IMPLEMENTED/PARTIAL | Common E0/E1/E2 output contract and teacher-free micro-CNN. Final version must include the selected explicit modality-conditioning mechanism and final compute accounting. |
| `conditioning.py` | PARTIAL | Declared geometry-preserving intensity perturbations exist. Final ownership also includes tiny modality conditioning such as FiLM, conditional normalization, or a bounded embedding. |
| `cache.py` | IMPLEMENTED/PARTIAL | Exact-key feature caching with source, transform, preprocessing, state, dtype, and topology binding. Final inference storage and serialization policy remain to be completed. |
| `auxiliary.py` | PLANNED | Training-only local differential probe heads. These must not enter inference state. |
| `profiling.py` | PLANNED | Analytic preprocessing, encoder FLOPs, latency, cache bytes, and parameter reporting. |

The encoder must not create anchors, Gaussian topology, patient routes, or full
volumes.

### 7.4 `src/smagm/losses/`

Contains separately testable objective components. It must not open data or own
an optimizer.

| File | Status | Responsibility |
|---|---|---|
| `structural.py` | PARTIAL | Structural consistency, appearance sensitivity, reliability regularization, variance-floor, and registered cross-modality comparison. Final scope includes explicit spatial equivariance, covariance penalty, and local differential preservation. |
| `reconstruction.py` | PLANNED | Supported-mask-aware intensity losses and gradient/frequency-sensitive target-plane losses. |
| `field.py` | PLANNED | Field overlap, gradient regularity, Eikonal/sign/level-set terms when scientifically authorized. |
| `gaussian.py` | PLANNED | Scale, displacement, coverage, overlap, complexity, and topology acceptance regularizers. |
| `calibration.py` | PLANNED | Explicit uncertainty calibration objectives and diagnostics. |
| `compose.py` | PLANNED | Typed, switchable objective composition and component logging. |

A loss being implemented does not imply that its corresponding scientific gate
has passed.

### 7.5 `src/smagm/anchors/`

Owns patient-specific physical support anchors. Anchors are neither slices nor
Gaussians.

| File | Status | Responsibility |
|---|---|---|
| `contracts.py` | PLANNED | Anchor centers, partial/full frames, support scales, evidence, confidence, observability, contributing-plane references, and topology status. |
| `candidates.py` | PLANNED | Sparse structural candidate scoring and physical non-maximum suppression on observed feature maps. |
| `bootstrap.py` | PLANNED | Lift candidates into RAS-mm provisional anchors and perform the initial bootstrap in both differentiable training and frozen-weight inference modes. |
| `consolidation.py` | PLANNED | Cross-plane merge, duplicate suppression, conflict preservation, and disconnected-region coverage. |
| `aggregation.py` | PLANNED | Geometry-, modality-, distance-, registration-, and reliability-aware aggregation into compact anchor evidence. |
| `frames.py` | PLANNED | Partial initial frames, field-derived normals, tangent construction, and uncertainty-aware frame validation. |
| `index.py` | PLANNED | Spatial indexing and bounded anchor-neighborhood queries. |
| `adaptation.py` | PLANNED | Patient-state move, birth, split, merge, and prune proposals. Acceptance belongs to the update controller, not this file alone. |

Anchor code samples the feature cache. It must not rerun the encoder during
patient inference.

### 7.6 `src/smagm/fields/`

Owns the shared tiny anchor-local structural field and its blending. It does not
own evidence alignment, routing, or appearance reconstruction.

| File | Status | Responsibility |
|---|---|---|
| `contracts.py` | PLANNED | Local coordinates, field-query batches, field outputs, support weights, and field-status terminology. |
| `local.py` | PLANNED | One shared low-capacity MLP mapping local coordinate plus compact anchor evidence to a scalar structural-field value. |
| `blend.py` | PLANNED | Stable differentiable partition-of-unity or compact-support blending into a patient field. |
| `query.py` | PLANNED | Batch nearby-anchor lookup, local-coordinate construction, local evaluation, and blending. |
| `regularization.py` | PLANNED | Overlap consistency and optional gradient/Eikonal diagnostics. |

Until signed-distance behavior is demonstrated, public APIs and outputs use
`structural_field` or `level_set_field`, not unconditional `sdf` naming.

### 7.7 `src/smagm/gaussians.py`

**Status: IMPLEMENTED/PARTIAL.**

This is the low-level validated tensor representation for Gaussian primitives:
centers, covariance factors, amplitudes, appearance, gauge provenance, and
validation. It must remain independent of anchors, routing, training, and file
I/O.

It is not the complete dual-bank patient memory. Higher-level ownership belongs
in `memory/`.

### 7.8 `src/smagm/memory/`

Owns the patient-specific structural and volumetric Gaussian banks, their
observability, initialization, assimilation, topology, and spatial lookup.

| File | Status | Responsibility |
|---|---|---|
| `contracts.py` | PLANNED | Structural bank, volumetric bank, primitive type, observability, uncertainty, provenance, and memory summaries. |
| `initialize.py` | PLANNED | Initialize thin field-aligned structural Gaussians and interior volumetric appearance Gaussians from anchors and evidence. |
| `appearance.py` | PLANNED | Per-modality direct appearance slots or authorized compact codes and missing-modality uncertainty. |
| `observability.py` | PLANNED | Evidence counts, coverage, disagreement, residual history, propagation depth, and update-round tracking. |
| `assimilate.py` | PLANNED | Local appearance-first and conservative-geometry updates from a legally observed residual. |
| `topology.py` | PLANNED | Birth, split, merge, and prune proposals plus shared reconstruction/complexity acceptance energy. |
| `propagation.py` | PLANNED | Bounded anchor–Gaussian propagation with uncertainty growth and parent provenance. |
| `index.py` | PLANNED | Bounded-support spatial culling for plane and volume queries. |

No topology operation may inspect an unqueried image or hidden target before the
legal reveal point.

### 7.9 `src/smagm/renderer.py`

**Status: IMPLEMENTED.**

Owns the pure through-plane profile-aware Gaussian reference operator:

- physical-plane geometry;
- thin-plane or finite-slab evaluation;
- normalized additive intensity composition;
- support and unsupported diagnostics;
- deterministic differentiable rendering;
- no ledger mutation and no target reveal.

This file remains a low-level numeric reference. It must not become the owner of
patient-state orchestration, routing, output export, or evaluation.

### 7.10 `src/smagm/state/`

Owns composition and versioning of patient-specific state. Global model
parameters must never be registered as patient state, and patient state must
never become persistent global parameters.

| File | Status | Responsibility |
|---|---|---|
| `patient.py` | PLANNED | `PatientState`: observation ledger reference, evidence cache, anchors, local fields, Gaussian memory, observability, uncertainty, residual history, and trajectory history. |
| `builder.py` | PLANNED | Construct a state from legal context or bootstrap observations using stable module interfaces. |
| `versioning.py` | PLANNED | Canonical hashes and immutable snapshots binding all scientific inputs without target pixels. |
| `update.py` | PLANNED | Apply an accepted local update transaction and emit state-change diagnostics. |
| `serialization.py` | PLANNED | Safe state checkpoint schema, version migration, and artifact hashes. |

`state/` orchestrates patient state but does not implement encoder, field,
Gaussian, or router internals.

### 7.11 `src/smagm/training/`

Owns offline optimization. It is the only package allowed to combine data,
features, anchors, fields, memory, renderer, and losses into optimizer steps.
Lower-level packages must not import `training/`.

| File | Status | Responsibility |
|---|---|---|
| `episode.py` | PLANNED | Legal context-only state construction, target metadata exposure, commit, render, receipt registration, reveal, and live-loss handoff. |
| `objective.py` | PLANNED | Build the switchable predictive, structural, field, Gaussian, and calibration objective from typed results. |
| `trainer.py` | PLANNED | Optimizer, AMP policy, gradient handling, checkpointing, validation, and bounded logging. |
| `schedule.py` | PLANNED | Structural warm-up, joint reconstruction, and reconstruction-dominant refinement without encoding phase names into model APIs. |
| `sampling.py` | PLANNED | Deterministic patient/episode/target sampling and matched schedules across variants. |
| `metrics.py` | PLANNED | Training diagnostics: unsupported fraction, collapse, feature alignment, gradient health, memory size, and compute. |
| `provenance.py` | PLANNED | Resolved config, commit, manifest, assignment, environment, hardware, and artifact bindings. |

A final legal training step is:

```text
sample assignment
→ open context only
→ encode/build patient state
→ freeze state version
→ expose target geometry
→ commit target
→ render live prediction
→ register receipt from detached audit copy
→ reveal target
→ compute supported-mask-aware objective on the live prediction
→ backward and optimizer step
```

### 7.12 `src/smagm/routing/`

Owns legal active observation selection from current patient state. It never
loads candidate pixels.

| File | Status | Responsibility |
|---|---|---|
| `contracts.py` | PLANNED | Candidate metadata, utility components, decisions, proposal provenance, remaining budget, and no-candidate reasons. |
| `candidates.py` | PLANNED | Build legal unqueried actions from metadata and ledger state. |
| `descriptors.py` | PLANNED | State-derived plane uncertainty, coverage, missing modality, residual neighborhood, redundancy, and cost descriptors. |
| `utility.py` | PLANNED | R0/R1 analytic utilities and optional authorized learned-gain interface. |
| `graph.py` | PLANNED | Candidate graph and local repair; no unqueried image features. |
| `waves.py` | PLANNED | Single- and multi-wave frontier proposal. |
| `scheduler.py` | PLANNED | Complementarity, overlap, load, and transition-cost-aware global selection. |
| `controller.py` | PLANNED | Score, legality check, budget check, commit, and action record. It does not assimilate pixels itself. |
| `stopping.py` | PLANNED | Persistent stopping conditions and `CONVERGED`, `INSUFFICIENTLY_OBSERVED`, `NO_CANDIDATES`, or `INVALID_STATE` status. |

The initial final-system implementation should support analytic routing before a
learned router. Learned counterfactual gain requires an explicitly isolated
simulator or audit protocol.

### 7.13 `src/smagm/reconstruction/`

Owns high-level queries from a complete patient state and creation of serialized
reconstruction outputs.

| File | Status | Responsibility |
|---|---|---|
| `plane.py` | PLANNED | Render a requested modality and physical plane from structural and volumetric banks, returning support and uncertainty diagnostics. |
| `volume.py` | PLANNED | Chunked full-grid reconstruction with spatial culling and preserved affine/orientation. |
| `field.py` | PLANNED | Structural field, gradients, normals, support, and optional surface extraction. |
| `uncertainty.py` | PLANNED | Compose distance, coverage, missing-modality, disagreement, residual, propagation, and calibrated components. |
| `package.py` | PLANNED | Build an immutable final reconstruction package with status, budget, state hashes, and trajectory. |
| `export.py` | PLANNED | NIfTI/JSON/tensor export without losing physical metadata or scientific provenance. |

This package generates predictions only. It must not load audit ground truth.

### 7.14 `src/smagm/evaluation/`

Owns evaluation after reconstruction has been serialized. It must not receive a
mutable patient state or participate in training decisions for the sealed audit
cohort.

| File | Status | Responsibility |
|---|---|---|
| `metrics.py` | PLANNED | MAE, NMSE, PSNR, SSIM, NCC, gradient, frequency, edge, and local-contrast metrics. |
| `medical_fidelity.py` | PLANNED | Lesion/ROI/boundary metrics and frozen downstream-model fidelity. |
| `budget.py` | PLANNED | Quality-budget, quality-latency, AUC, and target-quality slice count. |
| `uncertainty.py` | PLANNED | Calibration, coverage-risk, unsupported-region, and error-stratification analysis. |
| `audit.py` | PLANNED | Isolated loading of dense targets and labels after artifact serialization. |
| `statistics.py` | PLANNED | Patient-level paired summaries, confidence intervals, seed aggregation, and declared tests. |

### 7.15 `src/smagm/baselines/`

Owns matched alternatives used for attribution, not the main method.

| File | Status | Responsibility |
|---|---|---|
| `fixed_support.py` | IMPLEMENTED | Deterministic, value-independent, aligned supports shared across E0/E1/E2. |
| `fixed_gaussian.py` | IMPLEMENTED | Safe fixed-topology feature-to-Gaussian bridge for attribution. |
| `interpolation.py` | PLANNED | Sparse-slice interpolation floors under the same physical grid. |
| `free_gaussian.py` | PLANNED | Gaussian representation without anchor/field constraints. |
| `selection.py` | PLANNED | Uniform, random, and metadata-balanced observation selection. |
| `dense_reconstruction.py` | PLANNED | Declared dense voxel/convolutional comparator under matched observation access. |

Baselines must use the same legal observations, renderer or declared forward
operator, output grid, and accounting rules required by their comparison.

### 7.16 `src/smagm/cli/` and `scripts/`

Final CLIs are thin orchestration layers. Scientific logic belongs in packages.

| File | Status | Responsibility |
|---|---|---|
| `cli/t1a.py` | DIAGNOSTIC | Synthetic analytic-to-fixed-Gaussian contract check. Retain as a bounded diagnostic or later move under `cli/debug/`; do not treat it as final architecture. |
| `cli/t1b.py` | DIAGNOSTIC | Synthetic E0/E1/E2 render/backward contract check. It is not the legal joint trainer. |
| `cli/train.py` | PLANNED | Parse resolved config and call `training.Trainer`. |
| `cli/reconstruct.py` | PLANNED | Build/update patient state and serialize a reconstruction package. |
| `cli/evaluate.py` | PLANNED | Evaluate serialized predictions on a declared non-sealed or sealed cohort. |
| `cli/audit.py` | PLANNED | Verify hashes, opened-file ledgers, environment, and claim evidence. |
| `scripts/*.py` | PLANNED | Minimal executable wrappers only; no duplicate training or model logic. |

### 7.17 Quality and phase-gate governance infrastructure

The quality layer governs development and research evidence; it is not part of
the `src/smagm` runtime architecture.

| Path | Responsibility |
|---|---|
| `quality/checklists.json` | Machine-readable invariant, contract, evidence, and Human Gate requirements. |
| `scripts/check_phase.py` | Evidence runner that validates the catalog and executes declared checks; it is not scientific model code. |
| `tests/quality/` | Catalog and runner contract tests. |
| `docs/checklists/` | Human-readable phase-gate documentation and interpretation rules. |
| `quality/reports/` | Generated local JSON/Markdown evidence; ignored by Git. |

Quality tooling may inspect and execute tests, but it must not import or mutate
patient state. It cannot authorize a phase or write a Human Gate decision.
Implementation PRs must update the relevant checklist evidence whenever a
public interface or test path changes.

## 8. Dependency direction

Allowed high-level dependency direction:

```text
contracts
├── data
├── features
├── gaussians
└── low-level renderer

features → contracts
anchors → contracts + features
fields → contracts + anchors
memory → contracts + anchors + fields + gaussians
state → contracts + features + anchors + fields + memory
routing → contracts + state + reconstruction diagnostics
reconstruction → contracts + state + renderer
training → all model/data packages + losses
baselines → contracts + selected low-level packages
evaluation → serialized outputs + contracts only
cli/scripts → training, routing, reconstruction, evaluation
```

Forbidden dependencies:

- `contracts/` importing model or orchestration packages;
- `features/` importing anchors, memory, routing, or training;
- `renderer.py` mutating an episode ledger;
- `routing/` reading candidate pixels;
- `reconstruction/` loading audit targets;
- `evaluation/` modifying patient state;
- any lower-level package importing `training/` or CLI code.

## 9. Core cross-module contracts

### 9.1 Physical coordinates

- Patient geometry uses canonical RAS millimetres.
- Plane tensors use `[H, W]` with `[v, u]` indexing.
- Every feature grid exposes an explicit transform to its bound physical plane.
- Slice thickness and through-plane profile are explicit.
- Source-affine provenance is retained for non-synthetic observations.

### 9.2 Observation legality

- Permanent availability is separate from episode role.
- Only context or committed observations may enter patient state.
- Target geometry may be exposed before prediction; target pixels may not.
- Prediction receipt registration precedes target reveal.
- Unqueried candidate pixels never enter routing descriptors.
- Dense audit targets are opened only by an isolated evaluator.

### 9.3 Parameter ownership

Global trainable state may include encoder, local-field, aggregation/update,
uncertainty-calibration, and authorized routing parameters.

Patient-specific state includes cache, anchors, fields, Gaussian memory,
observability, uncertainty, residuals, and trajectory. It is created and updated
per patient and is not registered as global trainable model state at inference.

### 9.4 Encoder and cache

- One encoding pass per committed slice at patient inference.
- Cache keys bind pixels/preprocessing, plane, transform, encoder config/state,
  dtype, channels, and valid topology.
- Anchor and topology updates sample cached maps rather than rerunning the
  encoder.

### 9.5 Anchors, fields, and memory

- Anchors are geometric reference supports, not Gaussians.
- The tiny MLP receives only local coordinates and already aggregated compact
  evidence.
- Structural and volumetric Gaussian banks remain distinguishable.
- Geometry updates are more conservative than modality appearance updates.
- Propagated support carries parent provenance and increasing uncertainty.

### 9.6 Rendering and reconstruction

- MRI slices are physical planes, not perspective views.
- The low-level renderer is pure and differentiable.
- Unsupported output remains explicit; it is not silently filled with
  confidence.
- Full-volume reconstruction is chunked and preserves affine metadata.

### 9.7 Scientific claims

- Unit tests prove software contracts, not reconstruction quality.
- Synthetic demos prove execution and gradients, not medical validity.
- A structural warm-up loss does not prove useful representation.
- A true-SDF name requires sign, distance, and Eikonal evidence.
- A final claim requires matched baselines, patient-level statistics, medical
  fidelity, compute accounting, and isolated audit evidence.

## 10. Execution modes

### 10.1 Offline teacher-free training

Global weights are trainable. Patient state is rebuilt per legal episode.
Gradients may pass through encoder, bootstrap, field, Gaussian state, and
renderer according to the authorized experiment.

### 10.2 New-patient static reconstruction

Global weights are frozen. Legal bootstrap observations are encoded once,
patient state is created, and reconstruction is produced without per-patient
network fine-tuning unless a separate authorized study explicitly tests it.

### 10.3 Active acquisition

Global weights remain frozen. The router scores legal metadata/state-derived
candidates, commits an observation, renders before update, loads pixels,
appends the cache, and updates only affected patient state.

### 10.4 Final reconstruction and audit

The reconstruction process serializes outputs first. A separate evaluation
process then opens audit ground truth and computes metrics. Audit targets do not
flow back into mutable state, selection, or training.

## 11. Configuration and experiment ownership

Configuration is declarative and resolved before execution.

```text
configs/data/             dataset, cohort, manifest, registration
configs/model/            encoder, anchors, field, memory, renderer
configs/training/         objective, optimizer, schedule, precision
configs/routing/          candidate, utility, wave, stopping
configs/reconstruction/   grid, chunking, export, uncertainty
configs/experiments/      named matched experiment compositions
```

Every experiment writes an immutable manifest containing at least:

- repository commit and dirty-state check;
- resolved configuration hash;
- environment and dependency lock;
- data, cohort, manifest, split, and episode-assignment hashes;
- model and patient-state schema versions;
- random seeds;
- hardware and precision;
- support/primitive counts and renderer profile;
- artifact hashes and output locations;
- runtime and memory accounting.

## 12. Testing architecture

Tests mirror package ownership. Each scientific contract requires a focused test
near its implementation and an integration test across its legal boundary.

Minimum final groups:

- geometry and affine reference tests;
- manifest, split, episode, receipt, reveal, and audit-isolation tests;
- analytic feature and feature-grid alignment tests;
- encoder equivariance, modality conditioning, anti-collapse, and gradient tests;
- cache exactness and forbidden target-derived insertion tests;
- anchor lifting, consolidation, frame, and evidence aggregation tests;
- local-field blending, overlap, gradient, and naming-gate tests;
- structural/volumetric memory initialization and topology tests;
- plane/slab renderer analytic-reference and chunking tests;
- legal joint training-step and target-never-enters-state tests;
- router no-pixel-access, budget, redundancy, wave, and stopping tests;
- volume affine, unsupported-region, uncertainty, and export round trips;
- isolated evaluation and patient-level statistics tests;
- end-to-end synthetic reconstruction with deterministic provenance.

Use small CPU analytic or synthetic references before large medical-data runs.

## 13. Current implementation map

At the time this file was introduced, `main` contains the following executable
foundation:

```text
IMPLEMENTED
├── physical coordinates and plane contracts
├── sparse availability, episode assignment, receipt, reveal, and cost contracts
├── low-level Gaussian tensor and amplitude-gauge contracts
├── through-plane profile-aware Gaussian reference renderer
├── analytic feature bank and feature-grid contracts
├── E0/E1/E2 encoder output contracts
├── exact feature-cache contract
├── fixed support and fixed-topology Gaussian attribution baseline
├── initial structural-loss components
├── T1-A and T1-B synthetic diagnostic CLIs
└── focused CPU tests

NOT YET COMPLETE
├── full structural warm-up experiment
├── explicit modality conditioning selected by ablation
├── legal joint sparse reconstruction trainer
├── anchor bootstrap and evidence aggregation
├── shared tiny local structural field
├── dual-bank Gaussian memory and propagation
├── active routing and local assimilation
├── full-volume reconstruction and export
└── isolated medical-fidelity and final-audit evaluation
```

## 14. Phase-to-code implementation map

This mapping controls sequencing only.

| Phase | Allowed architectural subset |
|---|---|
| T0 | `contracts/coordinates.py`, `gaussians.py`, low-level `renderer.py` |
| T0.5 | observation, manifest, episode, receipt, cost, split, and legality tests |
| T1 | `features/`, `losses/`, fixed baselines, legal `training/`, matched configs and evaluation |
| T2 | `anchors/`, `fields/`, memory initialization, state builder |
| T3 | memory propagation, assimilation, topology, observability, state updates |
| T4 | `routing/`, active controller, stopping, quality-budget experiments |
| T5 | `reconstruction/`, serialization/export, isolated `evaluation/` and audit |

A phase implementation should modify the stable files above. It should not
invent a separate phase-shaped subsystem.

## 15. Rules for adding or changing files

1. Read the corresponding reconstruction module or phase document first.
2. Identify the stable owner in this file.
3. Extend an existing file when the responsibility is the same; create a new
   file only for a distinct, testable responsibility.
4. Do not add generic `utils.py`, `helpers.py`, or duplicate contract types when
   a scientific owner exists.
5. Do not place model logic in CLI, scripts, notebooks, or experiment configs.
6. Do not create empty future packages or placeholder APIs before authorization.
7. Preserve dependency direction and avoid cyclic scientific ownership.
8. Add focused tests and update the nearest executable handoff.
9. Update `CODEBASE.md` only when final ownership or interfaces change, not for
   routine phase progress.
10. Update `docs/codex/README.md` when executable status changes.

## 16. Definition of a complete research codebase

The project is structurally complete only when:

- every reconstruction module has one stable software owner;
- the same legal data and patient-state interfaces support training and
  inference without duplicated logic;
- global parameters and patient state are explicitly separated;
- context-to-target training enforces prediction-before-reveal end to end;
- the encoder runs once per committed inference slice;
- anchors, fields, dual Gaussian banks, propagation, and routing are separately
  ablatable;
- full volumes are reconstructed from patient state rather than hidden targets;
- audit evaluation runs on serialized predictions in an isolated process;
- every headline result maps to immutable experiment and artifact provenance;
- phase documents can be removed from runtime context without making the final
  code architecture ambiguous.
