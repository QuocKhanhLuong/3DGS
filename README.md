# SDF-Manifold Active Gaussian Memory

Research repository for an ISBI-oriented medical imaging project that builds a patient-specific Gaussian representation from sparse, adaptively selected multi-sequence MRI observations.

## Core idea

The method links two main contributions:

1. **SDF/level-set adaptive low-DoF Gaussian prior** — a structural field derives Gaussian orientation and constrains position/covariance instead of optimizing free position–quaternion–scale parameters.
2. **Balanced multi-wave observability routing** — parallel information fronts select complementary sequence–slice observations over an evolving feature–geometry manifold and stop at an observability fixed point.

```text
Sparse observed MRI planes
    → analytic differential scaffold
    → teacher-free high-resolution micro-CNN
    → compact structural and appearance cache
    → adaptive anchor-local fields
    → SDF/level-set constrained structural Gaussians
      + volumetric appearance Gaussians
    → anchor–Gaussian propagation
    → observability graph
    → balanced multi-wave routing
    → incremental local assimilation
    → complete multi-sequence 3D reconstruction
    → uncertainty and trajectory record
```

## Start here

- [`docs/reconstruction/README.md`](docs/reconstruction/README.md) — current reconstruction-focused research package and decision index.
- [`docs/reconstruction/FULL_FLOW.md`](docs/reconstruction/FULL_FLOW.md) — complete four-phase flow.
- [`docs/reconstruction/PROOFREAD_NOTES.md`](docs/reconstruction/PROOFREAD_NOTES.md) — phase-by-phase review before implementation.
- [`MASTER_KNOWLEDGE.md`](MASTER_KNOWLEDGE.md) — earlier consolidated conclusions and formulation.
- [`KNOWLEDGE_PACKAGE.md`](KNOWLEDGE_PACKAGE.md) — earlier implementation and paper-development package.
- [`architecture.md`](architecture.md) — earlier block architecture and interfaces.
- [`pipeline.md`](pipeline.md) — earlier training and inference pipeline.

## Current primary task

**Budgeted active multi-sequence MRI 3D reconstruction from sparse sequence–slice observations.**

The main method learns from permanently sparse patient manifests. It does not require teacher distillation or complete-volume targets. Within a training episode, only context slices enter the patient state; acquired sparse target slices are revealed only after rendering. Fully sampled volumes, when available, are isolated for audit evaluation and privileged upper-bound ablations.

## Current locked decisions

- permanently sparse main training supervision;
- analytic differential scaffold plus teacher-free high-resolution micro-CNN;
- no teacher distillation in the main architecture;
- one shared tiny MLP for anchor-local structural-field decoding;
- cached evidence encoding once per queried slice;
- patient-specific adaptive anchors and Gaussian memory;
- structural surface Gaussians plus volumetric appearance Gaussians;
- physical-plane rendering rather than camera-view rendering;
- closed-loop active trajectory and explicit uncertainty.

## Reconstruction package map

| Document | Scope |
|---|---|
| `docs/reconstruction/FULL_FLOW.md` | Complete system flow and global state contracts |
| `docs/reconstruction/PROOFREAD_NOTES.md` | Four-phase review and code-entry checklist |
| `docs/reconstruction/phases/01_DIRECT_SPARSE_TRAINING.md` | Teacher-free permanently sparse training |
| `docs/reconstruction/phases/02_INITIAL_ANCHOR_BOOTSTRAP.md` | Initial observation selection, provisional anchors, local fields, and Gaussian initialization |
| `docs/reconstruction/phases/03_ACTIVE_TRAJECTORY_UPDATE.md` | Multi-wave query selection and incremental state update |
| `docs/reconstruction/phases/04_FINAL_RECONSTRUCTION.md` | Full-volume, arbitrary-plane, geometry, and uncertainty reconstruction |
| `docs/reconstruction/modules/EVIDENCE_ENCODER.md` | Teacher-free structural evidence encoder |
| `docs/reconstruction/modules/ANCHOR_LOCAL_FIELD.md` | Shared tiny local MLP and field blending |
| `docs/reconstruction/modules/SDF_GAUSSIAN_MEMORY.md` | Structural and volumetric Gaussian memory |
| `docs/reconstruction/modules/TRAJECTORY_ROUTER.md` | Reconstruction-driven active routing |
| `docs/reconstruction/modules/PLANE_RENDERER_RECONSTRUCTOR.md` | MRI plane rendering and continuous 3D output |

## Current implementation priority

1. sparse-only patient manifest loader and leakage tests;
2. analytic differential channel bank;
3. teacher-free structural/appearance micro-CNN;
4. structural warm-up losses and anti-collapse diagnostics;
5. physical-plane target prediction prototype;
6. anchor-local shared tiny MLP and local-field blending;
7. initial anchor bootstrap from sparse observed planes;
8. SDF/level-set constrained structural Gaussian state;
9. volumetric appearance Gaussian state;
10. physical-plane renderer and static sparse reconstruction baseline;
11. uncertainty and observability state;
12. single-wave and balanced multi-wave routing;
13. adaptive topology and local graph repair;
14. isolated full-volume audit evaluation.

## Status

Research design and pre-implementation proof-reading stage. High-fidelity full-volume reconstruction from a small query budget remains a hypothesis to be validated experimentally.
