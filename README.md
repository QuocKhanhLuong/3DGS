# SDF-Manifold Active Gaussian Memory

Research repository for an ISBI-oriented medical imaging project that builds a patient-specific Gaussian representation from sparse, adaptively selected multi-sequence MRI observations.

## Core idea

The method links two main contributions:

1. **SDF/3D-SLNR-adaptive low-DoF Gaussian prior** — an anatomical structural field derives Gaussian orientation and constrains position/covariance instead of optimizing free position–quaternion–scale parameters.
2. **Balanced multi-wave observability routing** — parallel information fronts select complementary sequence–slice observations over an evolving feature–geometry manifold and stop at an observability fixed point.

```text
Sparse observed MRI planes
    → compact evidence cache
    → adaptive anchor-local fields
    → SDF-constrained structural Gaussians
      + volumetric appearance Gaussians
    → observability graph
    → balanced multi-wave routing
    → incremental local assimilation
    → complete multi-sequence 3D reconstruction
    → uncertainty and trajectory record
```

## Start here

- [`docs/reconstruction/README.md`](docs/reconstruction/README.md) — current reconstruction-focused research package and decision index.
- [`docs/reconstruction/FULL_FLOW.md`](docs/reconstruction/FULL_FLOW.md) — complete direct sparse training, initialization, trajectory, and final reconstruction flow.
- [`MASTER_KNOWLEDGE.md`](MASTER_KNOWLEDGE.md) — earlier consolidated conclusions, formulation, convergence, task, losses, metrics, and ablations.
- [`KNOWLEDGE_PACKAGE.md`](KNOWLEDGE_PACKAGE.md) — earlier navigable implementation and paper-development package.
- [`architecture.md`](architecture.md) — earlier block architecture and interfaces.
- [`pipeline.md`](pipeline.md) — earlier training and inference pipeline.

## Current primary task

**Budgeted active multi-sequence MRI 3D reconstruction from sparse sequence–slice observations.**

The model directly learns from sparse patient episodes without teacher distillation in the core path. In one episode, only a small observed subset is encoded into the patient state. A separate hidden subset of slices or 3D points may be loaded only as reconstruction supervision. The complete registered volume is never provided as one model input.

## Current locked decisions

- direct sparse episodic training;
- no teacher distillation in the main architecture;
- one shared tiny MLP for anchor-local structural-field decoding;
- cached evidence encoding once per queried slice;
- patient-specific adaptive anchors and Gaussian memory;
- structural surface Gaussians plus volumetric appearance Gaussians;
- physical-plane rendering rather than camera-view rendering;
- closed-loop active trajectory and explicit uncertainty.

The exact low-FLOP evidence encoder remains an open experimental decision.

## Reconstruction package map

| Document | Scope |
|---|---|
| `docs/reconstruction/FULL_FLOW.md` | Complete system flow and global state contracts |
| `docs/reconstruction/phases/01_DIRECT_SPARSE_TRAINING.md` | Direct sparse episodic optimization without distillation |
| `docs/reconstruction/phases/02_INITIAL_ANCHOR_BOOTSTRAP.md` | Initial observation selection, provisional anchors, local fields, and Gaussian initialization |
| `docs/reconstruction/phases/03_ACTIVE_TRAJECTORY_UPDATE.md` | Multi-wave query selection and incremental state update |
| `docs/reconstruction/phases/04_FINAL_RECONSTRUCTION.md` | Full-volume, arbitrary-plane, geometry, and uncertainty reconstruction |
| `docs/reconstruction/modules/EVIDENCE_ENCODER.md` | Replaceable low-FLOP encoder contract and candidate families |
| `docs/reconstruction/modules/ANCHOR_LOCAL_FIELD.md` | Shared tiny local MLP and field blending |
| `docs/reconstruction/modules/SDF_GAUSSIAN_MEMORY.md` | Structural and volumetric Gaussian memory |
| `docs/reconstruction/modules/TRAJECTORY_ROUTER.md` | Reconstruction-driven active routing |
| `docs/reconstruction/modules/PLANE_RENDERER_RECONSTRUCTOR.md` | MRI plane rendering and continuous 3D output |

## Current implementation priority

1. registered MRI slice provider and physical metadata;
2. sparse episode sampler with observed/hidden-role enforcement;
3. physical-plane Gaussian renderer;
4. anchor-local shared tiny MLP and local-field blending;
5. SDF-constrained structural Gaussian state;
6. volumetric appearance Gaussian state;
7. initial anchor bootstrap from sparse observed planes;
8. reconstruction losses and budgeted evaluation;
9. uncertainty and observability state;
10. single-wave and balanced multi-wave routing;
11. adaptive topology and local graph repair.

## Status

Research design and implementation planning stage. Claims such as high-fidelity full-volume reconstruction from a small query budget remain hypotheses to be validated experimentally.
