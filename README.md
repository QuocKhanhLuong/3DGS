# Sparse Multi-Sequence MRI Gaussian Reconstruction

Research repository for an ISBI 2027 medical-imaging project that reconstructs
a patient-specific 3D Gaussian representation from permanently sparse
multi-sequence MRI observations.

## Core idea

The primary thesis is a static sparse support-anchor representation:

```text
permanently sparse multi-sequence MRI slices
    → analytic differential scaffold
    → compact teacher-free structural evidence
    → physical support anchors
    → one shared tiny anchor-local structural field
    → anchored seed Gaussians
    → anchor–Gaussian propagation
    → latent patient-specific 3D Gaussian representation
    → full-volume reconstruction
```

T0 provides legal data access, physical geometry, and the through-plane
profile-aware Gaussian reference renderer. It is enabling infrastructure, not
the representation novelty. Active trajectory and adaptive acquisition are
later extensions after the static representation has passed matched baselines.

## Start here

- [`docs/strategies/2026-07-31-execution-status-addendum.md`](docs/strategies/2026-07-31-execution-status-addendum.md) — live executable status addendum; it does not rewrite the dated strategy.
- [`docs/reconstruction/README.md`](docs/reconstruction/README.md) — theoretical reconstruction backbone.
- [`CODEBASE.md`](CODEBASE.md) — final theory-to-code architecture, ownership, and dependency direction.
- [`docs/codex/README.md`](docs/codex/README.md) — executable handoffs and current software/Human Gate matrix.
- [`quality/README.md`](quality/README.md) and [`docs/checklists/PHASE_GATE_SYSTEM.md`](docs/checklists/PHASE_GATE_SYSTEM.md) — machine-readable and human-readable gate evidence.
- [`docs/reconstruction/FULL_FLOW.md`](docs/reconstruction/FULL_FLOW.md) — complete four-phase flow.
- [`docs/reconstruction/PROOFREAD_NOTES.md`](docs/reconstruction/PROOFREAD_NOTES.md) — phase-by-phase review before implementation.
- [`docs/strategies/2026-07-29-isbi-realignment.md`](docs/strategies/2026-07-29-isbi-realignment.md) — authoritative venue, thesis, tranche, gate, and claim policy.
- [`docs/plans/2026-07-29-t05-t1-teacher-free-encoder-fixed-gaussian-baseline.md`](docs/plans/2026-07-29-t05-t1-teacher-free-encoder-fixed-gaussian-baseline.md) — approved T0.5/T1 implementation plan.

The authority and reading order is:

```text
strategy/addendum → docs/reconstruction → CODEBASE.md → docs/codex → quality system
```

## Current primary task

**Legal episodic sparse training followed by a teacher-free encoder and
fixed-topology Gaussian baseline.**

The main method learns from permanently sparse patient manifests. It does not require teacher distillation or complete-volume targets. Within a training episode, only context slices enter the patient state; acquired sparse target slices are revealed only after rendering. Fully sampled volumes, when available, are isolated for audit evaluation and privileged upper-bound ablations.

## Current locked decisions

- permanently sparse main training supervision;
- episode context/target roles separated from permanent availability;
- render and prediction receipt registered before target reveal;
- analytic differential scaffold plus teacher-free high-resolution micro-CNN;
- no teacher distillation in the main architecture;
- one shared tiny MLP for anchor-local structural-field decoding;
- cached evidence encoding once per queried slice;
- patient-specific adaptive anchors and Gaussian memory;
- structural surface Gaussians plus volumetric appearance Gaussians;
- physical-plane rendering rather than camera-view rendering;
- explicit unsupported coverage and failure reporting;
- active routing deferred until the static representation passes its gates.

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

## Current implementation order

1. **T0 — implemented software contract:** legal physical operator.
2. **T0.5 — implemented software contract:** legal episodic contracts.
3. **T1-A — implemented software contract:** analytic evidence and fixed Gaussian reference.
4. **T1-B — implemented software tranche; Human Gate passed:** teacher-free encoder, cache, structural objectives, and fixed-topology baseline.
5. **T1-C — blocked:** legal episodic trainer and matched training experiments.
6. **T2–T5 — blocked:** anchors, fields, memory, routing, reconstruction, and isolated evaluation.

## Status

T0, T0.5, and T1-A are implemented software contracts. T1-B software is merged
and its Human Gate is passed. T1-C and T2+ remain blocked. T1-B software
completion is not reconstruction success; the representation remains a
hypothesis until the required matched evidence and Human Gate decisions exist.
