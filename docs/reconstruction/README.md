# Sparse Active Multi-Sequence MRI 3D Reconstruction

> **ISBI precedence notice (2026-07-29).** This package contains the full
> long-horizon T1–T5 architecture, but
> [`../strategies/2026-07-29-isbi-realignment.md`](../strategies/2026-07-29-isbi-realignment.md)
> governs thesis, tranche order, and claims. Static sparse support-anchor
> reconstruction is primary; active trajectory is a deferred T4 extension.
> Any CVPR-first or active-policy-first wording below is historical unless the
> ISBI strategy explicitly reauthorizes it.

## Status

This directory defines the long-horizon design for **patient-specific 3D
reconstruction from permanently sparse multi-sequence MRI slices**. It remains
the theoretical method backbone. The live executable software and Human Gate
state is maintained in [`docs/codex/README.md`](../codex/README.md); the
current status addendum is
[`docs/strategies/2026-07-31-execution-status-addendum.md`](../strategies/2026-07-31-execution-status-addendum.md).

The repository currently contains implemented software contracts for T0, T0.5,
and T1-A, plus the implemented T1-B software tranche with its Human Gate
pending. T1-C and T2+ remain blocked. Merged software is not scientific
validation or Human Gate approval.

### Decisions currently locked

1. **Primary task:** reconstruct registered 3D MRI volumes from sparse sequence–slice observations.
2. **Main training regime:** permanently sparse patient supervision; no complete-volume targets in the main path.
3. **Encoder:** analytic differential scaffold plus a shared high-resolution micro-CNN.
4. **No teacher distillation:** teacher or pretrained dense encoders are upper-bound ablations only.
5. **Observation legality:** availability is permanent; episode context/target
   roles are temporary, and target pixels require a renderer-minted prediction
   receipt before reveal.
6. **Local field decoder:** one shared tiny MLP with the smallest possible role.
7. **Static representation thesis:** physical supports, a shared tiny
   anchor-local structural field, anchored Gaussian birth and propagation form
   the T2–T3 hypothesis; they are not yet implemented.
8. **Inference:** model weights are frozen; only patient-specific state is
   updated once those future tranches are authorized.

### Implemented software contracts

- T1 encoder dimensions: configurable structural channels (default 16),
  appearance channels (default 8), and exactly one bounded reliability channel.
- Per-item feature-grid geometry with locked
  `align_corners=False` half-pixel semantics, strides 1/2/4, and explicit
  invalid right/bottom padding for odd shapes.
- Exact feature-cache key binding for observation, source plane, encoder
  variant/configuration/state, preprocessing, transform, valid mask, dtype, and
  output contract.
- Common E0/E1/E2 software path through deterministic fixed supports and the
  shared Gaussian head.

These are executable interface contracts, not evidence that one representation
is scientifically better.

### Decisions intentionally left open

- exact parameterization of volumetric appearance Gaussians;
- fair matched training protocol and real-data optimization;
- explicit modality conditioning selected by ablation;
- analytic versus learned trajectory utility;
- whether geometry is represented by a true SDF, a local SDF bundle, or a structural level-set field;
- exact protocol used to train or audit active routing under permanently sparse supervision.

The following remain incomplete or planned: modality conditioning, real-data
training, matched attribution experiments, support anchors, local fields,
propagation, routing, and T5 reconstruction/export/evaluation.

The downstream phase interfaces remain stable while these decisions are experimentally compared.

---

## Document map

### End-to-end specification

- [`FULL_FLOW.md`](FULL_FLOW.md) — complete teacher-free training, initialization, active-update, and final-reconstruction flow.
- [`PROOFREAD_NOTES.md`](PROOFREAD_NOTES.md) — phase-by-phase consistency review and code-entry checklist.

### Phase specifications

- [`phases/01_DIRECT_SPARSE_TRAINING.md`](phases/01_DIRECT_SPARSE_TRAINING.md) — teacher-free permanently sparse structural and reconstruction training.
- [`phases/02_INITIAL_ANCHOR_BOOTSTRAP.md`](phases/02_INITIAL_ANCHOR_BOOTSTRAP.md) — initial sparse evidence, anchors, local fields, and Gaussian initialization.
- [`phases/03_ACTIVE_TRAJECTORY_UPDATE.md`](phases/03_ACTIVE_TRAJECTORY_UPDATE.md) — active sequence–slice selection and incremental patient-state update.
- [`phases/04_FINAL_RECONSTRUCTION.md`](phases/04_FINAL_RECONSTRUCTION.md) — continuous full-volume reconstruction and uncertainty export.

### Module architecture specifications

- [`modules/EVIDENCE_ENCODER.md`](modules/EVIDENCE_ENCODER.md) — teacher-free structural evidence encoder.
- [`modules/ANCHOR_LOCAL_FIELD.md`](modules/ANCHOR_LOCAL_FIELD.md) — shared tiny local MLP and field blending.
- [`modules/SDF_GAUSSIAN_MEMORY.md`](modules/SDF_GAUSSIAN_MEMORY.md) — structural and volumetric Gaussian memory.
- [`modules/TRAJECTORY_ROUTER.md`](modules/TRAJECTORY_ROUTER.md) — reconstruction-driven active routing.
- [`modules/PLANE_RENDERER_RECONSTRUCTOR.md`](modules/PLANE_RENDERER_RECONSTRUCTOR.md) — MRI plane rendering and continuous 3D output.

---

## Core problem formulation

For training patient \(i\), the main method receives only a fixed sparse acquisition set

\[
\Omega_i^{sparse}
=
\{(a_{i,j},I_{i,j})\}_{j=1}^{K_i}.
\]

A Phase-1 episode uses a legal context–target split:

\[
\mathcal C_i\subset\Omega_i^{sparse},
\qquad
\mathcal Q_i\subset\Omega_i^{sparse}\setminus\mathcal C_i.
\]

Only \(\mathcal C_i\) may create the patient representation. The acquired sparse target pixels in \(\mathcal Q_i\) are revealed only after rendering.

The patient-specific continuous state is

\[
\mathcal S_t=(F_t,\mathcal A_t,\mathcal G_t,\mathcal C_t,\mathcal U_t),
\]

from which each modality is reconstructed at arbitrary physical coordinates:

\[
\hat V_t^m(\mathbf x)=R_m(\mathbf x;\mathcal S_t).
\]

In the current static program, reconstruction uses a declared permanently
sparse observation set. A future T4 deployment extension may choose additional
legal observations under an exact acquisition budget, but routing is not part
of T0.5/T1 and is not the current headline.

---

## One-line architecture

```text
Sparse observed MRI planes
→ analytic structural channels
→ shared high-resolution micro-CNN
→ compact structural and appearance cache
→ provisional physical anchors
→ shared tiny anchor-local field
→ StructuralField/level-set-constrained structural Gaussians
  + volumetric appearance Gaussians
→ anchor–Gaussian propagation
→ active sequence–slice trajectory
→ continuous multi-sequence 3D reconstruction
```

---

## Training and evaluation separation

Main training:

- uses permanently sparse patient manifests;
- does not open non-manifest slices;
- does not use teacher features;
- does not use complete-volume targets.

A patient-disjoint T1 lesion-validation evaluator may use a predeclared sparse
input manifest plus isolated dense targets/labels for its one-shot medical
fidelity gate. A separate sealed final-audit cohort remains reserved for T5.

Separate privileged/synthetic protocols may be used for leakage-positive
controls, future oracle trajectory studies, or E3/E4 upper bounds, but their
pixels, labels, checkpoints, and decisions cannot enter or select the main
training path.
