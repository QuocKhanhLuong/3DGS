# Sparse Active Multi-Sequence MRI 3D Reconstruction

## Status

This directory defines the current research direction for **patient-specific 3D reconstruction from sparse, actively selected multi-sequence MRI slices**.

### Decisions currently locked

1. **Primary task:** reconstruct registered 3D MRI volumes from sparse sequence–slice observations.
2. **Main training regime:** permanently sparse patient supervision; no complete-volume targets in the main path.
3. **Encoder:** analytic differential scaffold plus a shared high-resolution micro-CNN.
4. **No teacher distillation:** teacher or pretrained dense encoders are upper-bound ablations only.
5. **Observation legality:** only committed or context sequence–slice observations may enter the patient state.
6. **Local field decoder:** one shared tiny MLP with the smallest possible role.
7. **Patient state:** adaptive anchors, local fields, structural and volumetric Gaussian primitives, observability, and cached evidence.
8. **Inference:** model weights are frozen; only patient-specific state is updated.

### Decisions intentionally left open

- exact analytic channels and micro-CNN capacity;
- exact encoder output stride in \(\{1,2,4\}\);
- exact parameterization of volumetric appearance Gaussians;
- analytic versus learned trajectory utility;
- whether geometry is represented by a true SDF, a local SDF bundle, or a structural level-set field;
- exact protocol used to train or audit active routing under permanently sparse supervision.

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

At deployment, the trajectory chooses the next legal slice to maximize reconstruction improvement under a finite observation budget.

---

## One-line architecture

```text
Sparse observed MRI planes
→ analytic structural channels
→ shared high-resolution micro-CNN
→ compact structural and appearance cache
→ provisional physical anchors
→ shared tiny anchor-local field
→ SDF/level-set constrained structural Gaussians
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

A physically isolated fully sampled audit split may be used for:

- final reconstruction metrics;
- leakage checks;
- oracle trajectory studies;
- privileged-training upper bounds clearly labeled as ablations.
