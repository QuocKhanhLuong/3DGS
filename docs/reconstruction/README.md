# Sparse Active Multi-Sequence MRI 3D Reconstruction

## Status

This directory defines the current research direction after the task was narrowed to **patient-specific 3D reconstruction from sparse, actively selected multi-sequence MRI slices**.

### Decisions currently locked

1. **Primary task:** reconstruct registered 3D MRI volumes from sparse sequence–slice observations.
2. **Core training regime:** direct sparse episodic training; no teacher distillation in the main path.
3. **Observation legality:** only committed sequence–slice observations may enter the patient state.
4. **Local field decoder:** one shared tiny MLP with the smallest possible role.
5. **Patient state:** adaptive anchors, local fields, Gaussian primitives, observability, and cached evidence.
6. **Inference:** model weights are frozen; only patient-specific state is updated.

### Decisions intentionally left open

- exact evidence encoder family;
- whether initial structural candidates come from fixed differential filters, a shallow CNN, or a hybrid;
- exact parameterization of volumetric appearance Gaussians;
- analytic versus learned trajectory utility;
- whether geometry is represented by one SDF, a local SDF bundle, or a more general structural level-set field.

The interfaces are written so these choices can be compared without changing the whole system.

---

## Document map

### End-to-end specification

- [`FULL_FLOW.md`](FULL_FLOW.md) — complete offline-training, initialization, active-update, and final-reconstruction flow.

### Phase specifications

- [`phases/01_DIRECT_SPARSE_TRAINING.md`](phases/01_DIRECT_SPARSE_TRAINING.md)
- [`phases/02_INITIAL_ANCHOR_BOOTSTRAP.md`](phases/02_INITIAL_ANCHOR_BOOTSTRAP.md)
- [`phases/03_ACTIVE_TRAJECTORY_UPDATE.md`](phases/03_ACTIVE_TRAJECTORY_UPDATE.md)
- [`phases/04_FINAL_RECONSTRUCTION.md`](phases/04_FINAL_RECONSTRUCTION.md)

### Module architecture specifications

- [`modules/EVIDENCE_ENCODER.md`](modules/EVIDENCE_ENCODER.md)
- [`modules/ANCHOR_LOCAL_FIELD.md`](modules/ANCHOR_LOCAL_FIELD.md)
- [`modules/SDF_GAUSSIAN_MEMORY.md`](modules/SDF_GAUSSIAN_MEMORY.md)
- [`modules/TRAJECTORY_ROUTER.md`](modules/TRAJECTORY_ROUTER.md)
- [`modules/PLANE_RENDERER_RECONSTRUCTOR.md`](modules/PLANE_RENDERER_RECONSTRUCTOR.md)

---

## Core problem formulation

For a patient, let the registered multi-sequence volume collection be

\[
\mathcal V=\{V^m\mid m\in\mathcal M\},
\]

where a candidate observation is a physical MRI slice

\[
a=(m,z).
\]

The full collection exists on disk for training supervision and evaluation, but the model state at time \(t\) may depend only on the committed observation set

\[
\mathcal O_t=\{a_1,\ldots,a_t\}.
\]

The reconstruction objective is to build a patient-specific continuous representation

\[
\mathcal S_t=(F_t,\mathcal A_t,\mathcal G_t,\mathcal C_t,\mathcal U_t)
\]

from which each modality can be reconstructed at arbitrary physical coordinates:

\[
\hat V_t^m(\mathbf x)=R_m(\mathbf x;\mathcal S_t).
\]

The trajectory chooses the next legal slice to maximize reconstruction improvement under a finite observation budget.

---

## Training-data clarification

Direct sparse training does **not** mean that most training slices can never appear across the entire optimization run. It means that within one patient episode:

- only a small observed subset is encoded into the state;
- only a small hidden subset of slices or 3D points is loaded as reconstruction supervision;
- hidden targets cannot influence anchor creation, routing, or memory updates;
- the model never receives the complete patient volume as one input.

Across epochs, different subsets may be sampled. The scientific claim concerns the observation budget and information available **per episode**, not whether a training file was ever sampled during the lifetime of training.

---

## One-line architecture

```text
Sparse observed MRI planes
→ compact evidence cache
→ provisional physical anchors
→ shared tiny anchor-local field
→ SDF-constrained structural Gaussians
  + volumetric appearance Gaussians
→ active sequence–slice trajectory
→ incremental patient-state updates
→ continuous multi-sequence 3D reconstruction
```
