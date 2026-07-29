# Full Flow — Direct Sparse Active 3D Reconstruction

## 1. Scope

This document defines the complete research flow for reconstructing registered multi-sequence MRI volumes from a small number of actively selected 2D slices.

The core method does not use teacher distillation. The encoder, shared local tiny MLP, update rules, and trajectory components are trained directly through sparse reconstruction episodes.

---

## 2. Highest-level flow

```text
PHASE 1 — DIRECT SPARSE EPISODIC TRAINING
Training patient on disk
→ sample sparse observed slices O
→ sample hidden reconstruction targets H
→ encode only O
→ create/update patient representation from O
→ render H
→ compare with hidden target pixels/points
→ backpropagate global model parameters

PHASE 2 — INITIAL PATIENT BOOTSTRAP
New patient candidate pool
→ metadata-only initial slice selection
→ commit and load K0 slices
→ encode each committed slice once
→ cache compact evidence maps
→ lift structural candidates to physical 3D anchors
→ build local fields with shared tiny MLP
→ initialize structural and appearance Gaussian memory

PHASE 3 — ACTIVE TRAJECTORY UPDATE
Current patient representation
→ estimate candidate reconstruction gain
→ select one or a small batch of new slices
→ render-before-update
→ load and encode selected slices once
→ calculate residuals
→ update local evidence, fields, anchors, Gaussians, and uncertainty
→ repair trajectory graph locally
→ repeat until stopping or budget exhaustion

PHASE 4 — FINAL 3D RECONSTRUCTION
Final patient representation
→ evaluate continuous fields on target physical grid
→ reconstruct every requested MRI modality
→ reconstruct geometry and uncertainty
→ export full volumes, slices, surfaces, and trajectory record
```

---

## 3. Global entities

### 3.1 Global trainable parameters

These parameters are shared across patients and learned offline:

```text
Theta_global
├── evidence encoder parameters theta_E
├── shared local tiny-MLP parameters theta_F
├── optional evidence aggregation parameters theta_A
├── Gaussian update-rule parameters theta_U
├── optional trajectory utility parameters theta_Q
└── optional uncertainty calibration parameters theta_C
```

At inference, these parameters are frozen.

### 3.2 Patient-specific state

These values are created separately for each patient:

```text
PatientState_t
├── observation ledger O_t
├── compact feature cache C_t
├── representation anchors A_t
├── local structural fields F_t
├── structural Gaussian bank G_t^surface
├── volumetric appearance bank G_t^volume
├── modality observability state M_t
├── reconstruction uncertainty U_t
├── residual history E_t
└── trajectory history tau_t
```

Patient-specific state is updated during the active trajectory. It is not a new set of global network weights.

---

## 4. Observation legality

The complete patient dataset is

\[
\Omega=\{(m,z)\},
\]

but at time \(t\), only committed observations

\[
\mathcal O_t\subset\Omega
\]

may contribute pixels or learned features to the patient representation.

For an unqueried candidate, the router may use only legal descriptors such as:

- modality identity;
- physical plane origin, normal, spacing, and thickness;
- distance to current anchors or uncertain regions;
- current representation-derived predictions on that plane;
- population-level priors learned only from the training cohort.

It may not use unqueried image pixels, target volume intensities, or labels.

---

## 5. Direct sparse training flow

For each training episode:

### Step 1 — Select patient and budget

Choose one training patient and an observation budget \(B\).

### Step 2 — Sample observed input set

Sample or route a small legal observation set

\[
\mathcal O=\{(m_1,z_1),\ldots,(m_B,z_B)\}.
\]

Only these slices enter the evidence encoder.

### Step 3 — Sample hidden supervision

Choose a disjoint hidden target set

\[
\mathcal H\subset\Omega\setminus\mathcal O
\]

or sample physical 3D target points. Hidden targets are used only after the representation has been built from \(\mathcal O\).

### Step 4 — Build patient state

```text
Observed slices O
→ evidence encoder
→ compact cached feature maps
→ provisional anchors
→ anchor-local evidence aggregation
→ shared tiny local fields
→ structural + volumetric Gaussian memory
```

### Step 5 — Render hidden targets

For every hidden plane or 3D point, render each required modality from the current representation.

### Step 6 — Compute losses

Recommended loss families:

\[
\mathcal L=
\lambda_{pix}\mathcal L_{pixel}
+\lambda_{str}\mathcal L_{structural}
+\lambda_{freq}\mathcal L_{frequency}
+\lambda_{field}\mathcal L_{field}
+\lambda_{reg}\mathcal L_{regularization}
+\lambda_{cal}\mathcal L_{calibration}.
\]

Possible terms:

- robust L1 or Charbonnier intensity loss;
- SSIM or local structural loss;
- gradient and edge reconstruction loss;
- low/high-frequency consistency;
- Eikonal or local field smoothness regularization;
- Gaussian compactness, overlap, and manifold penalties;
- uncertainty calibration against observed reconstruction error.

### Step 7 — Backpropagate

Update global model parameters. Do not optimize hidden target pixels into the patient state before rendering them.

---

## 6. Initial patient bootstrap flow

### Step 1 — Metadata-only initial observation selection

Select \(K_0\) spatially and cross-modally diverse candidate slices without loading their pixels.

### Step 2 — Commit and encode

For each selected slice:

```text
commit observation
→ load pixels
→ run evidence encoder once
→ store compact feature map and reliability map
```

### Step 3 — Generate provisional anchors

Detect sparse structural candidate locations on observed planes, convert their pixel coordinates to physical 3D coordinates, merge redundant candidates, and create provisional anchors.

### Step 4 — Aggregate local evidence

Each anchor samples all relevant cached planes and obtains a compact evidence vector that includes feature, modality, plane distance, orientation, and reliability information.

### Step 5 — Decode local fields

The shared tiny MLP receives only:

\[
[\boldsymbol\xi_i(\mathbf x),\mathbf h_i],
\]

where \(\boldsymbol\xi_i\) is the local anchor coordinate and \(\mathbf h_i\) is compact anchor evidence. It predicts one local structural field value.

### Step 6 — Blend into a structural scaffold

Local fields are blended with compact support weights to form a continuous patient-specific field.

### Step 7 — Refine anchors and initialize Gaussians

Project anchors toward stable level sets, derive local orientation, and initialize:

- thin structural Gaussians near important interfaces;
- larger volumetric Gaussians for interior appearance reconstruction.

---

## 7. Active trajectory flow

At round \(t\):

### Step 1 — Predict candidate utility

Estimate how much each legal unqueried slice may improve reconstruction. Candidate utility may combine:

\[
Q_t(a)=
\alpha_U U_t(a)
+\alpha_C C_t(a)
+\alpha_M M_t(a)
+\alpha_R R_t^{pred}(a)
-\alpha_D D_t(a).
\]

Interpretation:

- \(U_t\): current uncertainty intersected by the plane;
- \(C_t\): uncovered representation mass;
- \(M_t\): missing modality evidence;
- \(R_t^{pred}\): predicted residual or information gain;
- \(D_t\): redundancy and acquisition cost.

### Step 2 — Select and commit action

Choose one slice or a complementary small batch. Only after commitment are pixels loaded.

### Step 3 — Render before update

Render the selected physical plane from the current state:

\[
\hat I_t=R(\mathcal S_t,a_t).
\]

### Step 4 — Encode observation and calculate residual

\[
E_t=I_t-\hat I_t.
\]

The new slice is encoded once and appended to the evidence cache.

### Step 5 — Update affected state locally

Only anchors and Gaussians whose support intersects the new plane receive large updates.

Potential operations:

- update anchor evidence;
- move anchor along its local frame;
- change local support scale;
- birth support in unexplained regions;
- split heterogeneous primitives;
- prune unsupported primitives;
- update modality appearance values;
- update uncertainty and observability.

### Step 6 — Replan

Repair only affected graph regions when possible, then select the next observation.

---

## 8. Stopping conditions

A successful stop should require persistence over multiple rounds. Candidate conditions include:

- maximum predicted reconstruction gain below threshold;
- hidden-like self-consistency residual below threshold on legal validation projections;
- relative field change below threshold;
- anchor and Gaussian topology stable;
- worst relevant uncertainty below threshold;
- no legal candidate remaining.

Budget exhaustion without convergence must be reported as insufficient observation rather than convergence.

---

## 9. Final reconstruction flow

For a requested physical grid \(\mathcal X\):

### Geometry

Evaluate the blended structural field and optional zero level sets.

### Appearance

For modality \(m\), reconstruct intensity through normalized Gaussian composition:

\[
\hat V^m(\mathbf x)=
\frac{\sum_i w_i(\mathbf x)c_{i,m}}
{\sum_i w_i(\mathbf x)+\epsilon}.
\]

A small residual appearance decoder may be tested, but the default design keeps reconstruction as direct Gaussian composition to avoid introducing another large bottleneck.

### Uncertainty

Export uncertainty derived from evidence distance, modality missingness, Gaussian disagreement, residual history, and optional learned calibration.

### Outputs

```text
ReconstructionOutput
├── reconstructed volume for every requested modality
├── arbitrary reconstructed physical slices
├── structural field / surfaces
├── uncertainty volume
├── final anchors and Gaussian state
├── observation trajectory
└── per-round quality and compute diagnostics
```

---

## 10. Core research hypotheses

1. Sparse direct episodic training can learn a patient-specific reconstruction field without full-volume input in any episode.
2. A shared low-capacity anchor-local decoder is sufficient when physical geometry and local evidence are organized correctly before decoding.
3. SDF-constrained structural Gaussians reduce geometric freedom and improve stability under sparse observations.
4. A separate volumetric appearance bank is necessary for faithful internal MRI reconstruction.
5. Closed-loop active querying improves quality–budget curves relative to fixed uniform and uncertainty-only sampling.
6. Adaptive anchors and multi-wave routing reduce the number of queried slices required to reach a target reconstruction quality.
