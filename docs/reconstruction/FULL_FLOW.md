# Full Flow — Teacher-Free Sparse Active 3D Reconstruction

## 1. Scope

This document defines the complete research flow for reconstructing registered multi-sequence MRI volumes from a small number of actively selected 2D slices.

The main training path uses:

- permanently sparse patient acquisitions;
- no teacher distillation;
- an analytic differential scaffold plus a high-resolution micro-CNN;
- sparse acquired target planes for supervision;
- the previously locked anchor, local-field, Gaussian, trajectory, and reconstruction modules.

Complete volumes may exist only in a separate audit/evaluation split or privileged upper-bound ablation. They are not main-training targets.

---

## 2. Highest-level flow

```text
PHASE 1 — TEACHER-FREE PERMANENTLY SPARSE TRAINING
Fixed sparse training-patient manifest Ω_i^sparse
→ split into context C_i and acquired sparse targets Q_i
→ encode only C_i with analytic scaffold + micro-CNN
→ create/update patient representation from C_i
→ render Q_i
→ reveal acquired target pixels
→ optimize structural encoder and global reconstruction parameters

PHASE 2 — INITIAL PATIENT BOOTSTRAP
New patient candidate pool
→ metadata-only initial slice selection
→ commit and load K0 slices
→ encode each committed slice once
→ cache compact structural and appearance maps
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
├── teacher-free evidence encoder parameters theta_E
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

### 4.1 Main training cohort

Training patient \(i\) exposes only a fixed sparse acquisition set:

\[
\Omega_i^{sparse}
=
\{(a_{i,j},I_{i,j})\}_{j=1}^{K_i}.
\]

Only files listed in the sparse manifest may be opened by the main training loader.

Within an episode:

\[
\mathcal C_i\subset\Omega_i^{sparse},
\qquad
\mathcal Q_i\subset\Omega_i^{sparse}\setminus\mathcal C_i.
\]

- context pixels may create encoder features, anchors, fields, and Gaussians;
- target coordinates may be known for rendering;
- target pixels may be revealed only after state construction;
- non-manifest slices remain inaccessible throughout training.

### 4.2 Patient inference and active acquisition

At inference round \(t\), only committed observations

\[
\mathcal O_t
\]

may contribute pixels or learned features to the patient representation.

For an unqueried candidate, the router may use only legal descriptors such as:

- modality identity;
- physical plane origin, normal, spacing, and thickness;
- distance to current anchors or uncertain regions;
- current representation-derived predictions on that plane;
- population-level priors learned from the legal training cohort.

It may not use unqueried image pixels, target volume intensities, or labels.

### 4.3 Full-volume audit data

Fully sampled volumes, when available, must be isolated from the main training loader. They may be used for:

- final reconstruction evaluation;
- leakage auditing;
- oracle trajectory studies;
- privileged-training upper bounds explicitly labeled as such.

---

## 5. Teacher-free sparse training flow

For each training episode:

### Step 1 — Select patient manifest

Choose one training patient and load only the fixed sparse manifest \(\Omega_i^{sparse}\).

### Step 2 — Split context and acquired sparse targets

Sample:

\[
\mathcal C_i\subset\Omega_i^{sparse},
\qquad
\mathcal Q_i\subset\Omega_i^{sparse}\setminus\mathcal C_i.
\]

Different roles may be sampled across epochs, but only within the fixed sparse set.

### Step 3 — Build teacher-free evidence maps

For each context slice:

```text
normalized intensity
+ fixed derivatives
+ gradient magnitude
+ Laplacian
+ multi-scale local contrast
+ modality condition
→ shared high-resolution micro-CNN
→ Z_str, Z_app, optional reliability C
```

Only context slices enter the encoder.

### Step 4 — Build patient state through the locked downstream path

```text
context evidence cache
→ provisional anchors
→ anchor-local evidence aggregation
→ shared tiny local fields
→ SDF/level-set structural Gaussians
  + volumetric appearance Gaussians
→ propagation and latent 3D Gaussian state
```

### Step 5 — Render sparse acquired targets

For each target plane \(q\in\mathcal Q_i\), render the required modality from the context-built representation.

### Step 6 — Reveal target pixels and compute losses

The total objective is organized as

\[
\begin{aligned}
\mathcal L
={}&
\lambda_{pred}\mathcal L_{pred}
+
\lambda_{eq}\mathcal L_{eq}
+
\lambda_{inv}\mathcal L_{inv}\\
&+
\lambda_{xmod}\mathcal L_{xmod}
+
\lambda_{anti}\mathcal L_{anti-collapse}\\
&+
\lambda_{local}\mathcal L_{local}
+
\lambda_{field}\mathcal L_{field}
+
\lambda_G\mathcal L_{Gaussian}
+
\lambda_{cal}\mathcal L_{calibration}.
\end{aligned}
\]

The predictive reconstruction loss is the final task objective. Structural auxiliary terms replace teacher supervision and stabilize the compact encoder.

### Step 7 — Backpropagate

Update global model parameters. Target pixels must not enter the patient state before their predictions are generated.

### Step 8 — Training schedule

```text
E0: teacher-free structural warm-up
E1: joint sparse context-to-target reconstruction
E2: end-to-end refinement with reduced auxiliary weights
```

Learned routing is not required during initial representation training. Fixed or analytic selection should first isolate encoder and representation quality.

---

## 6. Initial patient bootstrap flow

### Step 1 — Metadata-only initial observation selection

Select \(K_0\) spatially and cross-modally diverse candidate slices without loading their pixels.

### Step 2 — Commit and encode

For each selected slice:

```text
commit observation
→ load pixels
→ construct analytic channels
→ run teacher-free evidence encoder once
→ store Z_str, Z_app, and reliability with physical metadata
```

### Step 3 — Generate provisional anchors

Detect sparse structural candidate locations on observed planes, convert their pixel coordinates to physical 3D coordinates, merge redundant candidates, and create provisional anchors.

### Step 4 — Aggregate local evidence

Each anchor samples all relevant cached planes and obtains a compact evidence vector containing structural feature, appearance feature, modality, plane distance, orientation, and reliability information.

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
- legal self-consistency residual below threshold;
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

Export uncertainty derived from evidence distance, modality missingness, Gaussian disagreement, residual history, propagation depth, and optional learned calibration.

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

1. A compact structural encoder can be learned without teacher distillation by combining analytic differential cues, teacher-free structural constraints, and permanently sparse target-plane supervision.
2. Complementary sparse observations across patients can train a shared reconstruction representation without complete-volume targets in the main path.
3. A shared low-capacity anchor-local decoder is sufficient when physical geometry and local evidence are organized correctly before decoding.
4. SDF/level-set constrained structural Gaussians reduce geometric freedom and improve stability under sparse observations.
5. A separate volumetric appearance bank is necessary for faithful internal MRI reconstruction.
6. Closed-loop active querying improves quality–budget curves relative to fixed uniform and uncertainty-only sampling.
7. Adaptive anchors and multi-wave routing reduce the number of queried slices required to reach a target reconstruction quality.

These hypotheses remain to be validated. In particular, no claim is made that arbitrary hidden pathology is uniquely recoverable from an insufficient observation set.
