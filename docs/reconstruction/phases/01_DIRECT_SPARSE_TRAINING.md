# Phase 1 — Direct Sparse Episodic Training

## 1. Objective

Train the reconstruction system directly from sparse observed slices without a teacher network and without presenting the complete patient volume as model input.

The full registered volume may exist on disk, but within one episode it is divided into legally separated roles:

```text
Observed subset O
= may enter encoder, anchors, Gaussian memory, and trajectory state

Hidden target subset H
= may be read only after state construction to compute loss

Unused subset
= is not loaded during the iteration
```

---

## 2. Episode formulation

For one patient with candidate pool \(\Omega\), sample:

\[
\mathcal O\subset\Omega,
\qquad
\mathcal H\subset\Omega\setminus\mathcal O.
\]

Typical training does not need to load all of \(\Omega\). Per-episode I/O is approximately

\[
|\mathcal O|+|\mathcal H|\ll|\Omega|.
\]

Different subsets may be sampled across epochs. This is not leakage because only \(\mathcal O\) may create the representation in any given episode.

---

## 3. Structured data flow

```text
Patient index
    │
    ▼
Budget sampler
    │
    ├── observed set O
    └── hidden target set H
            │
Observed O  │
    │       │
    ▼       │
Evidence encoder
    │       │
    ▼       │
Compact evidence cache
    │       │
    ▼       │
Anchor bootstrap / update
    │       │
    ▼       │
Shared tiny local field
    │       │
    ▼       │
SDF–Gaussian memory
    │       │
    ▼       │
Plane/point renderer ◄──── hidden target coordinates
    │
    ▼
Predicted hidden intensities
    │
    ▼
Load hidden target values
    │
    ▼
Losses and backpropagation
```

The hidden target content is loaded after the representation path has finished. Its coordinates may be known beforehand, but its pixel values must not affect state construction.

---

## 4. Dataset sampling unit

The preferred training unit is a **patient episode**, not an independent image batch.

```text
EpisodeBatch
├── patient_id
├── observed_slice_descriptors
├── observed_slice_pixels
├── hidden_target_descriptors
├── hidden_target_pixels or sampled 3D values
├── physical registration metadata
├── observation budget
└── deterministic episode seed
```

Multiple patients may be batched if memory allows, but each maintains a separate patient state.

---

## 5. Observation curricula

A direct sparse system may be difficult to optimize if it begins with the smallest budget. A budget curriculum is allowed without changing the final task.

Example:

```text
Stage A: 24–32 observed slices
Stage B: 12–20 observed slices
Stage C: 6–12 observed slices
Stage D: target deployment budget and active route
```

The curriculum changes the amount of legal input, not the model architecture.

Additional episode perturbations:

- random modality dropout;
- irregular slice spacing;
- missing central or peripheral regions;
- registration noise within realistic limits;
- variable slice thickness;
- hard target sampling near high-frequency structures;
- patient-level rather than slice-level data splitting.

---

## 6. Forward pass

### 6.1 Encode observations

Each observed slice is encoded once:

\[
Z_k=E_\theta(I_k,m_k,P_k).
\]

The exact encoder family remains replaceable. The required contract is a compact dense feature map aligned with the input physical plane.

### 6.2 Build representation

Observed evidence produces:

\[
\mathcal S(\mathcal O)=
(F,\mathcal A,\mathcal G^{surface},\mathcal G^{volume},\mathcal U).
\]

### 6.3 Render hidden targets

For hidden target plane \(P_h\) and modality \(m_h\):

\[
\hat I_h=R(\mathcal S(\mathcal O),P_h,m_h).
\]

The same representation may be queried at arbitrary 3D points instead of complete target slices.

---

## 7. Loss architecture

### 7.1 Pixel fidelity

Use robust L1 or Charbonnier rather than relying only on MSE:

\[
\mathcal L_{pix}=\rho(\hat I-I).
\]

### 7.2 Structural fidelity

Possible terms:

- SSIM or multi-scale SSIM;
- gradient magnitude error;
- Laplacian or edge error;
- local normalized cross-correlation;
- frequency-band consistency.

### 7.3 Cross-modality geometry consistency

Registered modalities should agree on common structural locations even when intensities differ. This loss should align geometry, not force identical appearance codes.

### 7.4 Field regularization

Depending on the field choice:

- Eikonal regularization;
- local gradient smoothness;
- continuity between overlapping local fields;
- surface support compactness;
- curvature-aware anchor density.

### 7.5 Gaussian regularization

- excessive-overlap penalty;
- unsupported primitive penalty;
- structural Gaussian normal-thickness penalty;
- volumetric Gaussian scale bounds;
- center displacement and manifold penalties;
- primitive-count or complexity regularization.

### 7.6 Uncertainty calibration

Predicted uncertainty should track held-out reconstruction error, not simply become large everywhere.

---

## 8. Backpropagation boundary

Trainable in this phase:

- evidence encoder;
- shared tiny local MLP;
- optional aggregation gate;
- Gaussian parameter initialization/update predictors;
- optional uncertainty calibration;
- optional trajectory utility in later unrolled training.

Patient-specific anchors, Gaussian attributes, and caches are differentiable episode state but are not persistent global parameters shared across patients.

---

## 9. Static-routing and active-routing training

Training can be organized in two regimes.

### Regime A — Representation training

Observed sets are sampled by fixed strategies such as uniform, random, or metadata-balanced selection. This isolates reconstruction representation quality.

### Regime B — Active unrolled training

The trajectory chooses sequential observations. The episode is unrolled through multiple rounds and receives quality–budget objectives.

The core reconstruction representation should first be scientifically evaluated independently of routing so that trajectory gains are not confused with encoder or renderer gains.

---

## 10. Required logs

Per episode record:

- number of observed slices;
- number of hidden target slices/points;
- modality distribution;
- encoder FLOPs and latency;
- number of anchors and Gaussians;
- local-field evaluations;
- reconstruction losses by modality;
- uncertainty calibration;
- observation leakage audit;
- random seed and patient split.

---

## 11. Failure conditions

This phase is not valid if:

- hidden target pixels are passed into the encoder;
- complete volume features are precomputed and exposed to the patient state;
- target slices affect anchor generation before rendering;
- patient train/test split is performed at slice level;
- the model silently loads unqueried data for normalization or routing;
- reported compute excludes hidden dense operations in preprocessing.
