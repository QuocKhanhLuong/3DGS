# Module Architecture — Teacher-Free Structural Evidence Encoder

## 1. Purpose

Convert each legally observed 2D MRI slice into compact, cacheable, spatially aligned evidence for the locked downstream path:

```text
observed slice
→ evidence encoder
→ provisional anchors
→ shared tiny local field
→ anchored Gaussians
→ propagation
→ latent 3D Gaussian representation
```

The encoder must provide enough local structure for physical anchor generation and local SDF/level-set decoding without relying on teacher distillation in the main method.

---

## 2. Locked design decision

The recommended core encoder is:

\[
\boxed{
\text{analytic differential scaffold}
+
\text{shared high-resolution micro-CNN}
}
\]

Teacher distillation is not part of the main architecture. Frozen pretrained or teacher-distilled encoders remain upper-bound ablations only.

The exact channel count, output stride, and block count remain implementation variables.

---

## 3. Hard constraints

The encoder must:

- process only legally available sparse slices;
- run once per committed slice;
- produce cacheable spatial maps;
- preserve exact plane alignment;
- separate structural and modality-specific evidence;
- preserve boundary and local-intensity information;
- support shared weights across MRI modalities;
- expose FLOPs, latency, memory, cache size, and output stride;
- train without requiring complete-volume targets or a teacher;
- remain smaller than the anchor, Gaussian, and trajectory contribution being claimed.

It must not:

- construct the full 3D field itself;
- decide the trajectory;
- create the final Gaussian topology directly;
- perform global patient reasoning in a large decoder;
- access slices outside the legal sparse manifest.

---

## 4. Interface

### Input

```text
EvidenceEncoderInput
├── image: [1, H, W]
├── modality_id
├── plane origin and orientation
├── in-plane spacing
├── slice thickness
├── valid-content mask
├── normalization metadata
└── optional artifact metadata
```

### Output

```text
EvidenceEncoderOutput
├── structural_map Z_str: [C_str, H/s, W/s]
├── appearance_map Z_app: [C_app, H/s, W/s]
├── reliability_map C: [1, H/s, W/s] optional
├── feature stride s
├── encoder version
└── normalization record
```

All spatial outputs must support differentiable sampling at projected physical coordinates.

---

## 5. Analytic differential scaffold

For normalized slice \(I\), construct a configurable fixed channel bank:

\[
\Phi(I)=
[
I,
\partial_x I,
\partial_y I,
\|\nabla I\|,
\Delta I,
C_{r_1}(I),
C_{r_2}(I),
M_{valid}
].
\]

Recommended components:

- normalized intensity;
- horizontal and vertical derivatives;
- gradient magnitude;
- Laplacian;
- local contrast at multiple radii;
- valid-content mask;
- optional local signal-to-noise or artifact channel.

The scaffold provides explicit low-cost cues related to local orientation, transitions, and distance-like structure. The micro-CNN learns which cues correspond to anatomy rather than noise or scanner artifacts.

Analytic preprocessing cost must be included in compute accounting.

---

## 6. Recommended micro-CNN

Initial reference pattern:

```text
analytic/raw channels
+ modality condition
        ↓
3×3 convolution, approximately 24 channels
        ↓
depthwise residual block ×2
        ↓
optional stride-2 transition
        ↓
depthwise residual block ×2
        ↓
1×1 projections
        ├── Z_str, approximately 16 channels
        ├── Z_app, approximately 8 channels
        └── C, one channel optional
```

Preferred properties:

- high-resolution stem;
- output stride in \(\{1,2,4\}\) for the main study;
- shared trunk across modalities;
- modality-specific input normalization, FiLM, or tiny first-layer adapter;
- tens or hundreds of thousands of parameters rather than a large U-Net;
- no large decoder.

The channel counts above are starting points, not locked claims.

---

## 7. Structural versus appearance outputs

### Structural map

\[
Z^{str}=E^{str}_\theta(\Phi(I),e_m).
\]

Used by:

- structural candidate scoring;
- cross-plane anchor evidence aggregation;
- shared tiny local field;
- geometry consistency and uncertainty.

Desired properties:

- spatial equivariance;
- robustness to intensity scaling and bias field;
- partial invariance across registered modalities;
- preservation of thin and high-frequency structures.

### Appearance map

\[
Z^{app}=E^{app}_\theta(\Phi(I),e_m).
\]

Used by:

- modality-specific Gaussian appearance slots;
- local intensity reconstruction;
- modality disagreement diagnostics.

It is not forced to be identical across modalities.

### Reliability map

\[
C(u,v)\in(0,1).
\]

May combine learned and analytic evidence quality. It must be calibrated against reconstruction error and should not become a generic confidence shortcut.

---

## 8. Teacher-free training constraints

The encoder is optimized through sparse target-plane reconstruction plus direct structural constraints.

### Spatial equivariance

\[
\mathcal L_{eq}
=
\|E^{str}(TI)-T(E^{str}(I))\|_1.
\]

### Intensity invariance

\[
\mathcal L_{inv}
=
\|Z^{str}(g(I))-Z^{str}(I)\|_1,
\]

where \(g\) changes intensity but not geometry.

### Cross-modality structural consistency

At registered physical coordinates:

\[
\mathcal L_{xmod}
=
\sum_x w(x)
\|z_m^{str}(x)-z_n^{str}(x)\|_1.
\]

### Anti-collapse

\[
\mathcal L_{var}
=
\sum_c\max(0,\gamma-\operatorname{Std}(Z_c^{str})),
\]

\[
\mathcal L_{cov}
=
\sum_{p\neq q}\operatorname{Cov}(Z_p^{str},Z_q^{str})^2.
\]

### Local differential preservation

A training-only head predicts compact differential quantities from \(Z^{str}\):

\[
\mathcal L_{local}
=
\|\widehat{\nabla I}-\nabla I\|_1
+
\lambda_\Delta
\|\widehat{\Delta I}-\Delta I\|_1.
\]

### Sparse reconstruction gradient

```text
sparse target loss
→ plane renderer
→ Gaussian state
→ anchors and tiny local field
→ cached encoder maps
→ encoder
```

The reconstruction objective remains the final task criterion. Auxiliary constraints stabilize and shape the encoder representation.

---

## 9. Training schedule

### E0 — Structural warm-up

Optimize equivariance, invariance, cross-modality agreement, anti-collapse, and local differential preservation using only legally available sparse slices.

### E1 — Joint reconstruction

Connect the full downstream path and optimize sparse target-plane reconstruction with structural auxiliary losses.

### E2 — End-to-end refinement

Reduce auxiliary weights and let reconstruction dominate.

The encoder cache is rebuilt between offline optimization forwards when weights change. At patient inference, weights are frozen and each committed slice is encoded once.

---

## 10. Cache contract

```text
cache[(patient, modality, slice)]
├── Z_str
├── Z_app
├── reliability map
├── physical plane metadata
├── output stride
├── content digest
└── encoder-version hash
```

Anchor birth, move, split, merge, and propagation must sample this cache rather than re-run the encoder.

---

## 11. Candidate families and ablations

```text
E0 — analytic differential evidence only
E1 — raw-image shallow CNN
E2 — analytic scaffold + teacher-free micro-CNN, main method
E3 — frozen pretrained dense encoder, compute upper bound
E4 — dense teacher-distilled student, privileged-training upper bound
```

For all variants, hold constant:

- sparse manifests and context/target splits;
- tiny local MLP;
- anchor logic;
- Gaussian memory and propagation;
- renderer;
- observation budget;
- downstream optimization schedule.

Report quality versus:

- parameters;
- FLOPs per slice;
- latency per slice;
- cache size;
- total encoder cost at budget \(B\);
- target-plane reconstruction quality;
- final audit-volume reconstruction quality.

---

## 12. Theoretical role

The encoder does not claim to recover a unique anatomical representation. Its purpose is to identify a compact shared structural subspace that is:

- aligned with physical coordinates;
- stable under intensity nuisance;
- consistent across registered modalities;
- non-collapsed;
- useful for downstream sparse 3D reconstruction.

In a simplified linear multi-modality model, cross-modality covariance emphasizes shared structure when modality-private components are not consistently correlated. Equivariance and reconstruction losses then preserve the geometric information required by anchor-local decoding.

---

## 13. Code-entry checks

Before integrating the complete downstream system, verify:

1. spatial alignment after every allowed augmentation;
2. no structural feature collapse;
3. registered cross-modality points are more similar than mismatched points;
4. compact features preserve local differential information;
5. output stride does not erase thin structures;
6. E2 improves sparse target prediction over E0 and E1;
7. the sparse loader never opens non-manifest slices;
8. measured encoder cost includes analytic preprocessing.
