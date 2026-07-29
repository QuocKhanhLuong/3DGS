# Module Architecture — Sparse Evidence Encoder

## 1. Purpose

Convert each legally queried 2D MRI slice into a compact dense feature map that remains aligned with the physical slice plane.

This module is intentionally defined by its **contract**, not by a permanently fixed backbone. The research question is whether useful reconstruction evidence can be extracted with substantially lower computation than a large U-Net or pretrained dense backbone.

---

## 2. Hard constraints

The encoder must:

- process only committed observed slices;
- run once per newly queried slice;
- produce cacheable spatial features;
- preserve boundary and local-intensity information;
- support all MRI modalities through a shared or mostly shared architecture;
- expose exact output stride and physical alignment;
- have measurable FLOPs, latency, and memory;
- train directly through sparse reconstruction objectives;
- not require a teacher in the core method.

---

## 3. Interface

### Input

```text
EvidenceEncoderInput
├── image: [1, H, W]
├── modality_id
├── plane origin and orientation
├── in-plane spacing
├── slice thickness
├── valid-content mask
└── normalization metadata
```

### Output

```text
EvidenceEncoderOutput
├── feature_map Z: [C, H/s, W/s]
├── reliability_map C: [1, H/s, W/s] optional
├── feature stride s
├── encoder version
└── normalization record
```

The feature map must support differentiable sampling at projected physical coordinates.

---

## 4. Role boundary

The encoder should answer:

> What compact local evidence does this observed slice provide?

It should not be responsible for:

- constructing the complete 3D field;
- deciding the next trajectory action;
- maintaining patient memory;
- learning a full global 3D anatomy in one forward pass;
- producing final reconstructed volumes directly.

These separations keep the encoder replaceable and make FLOP–accuracy trade-offs scientifically measurable.

---

## 5. Candidate architecture families

### Family E0 — Fixed differential evidence

```text
normalized intensity
+ gradient x/y
+ gradient magnitude
+ Laplacian
+ multi-scale local contrast
```

Advantages:

- almost no trainable computation;
- transparent;
- useful lower-bound baseline.

Limitations:

- confuses noise and true structure;
- limited anatomical context;
- weak for low-contrast structures.

### Family E1 — Shallow high-resolution CNN

Example pattern:

```text
input channels
→ 3×3 convolution
→ depthwise residual blocks
→ limited downsampling, at most ×4
→ compact projection to C channels
```

Desired properties:

- high-resolution preservation;
- shared weights across modalities;
- small modality embedding or affine conditioning;
- tens or hundreds of thousands of parameters rather than a large segmentation network.

### Family E2 — Hybrid analytic + micro-CNN

```text
raw normalized intensity
+ fixed differential channels
→ small shared CNN
→ compact feature map
```

This family lets the CNN focus on suppressing noise and selecting useful structure instead of relearning basic image derivatives.

### Family E3 — Frozen pretrained dense encoder

A frozen pretrained model may be retained as an upper-bound or compute–accuracy comparison. It is not assumed to be the final architecture because:

- output stride may be too coarse;
- dense FLOPs may dominate initialization;
- feature semantics may not align with reconstruction geometry;
- it weakens the claim that the sparse representation, rather than the backbone, drives performance.

---

## 6. Modality handling

The preferred design is one shared encoder with low-cost modality conditioning:

\[
Z=E_\theta(I,\mathbf e_m).
\]

Possible conditioning methods:

- modality embedding concatenated to channels;
- small FiLM scale and shift;
- modality-specific input normalization;
- a tiny modality-specific first convolution followed by a shared trunk.

Avoid independent large encoders for each sequence unless an ablation proves a compelling gain.

---

## 7. Spatial resolution

Reconstruction requires local detail. Excessive downsampling can irreversibly remove thin structures.

Recommended evaluation points:

```text
stride 1
stride 2
stride 4
stride 8 upper-bound comparison
```

A small decoder cannot restore exact spatial information that the encoder has discarded. Therefore output stride is a primary experimental variable, not a minor implementation detail.

---

## 8. Reliability output

A reliability map may represent local evidence quality, but it must be calibrated against reconstruction error.

Possible non-learned components:

- valid-content mask;
- local signal-to-noise estimate;
- gradient consistency;
- artifact score;
- distance from image support.

Possible learned output:

\[
C(u,v)\in(0,1).
\]

The reliability head should remain small. It is optional because observability uncertainty can also be constructed analytically from coverage, distance, modality disagreement, and residual history.

---

## 9. Training

The encoder receives gradients from hidden reconstruction targets through:

```text
hidden target loss
→ renderer
→ Gaussian/local field representation
→ anchor evidence sampling
→ cached feature map
→ encoder
```

Additional direct auxiliary losses are allowed only when they improve the final reconstruction objective and do not require unavailable inference data.

Potential auxiliary constraints:

- cross-modality feature alignment at registered coordinates;
- local gradient preservation;
- feature stability under intensity augmentation;
- feature compactness;
- reconstruction-error-aware reliability calibration.

---

## 10. Cache behavior

For every queried slice, store the encoder output once:

```text
cache[(patient, modality, slice)]
├── compact feature map
├── reliability map
├── physical plane metadata
├── content digest
└── encoder-version hash
```

Anchor birth, move, split, and merge must sample from this cache rather than re-run the encoder.

If encoder weights change during offline training, the current episode cache is rebuilt as part of the next forward pass. At patient inference, encoder weights are frozen, so the cache remains valid.

---

## 11. Compute accounting

Report:

- parameters;
- multiply–accumulate operations per slice;
- latency per slice;
- peak memory;
- cache size per slice;
- total encoder cost at observation budget \(B\);
- fraction of total runtime spent in encoding.

The correct compute unit is **per queried slice**, because no unqueried slice is encoded.

---

## 12. Required ablations

```text
E0 fixed features
E1 shallow CNN
E2 fixed + micro-CNN
E3 frozen pretrained upper bound
```

For each, hold constant:

- observed slices;
- local tiny MLP;
- anchor logic;
- Gaussian renderer;
- optimization budget;
- trajectory.

Evaluate reconstruction gain per additional encoder FLOP, not accuracy alone.

---

## 13. Current research position

The evidence encoder is not yet locked. The only locked architectural requirement is that it produces compact, cacheable, spatially aligned evidence and does not force the shared local tiny MLP to perform global image understanding.
