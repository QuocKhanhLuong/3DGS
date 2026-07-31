# T0.5 + T1 Implementation Plan — Legal Episodic Sparse Training and Teacher-Free Encoder

Date: 2026-07-29  
Target venue context: ISBI 2027 / medical imaging  
Status: historical implementation plan. T0.5, T1-A, and T1-B software have
since been implemented; T1-C remains unimplemented. Current executable status
is controlled by [`docs/codex/README.md`](../codex/README.md). Do not treat this
historical plan as Human Gate approval or rewrite its body as retrospective
history.

## 1. Research direction to preserve

The implementation must preserve the original research hypothesis:

```text
permanently sparse multi-sequence MRI slices
→ compact teacher-free structural evidence
→ physical support anchors
→ shared tiny anchor-local structural field
→ anchored seed Gaussians
→ anchor–Gaussian propagation
→ latent patient-specific 3D Gaussian representation
→ full-volume reconstruction
```

T0 is enabling infrastructure. It must not replace the representation thesis with an active-policy-only paper direction.

The next implementation tranche is intentionally split into:

```text
T0.5 — repair legal episodic-training and renderer semantics
T1   — implement teacher-free encoder and a fixed-topology Gaussian reconstruction baseline
```

T1 does not yet implement learned anchor propagation, adaptive topology, routing, or a true-SDF claim. Those remain T2+.

---

## 2. Non-negotiable scope rules

### Main method

- target venue framing is ISBI/medical imaging, not CVPR-first;
- training patients are permanently sparse;
- complete volumes are audit/evaluation data only;
- no teacher distillation in the main path;
- teacher/pretrained encoders are privileged upper-bound ablations only;
- no candidate pixels before legal commit and prediction registration;
- no learned router in T1;
- no adaptive Gaussian birth/split/merge/prune in T1;
- no use of `SDF` in executable APIs until sign, Eikonal, and distance calibration are tested;
- no claim that the current renderer is scanner-PSF-correct.

### Allowed T1 approximation

T1 may use deterministic fixed support points and a fixed-topology Gaussian constructor to test whether the teacher-free encoder improves sparse context-to-target rendering. This is a baseline bridge, not the final anchor-propagation method.

---

# Part A — T0.5 Contract Corrections

## 3. T0.5 objective

Repair the mismatch between the current T0 contracts and Phase-1 episodic training before implementing the encoder.

The current contract stores `CONTEXT` or `TARGET` as a permanent property of an observation. Phase 1 instead requires a fixed sparse availability set whose observations can receive different context/target roles across episodes.

The legal abstraction must become:

```text
SparseAvailabilityManifest
└── all permanently acquired sparse observations for the patient

EpisodeAssignment
├── context_ids
└── target_ids

EpisodeLedger
├── open context payloads
├── expose target metadata
├── commit target
├── register prediction receipt
└── reveal target once
```

---

## 4. T0.5 implementation tasks

### 4.1 Separate availability from episode role

Refactor or extend the observation contracts so that:

- the manifest states which sparse observations legally exist;
- context/target roles are assigned per episode;
- context and target IDs are disjoint;
- every assigned ID belongs to the same fixed sparse manifest;
- episode assignment is immutable and canonically hashable;
- roles may change across epochs without changing the underlying sparse manifest;
- patient-level split legality remains manifest-level.

Recommended new records:

```text
EpisodeAssignment
├── episode_id
├── manifest_hash
├── patient_id
├── context_ids
├── target_ids
└── assignment_hash
```

The existing permanent `AccessLevel` field may remain only for backward compatibility during migration; new Phase-1 code must not depend on it.

### 4.2 Enforce prediction-before-reveal

A target reveal must require proof that a prediction was produced from the pre-reveal patient state.

Recommended flow:

```text
commit_target(target_id, state_version)
→ render prediction
→ register_prediction_receipt(
      target_id,
      state_version,
      plane_hash,
      renderer_version,
      prediction_digest
  )
→ reveal_target(receipt_capability)
```

The receipt is a scientific-validity contract, not a security boundary. It must make render-before-reveal testable and auditable.

### 4.3 Separate training role from acquisition cost

Context/target role is an optimization role. Acquisition cost is a deployment quantity.

The new design must support:

- zero-cost role assignment during offline context/target training;
- explicit observation cost for active deployment;
- initial bootstrap observations counting toward deployment budget;
- exact decimal accounting preserved.

### 4.4 Fix support-amplitude gauge before routing use

Normalized additive intensity is invariant to a common multiplicative scale on all Gaussian support amplitudes, while `support_mass` is not.

Therefore T1 must not call raw `support_mass` calibrated uncertainty or stable observability.

Implement one of the following named policies for baseline experiments:

1. **fixed gauge:** subtract the mean log amplitude per patient state;
2. **bounded normalized amplitude:** normalize amplitudes to a fixed total mass;
3. **geometry-only coverage:** compute coverage independently from trainable common amplitude scale.

The chosen policy must have a regression test showing that equivalent intensity parameterizations do not arbitrarily change supported/unsupported classification.

### 4.5 Correct renderer naming

Use the following wording in code and new docs:

> through-plane profile-aware Gaussian reference renderer

Do not use `PSF-correct` until scanner-specific or protocol-specific PSF semantics and in-plane footprint treatment exist.

### 4.6 Exact-main verification

Add a GitHub Actions CPU workflow for:

```text
python -m pytest -q
python -m compileall -q src tests
git diff --check
```

The reproducibility record must identify the exact clean commit being tested.

---

## 5. T0.5 files

Recommended changes:

```text
src/smagm/contracts/observation.py
src/smagm/contracts/episode.py              # new
src/smagm/data/manifest.py
src/smagm/renderer.py                       # gauge/coverage policy if needed
src/smagm/__init__.py

 tests/contracts/test_episode_assignment.py  # new
 tests/integration/test_prediction_receipt_barrier.py  # new
 tests/render/test_support_gauge.py          # new

.github/workflows/ci.yml                     # new
CHANGELOG.md
```

Compatibility wrappers may be retained temporarily, but no duplicate scientific semantics should remain undocumented.

---

## 6. T0.5 blocking tests

T0.5 passes only when:

1. one fixed sparse manifest supports multiple legal episode assignments;
2. context and target sets cannot overlap;
3. non-manifest observations cannot be assigned;
4. target payload cannot be revealed after commit alone;
5. target payload can be revealed only after a receipt bound to the committed state version and target plane;
6. a receipt from another target, state, episode, or ledger is rejected;
7. training role assignment does not consume deployment acquisition budget;
8. deployment bootstrap and target commitments both consume declared acquisition cost;
9. global amplitude-gauge changes do not arbitrarily change baseline supported-region semantics;
10. exact current-main CI passes.

---

# Part B — T1 Teacher-Free Structural Encoder

## 7. T1 objective

Test the Phase-1 hypothesis under a verified sparse legal and physical renderer:

> Can a compact teacher-free encoder preserve structural evidence that improves sparse context-to-target 3D Gaussian reconstruction over analytic-only and raw shallow-CNN baselines under matched observations, fixed topology, renderer, parameter budget, and optimization opportunity?

T1 is an attribution experiment. It is not yet the final support-anchor propagation method.

---

## 8. T1 system flow

```text
permanently sparse availability manifest
→ episode assignment: context C_i / target Q_i
→ open context only
→ analytic differential channel bank
→ encoder variant E0 / E1 / E2
→ compact Z_str, Z_app, optional reliability
→ deterministic provisional support points on context planes
→ fixed-topology Gaussian constructor
→ through-plane profile-aware renderer
→ register target prediction receipt
→ reveal sparse target
→ reconstruction + teacher-free structural losses
```

### Encoder variants

```text
E0 — analytic channels only, no learned encoder
E1 — raw-image shallow CNN
E2 — analytic scaffold + shared high-resolution micro-CNN, main T1 candidate
E3 — frozen pretrained encoder, compute upper bound
E4 — teacher-distilled student, privileged-training upper bound only
```

E3 and E4 must not affect the main-method claim.

---

## 9. T1 modules

### 9.1 Analytic differential bank

Create fixed differentiable channels such as:

```text
normalized intensity
d_x, d_y
gradient magnitude
Laplacian
local contrast at two scales
valid-content mask
optional artifact/reliability cue
```

Requirements:

- implemented as fixed tensor operations or fixed convolutions;
- preserves gradient path where required;
- explicit padding policy;
- explicit pixel-grid semantics;
- included in FLOP/runtime accounting;
- numerically stable for constant and low-signal slices.

Recommended file:

```text
src/smagm/features/analytic.py
```

### 9.2 Teacher-free micro-CNN

Initial reference architecture:

```text
analytic/raw channels + modality condition
→ 3×3 high-resolution stem
→ depthwise residual block ×2
→ optional stride-2 transition
→ depthwise residual block ×2
→ output projections
    ├── Z_str: approximately 16 channels
    ├── Z_app: approximately 8 channels
    └── reliability: one channel, optional
```

Requirements:

- shared trunk across modalities;
- tiny modality conditioning through FiLM, normalization, or embedding;
- output stride limited to 1, 2, or 4;
- feature-grid-to-plane transform returned explicitly;
- no large U-Net or transformer;
- no direct final Gaussian topology prediction;
- training cache lifetime is one forward/episode;
- inference cache is detached and version-hashed.

Recommended files:

```text
src/smagm/features/encoder.py
src/smagm/features/contracts.py
src/smagm/features/cache.py
```

### 9.3 Teacher-free losses

Implement separately testable losses:

```text
spatial equivariance
intensity invariance
registered cross-modality structural consistency
variance floor
channel covariance penalty
local differential preservation
sparse target-plane reconstruction
```

Rules:

- align `Z_str`, not `Z_app`;
- registration-confidence weighting is explicit;
- anti-collapse statistics are logged per channel;
- reconstruction is the final task objective;
- structural losses must be switchable for ablation.

Recommended file:

```text
src/smagm/losses/structural.py
```

### 9.4 Deterministic support-point baseline

T1 needs a downstream bridge that is fixed across E0/E1/E2.

Use deterministic support points from context planes, for example:

- fixed physical grid or Poisson-disc selection on observed planes;
- no learned point birth;
- no propagation;
- no SDF claim;
- same support count across encoder variants.

Each support point samples encoder evidence at an aligned feature coordinate.

Recommended file:

```text
src/smagm/baselines/fixed_support.py
```

### 9.5 Safe Gaussian constructor

Create runtime `GaussianBatch` from unconstrained trainable or predicted values:

```text
raw center offsets
raw lower-triangular covariance parameters
raw log amplitudes
raw modality appearance
→ bounded, valid GaussianBatch
```

Use positive diagonal parameterization such as:

```text
diagonal = softplus(raw_diagonal) + epsilon
```

Apply the selected amplitude-gauge policy.

No anchor propagation, split, merge, or prune in T1.

Recommended file:

```text
src/smagm/baselines/fixed_gaussian.py
```

### 9.6 Episodic trainer

Responsibilities:

- receive one `EpisodeAssignment`;
- open context only;
- build analytic features and encoder maps;
- construct fixed Gaussian state;
- commit target metadata;
- render and register prediction receipt;
- reveal target only after receipt;
- compute masked loss excluding unsupported pixels only as explicitly reported, never silently;
- log unsupported fraction as a failure diagnostic;
- keep all training tensors on the differentiable path.

Recommended files:

```text
src/smagm/training/episode.py
src/smagm/training/objective.py
scripts/train_t1.py
```

---

## 10. T1 tensor contracts

### Encoder input

```text
image:             [B,1,H,W]
valid_mask:        [B,1,H,W] bool
modality_index:    [B]
analytic_channels: [B,C_phi,H,W]
plane metadata:    immutable per observation
```

### Encoder output

```text
Z_str:             [B,C_str,H_f,W_f]
Z_app:             [B,C_app,H_f,W_f]
reliability:       [B,1,H_f,W_f] optional
feature_to_plane:  explicit transform
encoder_version:   hash
```

### Fixed support state

```text
support_positions_ras_mm: [N,3]
support_plane_ids:        [N]
sampled_Z_str:            [N,C_str]
sampled_Z_app:            [N,C_app]
support_reliability:      [N,1]
```

### Gaussian state

Use the existing `GaussianBatch`, created through a safe constructor rather than direct optimizer mutation of a validated Cholesky factor.

---

## 11. Training schedule

### T1-A — synthetic contract verification

- analytic channels;
- feature alignment;
- equivariance;
- fixed-support projection;
- differentiable context-to-target render;
- receipt barrier;
- no real medical data required.

### T1-B — teacher-free structural warm-up

Optimize:

```text
L_eq + L_inv + L_xmod + L_var + L_cov + L_local
```

using only permanently available sparse slices.

This stage is diagnostic and does not pass T1 by itself.

### T1-C — joint sparse reconstruction

Optimize:

```text
L_pred
+ lambda_struct * structural losses
+ Gaussian regularization
```

Hold support topology and renderer fixed across E0/E1/E2.

### T1-D — attribution ablation

Compare E0/E1/E2 under:

- identical episode assignments;
- identical support positions and primitive count;
- identical renderer and profile;
- identical training steps;
- matched parameter/FLOP reporting;
- at least three learned-model seeds once the pipeline is stable.

---

## 12. T1 tests

### Analytic-feature tests

- constant input produces finite expected derivatives;
- impulse and ramp inputs match independent references;
- padding and feature alignment are explicit;
- augmentation transform produces matching transformed features;
- float32 and float64 behavior is bounded.

### Encoder tests

- output shapes and feature-to-plane transform are correct;
- stride 1/2/4 sampling maps to the correct physical location;
- no teacher module is instantiated in the main path;
- modality conditioning changes allowed appearance behavior without changing tensor contracts;
- structural branch does not collapse on synthetic batches.

### Loss tests

- matched registered features reduce cross-modality loss;
- mismatched features score worse;
- appearance features are not accidentally aligned;
- variance/covariance losses have finite gradients;
- intensity-only augmentation leaves geometry target unchanged.

### Fixed-support/Gaussian tests

- support count is identical across E0/E1/E2;
- support coordinates match context planes;
- trainable covariance parameterization always produces valid SPD covariance;
- amplitude gauge is enforced;
- renderer gradients reach encoder parameters through support sampling and Gaussian construction.

### Leakage tests

- target pixels cannot enter analytic channels, encoder, support construction, or Gaussian state;
- commit without prediction receipt cannot reveal;
- target can be revealed only after pre-reveal state prediction is registered;
- audit volume cannot be opened by training process.

---

## 13. T1 experiment matrix

Minimum matrix:

| ID | Encoder | Structural auxiliaries | Downstream state |
|---|---|---|---|
| E0 | analytic only | none | fixed supports + fixed-topology Gaussian |
| E1 | raw shallow CNN | reconstruction only | same |
| E2a | analytic + micro-CNN | reconstruction only | same |
| E2b | analytic + micro-CNN | full teacher-free losses | same |
| E3 | frozen pretrained | reconstruction | same, upper bound |
| E4 | distilled student | reconstruction | same, privileged upper bound |

Report:

- sparse target MAE/NMSE/PSNR/SSIM;
- gradient and high-frequency error;
- unsupported fraction;
- parameters and FLOPs;
- encoding latency and cache bytes;
- total training/runtime cost;
- per-patient paired differences;
- lesion/ROI-sensitive audit metrics when labels exist.

---

## 14. T1 gates and stop rules

### Gate T0.5-L — Legal episodic training

Pass only when episode roles are separate from permanent availability and prediction-before-reveal is enforced.

### Gate T1-F — Feature validity

Pass only when:

- alignment tests pass;
- structural features do not collapse;
- matched registered points are more similar than mismatched points;
- local differential cues remain recoverable.

### Gate T1-R — Reconstruction attribution

Pass only when E2b improves over E0 and E1 under matched downstream state and compute accounting.

### Gate T1-M — Medical fidelity

Pass only when global metric improvement does not cause a meaningful lesion/ROI fidelity regression on the audit set.

### Stop rules

- If E0 matches E2b, remove the learned encoder from the main novelty path.
- If E1 matches E2b, remove analytic scaffold claims.
- If structural auxiliaries improve proxy features but not target reconstruction, demote them to diagnostics.
- If fixed Gaussian reconstruction cannot beat interpolation floors, do not implement anchor propagation yet; repair the state/renderer/data assumptions first.
- Do not proceed to active routing before a static representation baseline is competitive.

---

## 15. Agent-team ownership

### Researcher

- reframe all novelty analysis for ISBI/MICCAI medical imaging;
- analyze collision of the complete mechanism, not only individual blocks;
- maintain a table for:
  `sparse support seeds → local structural field → anchored Gaussian propagation → permanently sparse training`;
- do not demote the representation thesis without a direct full-mechanism collision or negative matched ablation.

### PM

- maintain one primary representation thesis for T1–T3;
- treat active routing as a later extension;
- enforce T0.5 and T1 stop rules;
- prevent scope expansion into CVPR-style policy engineering.

### Architect

- own EpisodeAssignment, prediction receipt, feature-grid geometry, safe Gaussian constructor, and training/inference cache semantics;
- keep APIs named `StructuralField` until SDF tests exist.

### Medical Data Steward

- write the permanently sparse manifest-construction protocol;
- define patient-level splits, registration assumptions, missing-modality policy, and physically isolated audit process;
- veto any loader that can access non-manifest or audit pixels.

### Dev

- implement T0.5 before T1;
- keep each scientific component in a separate module;
- preserve differentiability;
- update `CHANGELOG.md` with every module tranche;
- do not add routing, topology, or large decoders.

### QA

- make role separation, receipt barrier, affine alignment, analytic features, anti-collapse, amplitude gauge, autograd, and leakage-positive tests blocking.

### Experiment Lead

- own immutable E0/E1/E2 episode assignments and matched downstream configurations;
- report quality per FLOP and per observed slice;
- separate main sparse training from privileged upper bounds.

### Reproducibility Auditor

- require exact clean commit CI;
- hash manifests, episode assignments, configs, opened-file ledgers, checkpoints, and outputs;
- record runtime, cache bytes, primitive count, hardware, and software lock.

### Reviewer

- review T0.5 first, then T1;
- block any claim based only on auxiliary feature losses;
- block comparisons with different observations, support counts, renderers, or optimizer budgets;
- block use of `SDF`, `PSF-correct`, `active MRI acquisition`, or calibrated uncertainty without corresponding evidence.

---

## 16. Expected file tree after T1

```text
src/smagm/
├── contracts/
│   ├── coordinates.py
│   ├── observation.py
│   └── episode.py
├── features/
│   ├── analytic.py
│   ├── contracts.py
│   ├── encoder.py
│   └── cache.py
├── losses/
│   └── structural.py
├── baselines/
│   ├── fixed_support.py
│   └── fixed_gaussian.py
├── training/
│   ├── episode.py
│   └── objective.py
├── gaussians.py
└── renderer.py

scripts/
└── train_t1.py

tests/
├── contracts/
├── features/
├── losses/
├── baselines/
├── integration/
└── render/

configs/
└── t1/
    ├── e0_analytic.yaml
    ├── e1_raw_cnn.yaml
    ├── e2_teacher_free.yaml
    └── common.yaml
```

---

## 17. Exit decision

T1 exits with one of three outcomes:

### PASS

E2 teacher-free encoder improves matched sparse target reconstruction and preserves medical structure. Proceed to T2:

```text
learned structural support candidates
→ physical anchor bootstrap
→ shared tiny anchor-local field
→ anchored seed Gaussians
```

### PARTIAL

A simpler E0 or E1 encoder is sufficient. Proceed to T2 with the simpler encoder and remove unsupported encoder novelty claims.

### FAIL

The fixed-topology Gaussian baseline is not competitive or the sparse legal regime cannot support stable context-to-target reconstruction. Stop architecture expansion and revise the data, observation, or representation assumptions before implementing anchor propagation.
