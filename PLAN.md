# PLAN.md — CVPR 2027 Point-Guided MRI Reconstruction

## /goal

Build the research-locked point-guided MRI frontend continuously from the current repository state, through the fixed spectral anchor, point-level cross-plane spectral evidence, completed bounded Gate-C trajectory, explicit lightweight Gate-D decoder, and completed Gate-E supervision, then stop before Gate-F execution.

Codex execution contract:

1. Read `PLAN.md` first.
2. Resolve actual HEAD before editing; never reset or overwrite newer user changes.
3. Execute every **UNBLOCKED** phase in order without asking for confirmation between phases.
4. For each phase: inspect the smallest relevant codegraph scope, implement only the locked design, add focused tests, run verification, fix failures, then update the phase completion log in this file.
5. Never invent a decision marked **OPEN / BLOCKED**.
6. Stop only when all currently unblocked phases pass, the next phase is a research gate, or verification cannot be fixed without changing a locked scientific contract.
7. Preserve fail-closed behavior: no fake T1ce, no silent random/pretrained fallback, no hidden reuse of legacy 3DGS modules.
8. Do not implement Gate-F-and-later behavior until explicitly unlocked.
9. Do not replace a locked component with a simpler equivalent merely for implementation convenience.
10. Preserve MAIN defaults and explicitly retained ablations.

Immediate locked target:

```text
T1 / T2 / FLAIR
       │
       ▼
MedicalNet ResNet10 shared observation encoder
       │
       │ post Conv1 + BN + ReLU
       │ BEFORE MaxPool
       │
       ├──────────────────────────────────────────┐
       │                                          │
       │                                          ▼
       │                                    F_shallow.detach()
       │                                          │
       │                                axis-conditioned projection
       │                                          │
       │                              Bxy / Bxz / Byz
       │                                          │
       │                               2-level 2D SWT-Haar
       │                                          │
       │                      7 bands / plane, same spatial grid
       │                                          │
       │                        shared 1×1 Conv2d, 64 → 8
       │                           applied identically per band
       │                                          │
       │                               concat → Axy/Axz/Ayz
       │                                   56 channels / plane
       │                                          │
       │                               refined point p_i*
       │                                          │
       │                           bilinear query all 3 planes
       │                                          │
       │                         f_xy / f_xz / f_yz ∈ R^56
       │                                          │
       │                    deterministic q=[LL2,E1,E2] ∈ R^24
       │                                          │
       │                         pairwise cosine agreement
       │                                          │
       │                            softmax reliability α
       │                                          │
       │               weighted concat → f_i^spec ∈ R^168
       │
       ▼
    MaxPool
       ↓
 Layer1..Layer4
       ↓
 minimal semantic head
       ↓
 S_coarse = [normal brain, edema, tumor-core candidate]
       ↓
 existing point refinement + semantic-aware sparse PoU
```

`A` is a fixed spectral reference within the future trajectory. `P*` is the refined point field. `f_i^spec` is point-level spectral evidence. The dynamic reconstruction state `Z_t` remains unresolved and must not be implemented yet.

---

# 0. Repository source of truth

Repository:

```text
repository: QuocKhanhLuong/3DGS
branch: main
plan refresh base commit: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1
message: docs: add phased point-guided spectral frontend plan
```

That base commit only added the previous plan. The source implementation underneath remained based on:

```text
4ccffcef0d3df0b2734335c34223fc98eda900af
Refactor code
```

Before implementation:

- resolve actual HEAD;
- inspect all commits newer than the plan refresh base;
- preserve compatible user changes;
- never assume phases are still pending solely because this file says so;
- update the completion log based on actual repository state.

---

# 1. Already implemented — do not redo

Current owner:

```text
src/smagm/features/point_guided/
```

Already present before this plan:

- `PointGuidedConfig`;
- RAS-mm / voxel geometry contracts;
- full feature-only MedicalNet ResNet10 layout;
- strict local checkpoint loading + SHA256 provenance;
- deterministic 1-channel → 3-channel stem adaptation;
- frozen backbone parameter policy;
- frozen BatchNorm running-stat policy;
- minimal `Conv3d(512,K,1)` semantic head;
- soft full-resolution `S_coarse`;
- quasi-uniform point initialization;
- directional sampling at ±1/±2/±3 mm along XYZ;
- small MLP offset predictor;
- L2 displacement <= 2 mm from original center;
- refined-center semantic sampling;
- fixed 4 mm sphere support;
- quadratic compact spatial affinity;
- L1 semantic affinity;
- multiplicative semantic-spatial affinity;
- sparse normalized PoU;
- frontend-only public forward contract;
- fail-closed full `forward()` before T1ce synthesis.

Do not rebuild these from scratch.

---

# 2. Locked observation + point architecture

## 2.1 Input

```text
x: [B,3,D,H,W]
0 = T1
1 = T2
2 = FLAIR
```

T1ce is the reconstruction/synthesis target and never an observation input.

## 2.2 Shared MedicalNet encoder

Use one MedicalNet 3D ResNet10 for both semantic and spectral branches.

Locked split:

```text
Conv1(7³,stride=2) → BN → ReLU → F_shallow
```

MAIN spectral tap is **after ReLU and before MaxPool**.

Semantic path continues through:

```text
F_shallow → MaxPool → Layer1 → Layer2 → Layer3 → Layer4
→ minimal semantic head → S_coarse
```

No second spectral encoder.

## 2.3 Frozen / detach policy

MAIN:

```text
freeze_backbone = True
detach_backbone_features = True
spectral_tap = conv1_pre_maxpool
```

When frozen:

- MedicalNet parameters receive no gradients;
- BatchNorm running statistics stay frozen;
- branch-exported features are detached when configured.

Required ablations:

```text
spectral_tap:
  conv1_pre_maxpool   MAIN
  layer1              ABLATION

backbone:
  frozen              MAIN
  fine_tuned          ABLATION

feature_detach:
  true                MAIN
  false               ABLATION
```

Branching, freezing, and detaching are separate concepts.

## 2.4 Coarse semantic prior

Exactly three soft classes:

```text
0 normal brain
1 edema
2 tumor-core candidate
```

No enhancing class because T1ce is unavailable as observation. No uncertainty class; uncertainty may later be derived from entropy/dispersion.

`S_coarse` is a prior for point construction/refinement, point semantic identity, and semantic-aware PoU. Do not turn it into an always-on dense trajectory conditioning branch without a later research decision.

## 2.5 Existing point contract

MAIN defaults:

```text
num_points = 2048
point_count_ablation = 3072
directional offsets = ±1, ±2, ±3 mm
support radius = 4 mm
max displacement = 2 mm from ORIGINAL point center
semantic distance = L1
spatial kernel = quadratic compact support
affinity = spatial × semantic
PoU = sparse + normalized
```

Offset input uses center T1/T2/FLAIR + center `S_coarse` + directional neighbor differences. Sampling is trilinear.

After refinement:

```text
p_i* = p_i0 + Δ_i
π_i = S_coarse(p_i*)
```

No sphere pooling for point semantics.

Do not add point split/merge/prune/radius/orientation/covariance learning and do not create dense `[B,N,D,H,W]` point tensors.

---

# 3. Locked base tri-plane projection B

Exact flow:

```text
MedicalNet Conv1 + BN + ReLU
          ↓
      F_shallow
        detach
          ↓
   spectral branch
          │
   ┌──────┼────────┐
   │      │        │
   ▼      ▼        ▼
Z-local Y-local  X-local
scorer  scorer   scorer
1×1×3  1×3×1   3×1×1
   │      │        │
 weights weights weights
   │      │        │
collapse collapse collapse
   Z      Y        X
   │      │        │
 Bxy     Bxz      Byz
```

No separate residual spectral adapter. No mean-branch + learned-branch double path.

Each orientation uses one lightweight scalar scorer:

```text
logits shape = [B,1,D,H,W]
```

not channel-wise attention.

For PyTorch tensor order `[B,C,D,H,W]`:

```text
Bxy: collapse D / physical Z
score kernel = (3,1,1), padding=(1,0,0)
output = [B,C,H,W]

Bxz: collapse H / physical Y
score kernel = (1,3,1), padding=(0,1,0)
output = [B,C,D,W]

Byz: collapse W / physical X
score kernel = (1,1,3), padding=(0,0,1)
output = [B,C,D,H]
```

The conceptual `1×1×3 / 1×3×1 / 3×1×1` notation is XYZ-oriented; implementation must document the mapping to PyTorch DHW explicitly.

### Zero initialization

Zero-initialize scorer weights and bias:

```text
logits = 0
softmax(logits, collapsed_axis) = uniform
```

Therefore the MAIN learned projection begins exactly as a mean projection, without a redundant residual mean branch.

Required projection modes:

```text
mean
max
pointwise_weighted
axis_local_weighted   MAIN
```

`pointwise_weighted` uses `Conv3d(C,1,kernel=1)` followed by axis softmax and weighted collapse.

Do not replace the MAIN projector with full 3D attention, channel-wise 5D attention, transformer blocks, 3×3×3 encoder stacks, or hard argmax selection.

---

# 4. Research Gate A — CLOSED: fixed SWT-Haar spectral anchor A

**Status: CLOSED / LOCKED — Phase 6 is complete.**

## 4.1 Anchor concept

```text
B = {Bxy,Bxz,Byz}
→ 2D SWT-Haar independently per plane
→ shared per-band channel projection
→ concatenate bands
→ A = {Axy,Axz,Ayz}
```

`A` is generated once per subject/forward and reused unchanged throughout future trajectory steps:

```text
A0 = A1 = ... = AK = A
```

This means the tensor is immutable across trajectory iterations. It does **not** mean the anchor-building modules are detached from the training objective.

## 4.2 Wavelet family

MAIN:

```text
2D Stationary / Undecimated Haar Wavelet Transform (SWT-Haar)
```

Do not use decimated DWT as MAIN.

Reason encoded by the research decision:

- point shifts are small and continuous;
- decimated Haar is phase/shift sensitive;
- MedicalNet Conv1 already reduces sampling density;
- SWT preserves the plane spatial grid;
- no extra point-coordinate rescaling is introduced by the wavelet transform;
- Haar remains simple, fixed, real-valued, and differentiable.

Standard decimated Haar may later be an ablation/baseline. DT-CWT is not MAIN and must not be introduced without a new decision.

## 4.3 Decomposition level and bands

Exactly two SWT levels.

Store exactly seven bands per plane in this fixed order:

```text
0 LL2
1 LH1
2 HL1
3 HH1
4 LH2
5 HL2
6 HH2
```

Equivalent notation:

```text
{LL2, LH1, HL1, HH1, LH2, HL2, HH2}
```

Do not store `LL1` as an eighth output band; it is the intermediate approximation used to produce level 2.

For an input plane `[B,C,H,W]`, every stored SWT band remains `[B,C,H,W]`.

## 4.4 Haar filters

Use fixed normalized Haar filters:

```text
L = [1, 1] / sqrt(2)
H = [1,-1] / sqrt(2)
```

Construct the four separable 2D filters `LL`, `LH`, `HL`, `HH`.

Implementation should be differentiable with fixed grouped `Conv2d`/equivalent tensor ops, stride 1, and no trainable wavelet parameters.

Level 2 uses the stationary/à-trous dilation appropriate to SWT rather than downsampling.

## 4.5 Boundary mode

MAIN:

```text
reflect padding
```

Do not use zero padding as MAIN because it can introduce artificial edge discontinuities/high-frequency coefficients. Do not use circular wrapping for MRI anatomy.

Output shape must equal input plane shape at both levels.

## 4.6 Shared per-band projection

MedicalNet `F_shallow` has 64 channels in the current ResNet10 design. Each SWT band therefore initially has 64 channels.

Use one **shared** `1×1 Conv2d`:

```text
φ: 64 → 8
```

Apply the exact same `φ` parameters independently to all seven bands and all three plane orientations.

Do not create seven separate band projectors.

The shared projection only compresses channels; it must not mix bands together before concatenation.

For each plane `p`:

```text
A_p = concat([
  φ(LL2),
  φ(LH1),
  φ(HL1),
  φ(HH1),
  φ(LH2),
  φ(HL2),
  φ(HH2),
], channel_dim)
```

Therefore:

```text
Axy = [B,56,H,W]
Axz = [B,56,D,W]
Ayz = [B,56,D,H]
```

and the channel layout must remain documented and stable:

```text
[ LL2 | LH1 | HL1 | HH1 | LH2 | HL2 | HH2 ]
    8      8     8     8     8     8     8
```

## 4.7 Normalization

MAIN:

```text
anchor_norm = none
```

Do not independently normalize raw SWT bands before the shared projection in MAIN because relative spectral energy is useful evidence.

Optional retained stability ablation only:

```text
anchor_norm = band_gn
GroupNorm(num_groups=7, num_channels=56)
```

Do not silently enable it.

## 4.8 Gradient / immutability contract

MAIN gradient path:

```text
future reconstruction loss
        ↓
future trajectory / consumer
        ↓
A
        ↓
shared 1×1 band projector
        ↓
fixed SWT-Haar ops
        ↓
axis-conditioned base projector
        ↓
F_shallow.detach()
        X
MedicalNet backbone
```

Therefore:

- `A` is **not detached** after construction;
- the shared `1×1` projection remains trainable;
- the axis-conditioned scorer remains trainable;
- Haar filters remain fixed;
- gradients stop at `F_shallow.detach()` in MAIN;
- `A` is computed once and not mutated/recomputed per trajectory step.

Do not confuse `fixed across trajectory` with `frozen from learning`.

---

# 5. Research Gate B — CLOSED: point spectral query + cross-plane reliability

**Status: CLOSED / LOCKED — Phase 7 is complete.**

Goal: convert the fixed tri-plane anchor and a refined 3D point into one point-level spectral evidence vector:

```text
(Axy, Axz, Ayz, p_i*) → f_i^spec
```

No transformer, no cross-attention, no learned confidence MLP, and no second spectral encoder.

## 5.1 B1 — point query

For refined physical point:

```text
p_i* = (x,y,z)
```

map from physical/RAS-mm coordinates into the shallow-feature/anchor coordinate system using existing geometry contracts.

Do **not** hardcode `coordinate / 2`, even though Conv1 stride is currently 2. The mapping must remain geometry-aware and testable.

Query the same 3D location from the three planes:

```text
Axy at (x,y)
Axz at (x,z)
Ayz at (y,z)
```

Use single-point **bilinear interpolation** on each 2D plane.

Do not use:

- nearest-neighbor query;
- 3×3 patch pooling;
- sphere pooling;
- dense point-to-plane tensors.

Because SWT is undecimated, `A_p` has the same spatial grid as its base plane `B_p`; there is no DWT-induced extra `/2` rescaling.

Each query returns:

```text
f_xy ∈ R^56
f_xz ∈ R^56
f_yz ∈ R^56
```

Bilinear sampling must preserve differentiability with respect to the queried point coordinate when the geometry path permits it.

## 5.2 B2 — deterministic consistency descriptor

Keep each raw 56-d plane feature unchanged for final evidence.

Do **not** insert an additional learned `56→d` projector before consistency.

Use the known 7-band layout to derive a smaller deterministic descriptor only for reliability estimation.

Split each `f_p` into seven 8-d blocks:

```text
LL2, LH1, HL1, HH1, LH2, HL2, HH2
```

Define orientation-insensitive energy per SWT scale, elementwise across the 8 projected channels:

```text
E1 = sqrt(LH1^2 + HL1^2 + HH1^2 + eps)
E2 = sqrt(LH2^2 + HL2^2 + HH2^2 + eps)
```

Then:

```text
q_p = concat([LL2, E1, E2])
q_p ∈ R^24
```

Use one small fixed numerical epsilon only for finite numerical stability. It is not a learned weight and not a research-tuned fusion coefficient.

Important separation:

```text
q_p   = only for cross-plane reliability
f_p   = raw 56-d spectral evidence retained for final output
```

The energy descriptor intentionally removes 2D LH/HL/HH orientation disagreement when asking whether planes agree on local spectral activity. It does **not** erase orientation information from the raw 56-d feature.

## 5.3 B3 — pairwise cross-plane agreement

Given:

```text
q_xy, q_xz, q_yz ∈ R^24
```

compute pairwise cosine similarities:

```text
s_xy_xz = cosine(q_xy, q_xz)
s_xy_yz = cosine(q_xy, q_yz)
s_xz_yz = cosine(q_xz, q_yz)
```

Reliability score for each plane is the mean agreement with the other two:

```text
r_xy = (s_xy_xz + s_xy_yz) / 2
r_xz = (s_xy_xz + s_xz_yz) / 2
r_yz = (s_xy_yz + s_xz_yz) / 2
```

Convert to normalized reliability weights:

```text
[α_xy, α_xz, α_yz] = softmax([r_xy, r_xz, r_yz])
```

Required invariant:

```text
α_xy + α_xz + α_yz = 1
α_p >= 0
```

No learned MLP/confidence head in MAIN.

Do not hard-drop an inconsistent plane. Soft reliability only reduces its contribution.

Known limitation, to document rather than silently “fix”: agreement is not ground truth. If two incorrect planes agree and one correct plane is an outlier, majority-style consistency can assign higher reliability to the incorrect pair. Do not inject semantic priors or a learned judge into Gate B without a new research decision.

## 5.4 B4 — final spectral evidence packing

B4 is a **summary/packing step**, not a new orientation-processing module.

Do not channel-wise sum the three raw 56-d plane features because LH/HL/HH meanings depend on plane orientation.

Do not create a new 3D-canonical 104-d representation here; axis/plane provenance is already encoded and should remain explicit.

Apply reliability to each raw plane feature:

```text
f~_xy = α_xy * f_xy
f~_xz = α_xz * f_xz
f~_yz = α_yz * f_yz
```

Then concatenate while preserving plane identity:

```text
f_i^spec = concat([f~_xy, f~_xz, f~_yz])
```

Final shape:

```text
f_i^spec ∈ R^168
```

Stable block provenance:

```text
channels   0:56   = reliability-weighted XY evidence
channels  56:112  = reliability-weighted XZ evidence
channels 112:168  = reliability-weighted YZ evidence
```

Within every 56-d block, preserve the 7-band/8-channel ordering from Gate A.

MAIN does **not** compress `168→64` or pass the packed feature through an MLP. A later trajectory/updater may consume this evidence only after Gate C is resolved.

Optional baseline for future ablation registry only:

```text
naive_concat = concat([f_xy,f_xz,f_yz])
```

MAIN:

```text
consistency_aware = weighted concat using α
```

---

# 6. Historical dynamic-trajectory boundary (superseded by completed Gate C)

Future high-level contract:

```text
A        = fixed spectral reference
P*       = refined sparse spatial/semantic carrier
f_i^spec = point-level reliability-aware spectral evidence
Z_t      = dynamic reconstruction tri-plane

Z0 → Z1 → ... → ZK
```

Still OPEN:

- how `Z0` is initialized;
- selector score and top-k policy;
- whether points may be revisited;
- local updater inputs and architecture;
- scatter/update overlap behavior;
- history/state representation;
- fixed K vs stopping/convergence;
- decoder from `Z_K` to T1ce;
- reconstruction/spectral/pathology losses;
- training schedule and differentiable selection strategy.

No implementation beyond placeholder interfaces that already exist.

---

# 7. Implementation phases

## Phase 0 — Resolve actual HEAD

**Status: COMPLETE — HEAD RESOLVED**

### /phase-goal

Work from the actual latest code and do not reimplement already merged work.

Tasks:

- [ ] `git status --short`
- [ ] `git rev-parse HEAD`
- [ ] inspect commits newer than plan refresh base `d623d179...`
- [ ] inspect `CODEBASE.md`
- [ ] inspect `CODEGRAPH.json`
- [ ] run the smallest relevant codegraph task, starting with `python scripts/codegraph.py --task frontend` if still valid
- [ ] inspect current point-guided public interfaces
- [ ] mark any already-completed phases truthfully instead of redoing them

Verify:

```bash
git diff --check
```

Proceed automatically to the first incomplete unblocked phase.

---

## Phase 1 — Shared MedicalNet intermediate-feature API

**Status: COMPLETE**

### /phase-goal

Expose the pre-MaxPool MAIN spectral tap and Layer1 ablation tap without duplicating the backbone or breaking semantic-prior compatibility.

Tasks:

- [ ] expose `Conv1→BN→ReLU` pre-MaxPool feature;
- [ ] expose Layer1 feature for ablation;
- [ ] expose deep/final feature needed by semantic head;
- [ ] preserve `forward_features()` compatibility;
- [ ] do not run the stem twice;
- [ ] keep checkpoint state-dict keys unchanged;
- [ ] keep 3-channel stem adaptation unchanged;
- [ ] document tensor shapes;
- [ ] test prepool tap is truly before pooling;
- [ ] test existing final feature output remains unchanged in eval mode.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_semantic_prior.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
git diff --check
```

Proceed automatically to Phase 2.

---

## Phase 2 — Frozen feature detach + ablation controls

**Status: COMPLETE**

### /phase-goal

Make the observation-encoder freeze/detach contract explicit and reusable by semantic and spectral branches.

Tasks:

- [ ] config for `detach_backbone_features`;
- [ ] config for spectral tap;
- [ ] MAIN tap = pre-MaxPool;
- [ ] Layer1 tap = ablation;
- [ ] preserve frozen BN eval behavior;
- [ ] detach exported branch features when configured;
- [ ] semantic head still receives gradients;
- [ ] future spectral projector still receives gradients;
- [ ] preserve frozen/fine-tuned and detached/non-detached ablations;
- [ ] add gradient-path tests.

Required tests:

1. frozen backbone has no trainable parameters;
2. BN running stats do not change;
3. detached shallow feature cannot backprop into MedicalNet;
4. a downstream trainable module still receives gradients;
5. `detach=False` remains valid for fine-tuning ablation.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
git diff --check
```

Proceed automatically to Phase 3.

---

## Phase 3 — Lock 3-class coarse semantics in code

**Status: COMPLETE**

### /phase-goal

Align implementation with the final coarse semantic meaning consumed by point refinement and PoU.

Tasks:

- [ ] production/main semantic count = exactly 3;
- [ ] define names/order in one explicit contract;
- [ ] reject accidental 4/6-class MAIN configuration;
- [ ] update semantic-prior tests;
- [ ] update point-descriptor shape tests;
- [ ] preserve softmax probabilities;
- [ ] do not add uncertainty channel;
- [ ] do not invent semantic supervision/loss here.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
git diff --check
```

Proceed automatically to Phase 4.

---

## Phase 4 — Axis-conditioned base tri-plane projector B

**Status: COMPLETE**

### /phase-goal

Implement `F_shallow.detach() → Bxy/Bxz/Byz` exactly as locked.

Suggested module:

```text
src/smagm/features/point_guided/triplane_projection.py
```

MAIN:

```text
XY: Conv3d(C,1,(3,1,1)) → softmax D → weighted sum D → [B,C,H,W]
XZ: Conv3d(C,1,(1,3,1)) → softmax H → weighted sum H → [B,C,D,W]
YZ: Conv3d(C,1,(1,1,3)) → softmax W → weighted sum W → [B,C,D,H]
```

Zero-init scorer kernels and biases.

Required modes:

- [ ] mean;
- [ ] max;
- [ ] pointwise_weighted;
- [ ] axis_local_weighted MAIN.

Required tests:

- [ ] exact plane shapes;
- [ ] softmax weights sum to one along collapsed axis;
- [ ] zero-init MAIN equals mean projection within tolerance;
- [ ] DHW/physical XYZ mapping correct using synthetic coordinate ramps;
- [ ] scorer logits are `[B,1,D,H,W]`;
- [ ] gradients reach scorer parameters;
- [ ] detached MedicalNet feature blocks backbone gradients;
- [ ] no second encoder introduced.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Proceed automatically to Phase 5.

---

## Phase 5 — Compose shared encoder through base planes B

**Status: COMPLETE**

### /phase-goal

Wire one shared MedicalNet execution into semantic and spectral branches through `B`, without yet changing point semantics.

Tasks:

- [ ] compute the shared MedicalNet stem once;
- [ ] route the configured shallow tap to spectral branch;
- [ ] continue the same encoder through semantic branch;
- [ ] respect detach config;
- [ ] produce `Bxy/Bxz/Byz`;
- [ ] expose base planes through typed internal/diagnostic outputs;
- [ ] preserve existing point/refinement/PoU outputs where practical;
- [ ] do not duplicate MedicalNet forward;
- [ ] do not change point behavior.

Tests:

- [ ] no duplicate complete MedicalNet forward;
- [ ] existing frontend invariants remain green;
- [ ] B planes deterministic in eval mode;
- [ ] frozen MedicalNet unchanged after projector-only optimizer step.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Proceed automatically to Phase 6. Gate A is CLOSED.

---

## Phase 6 — 2-level SWT-Haar spectral anchor A

**Status: COMPLETE — GATE A CLOSED / LOCKED**

### /phase-goal

Implement the fixed-grid differentiable 2-level SWT-Haar anchor exactly as locked and attach it to the base tri-plane branch.

Suggested modules:

```text
src/smagm/features/point_guided/swt_haar.py
src/smagm/features/point_guided/spectral_anchor.py
```

Names may be adapted to existing conventions, but responsibilities must stay separated and lightweight.

Tasks:

- [x] implement fixed normalized Haar low/high filters;
- [x] implement differentiable 2D stationary/undecimated transform with stride 1;
- [x] use level-appropriate dilation, no downsampling;
- [x] use reflect boundary handling;
- [x] produce exact seven-band order `LL2,LH1,HL1,HH1,LH2,HL2,HH2`;
- [x] keep every band on the same spatial grid as the input base plane;
- [x] implement one shared `1×1 Conv2d(64,8)`;
- [x] reuse the exact same projection parameters for every band and every plane;
- [x] concatenate seven projected 8-d bands → 56 channels;
- [x] MAIN normalization = none;
- [x] optional `band_gn` ablation is implemented and defaults off;
- [x] compute `Axy/Axz/Ayz` once per forward;
- [x] do not detach A;
- [x] ensure gradients reach shared band projection and base tri-plane scorer;
- [x] ensure gradients do not cross detached MedicalNet boundary in MAIN;
- [x] expose band layout as an explicit stable contract/constant rather than relying on magic slicing.

Required tests:

1. Haar filters are fixed/non-trainable and normalized as specified;
2. SWT output has no spatial downsampling;
3. all seven bands have exact input-plane spatial shape;
4. band order is stable and documented;
5. reflect boundary path preserves output shape for representative odd/even sizes;
6. level 2 genuinely uses stationary dilation rather than a second decimation;
7. `LL1` is not emitted in the final seven-band anchor;
8. shared `1×1` parameters are actually shared across all seven bands and three planes;
9. anchor shapes are exactly `[B,56,H,W]`, `[B,56,D,W]`, `[B,56,D,H]`;
10. MAIN path has no normalization;
11. optional band GroupNorm, if implemented, uses 7 groups / 56 channels and defaults off;
12. gradient reaches band projector;
13. gradient reaches axis-conditioned base projector;
14. gradient does not reach detached MedicalNet in MAIN;
15. anchor tensor is reused as a single result, not iteratively mutated.

Synthetic spectral tests:

- [x] constant plane produces negligible high-pass response away from unavoidable numerical tolerance/boundary behavior;
- [x] simple horizontal/vertical ramps verify LH/HL implementation convention;
- [x] test documents the actual `LH/HL` orientation convention rather than assuming a library convention;
- [x] small translations do not alter output grid/alignment.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_swt_haar.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_spectral_anchor.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Proceed automatically to Phase 7. Gate B is CLOSED.

---

## Phase 7 — Point spectral query + cross-plane reliability fusion

**Status: COMPLETE — GATE B CLOSED / LOCKED**

### /phase-goal

Query the fixed anchor at every refined point, derive deterministic cross-plane reliability, and return a 168-d reliability-weighted spectral evidence vector without adding a learned fusion encoder.

Suggested module boundaries:

```text
src/smagm/features/point_guided/spectral_query.py
src/smagm/features/point_guided/cross_plane_consistency.py
```

Names may follow existing package conventions.

### Phase 7A — geometry-aware bilinear query

Tasks:

- [x] map each refined RAS-mm point to the shallow-anchor coordinate system using existing geometry contracts;
- [x] do not hardcode Conv1 `/2` coordinate conversion;
- [x] query Axy using `(x,y)`;
- [x] query Axz using `(x,z)`;
- [x] query Ayz using `(y,z)`;
- [x] use bilinear interpolation;
- [x] return `f_xy/f_xz/f_yz`, each 56-d;
- [x] no patch pooling, sphere pooling, or nearest-neighbor mode in MAIN;
- [x] keep sampling differentiable with respect to point coordinates where supported by the existing geometry representation.

Tests:

- [x] exact-center query equals exact anchor pixel value;
- [x] fractional coordinate equals manual bilinear interpolation on a synthetic plane;
- [x] plane-axis coordinate mapping is correct using synthetic coordinate ramps;
- [x] SWT does not introduce extra coordinate scaling;
- [x] point perturbation changes sampled feature smoothly;
- [x] gradients with respect to continuous query coordinates are finite where expected;
- [x] no dense `[B,N,H,W]` or `[B,N,D,H,W]` helper tensor is created.

### Phase 7B — deterministic 24-d consistency descriptor

Tasks:

- [x] split each 56-d feature using the explicit seven-band layout;
- [x] compute `E1=sqrt(LH1²+HL1²+HH1²+eps)` elementwise;
- [x] compute `E2=sqrt(LH2²+HL2²+HH2²+eps)` elementwise;
- [x] build `q=[LL2,E1,E2]` → 24-d;
- [x] use fixed numerical epsilon only for stability;
- [x] retain original 56-d raw feature unchanged;
- [x] no learned `56→d` projector.

Tests:

- [x] q shape exactly 24;
- [x] E1/E2 equal manual calculation;
- [x] permuting LH/HL/HH inside the same scale leaves energy descriptor unchanged within tolerance;
- [x] raw 56-d feature remains unchanged and orientation-specific bands remain available.

### Phase 7C — pairwise reliability

Tasks:

- [x] compute the three pairwise cosine similarities;
- [x] compute each `r_p` as mean agreement with the other two planes;
- [x] softmax the three reliability scores;
- [x] expose `α_xy/α_xz/α_yz` for diagnostics/tests;
- [x] no confidence MLP;
- [x] no hard plane drop;
- [x] document the known majority-consistency limitation.

Tests:

- [x] identical q vectors produce equal reliability weights;
- [x] one synthetic outlier receives lower reliability than two mutually similar views;
- [x] α values are finite, nonnegative, and sum to one;
- [x] zero/near-zero descriptors remain numerically finite under the cosine implementation;
- [x] reliability has no trainable parameters.

### Phase 7D — weighted concat packing

Tasks:

- [x] `f~_xy = α_xy * f_xy`;
- [x] `f~_xz = α_xz * f_xz`;
- [x] `f~_yz = α_yz * f_yz`;
- [x] `f_i^spec = concat([f~_xy,f~_xz,f~_yz])`;
- [x] final point spectral feature = exactly 168-d;
- [x] preserve plane block provenance XY→XZ→YZ;
- [x] preserve seven-band ordering inside each plane block;
- [x] do not channel-wise sum planes;
- [x] do not introduce the previously considered 104-d canonical-orientation module;
- [x] do not add `168→64` MLP/compression in MAIN;
- [x] optional naive concat may exist only as an explicit baseline mode, never as MAIN.

Required integration tests:

- [x] refined points from existing frontend can query the produced anchor;
- [x] point count 2048 works without dense global point-volume allocation;
- [x] 3072 ablation remains shape-valid;
- [x] output spectral evidence shape is `[B,N,168]` or the repository's equivalent sparse/typed point-batch representation;
- [x] changing one plane reliability only scales that plane's 56-d block;
- [x] spectral query/fusion does not modify `S_coarse`, point coordinates, or PoU semantics;
- [x] gradient from a dummy downstream loss reaches shared band projector and axis scorer through `f_i^spec`;
- [x] gradient stops at MedicalNet shallow detach in MAIN.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_spectral_query.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_cross_plane_consistency.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

After Phase 7 passes, consult the dedicated post-Phase-7 plans before any
later-gate work. `PLAN_GATE_C_D_E.md` is the locked authority for C/D/E and
`PLAN_GATE_F_G.md` is the locked baseline authority for F/G.

---

# 8. Research Gate C — Dynamic trajectory

**Status: COMPLETE — C1–C7, Gate D D1, and Gate E E1–E9; Gate F server software is ready**

The historical Gate-C research questions are resolved by the immutable,
dedicated `PLAN_GATE_C_D_E.md` plan materialized from
`origin/main:PLAN_GATE_C_D_E.md` (blob
`3ce39700402468894cf462c3edb9cc9f69d91316`). Gate C is complete with only
its C1–C7 sequence: shared `B -> Z0`, dynamic state/reward query, explicit
reward-cost utility, adaptive selection, shared local updater, compact 4-mm
write-back, and trajectory composition. It does not decode or synthesize T1ce.

Gate D D1 is complete: final `DynamicTriPlanes` are queried in chunks through
the shared feature-grid geometry and decoded by the locked `96 -> 64 -> 32 ->
1` SiLU MLP through an explicit API, while generic `forward()` remains
fail-closed. Gate E E1–E9 is complete only as a typed target-after-inference
supervision objective; it adds no optimizer or training loop. Gate F/G have
the immutable dedicated `PLAN_GATE_F_G.md` baseline plan (blob
`1c9b01569126309eada53ce0a90309353875a9c6`); Gate F F1/F2 are complete and
the additive F3/F4 server software is ready, but neither experiment has been
executed here. Gate G G1-G4 software is complete; trained-checkpoint and
held-out results remain pending server execution. Gate H has no locked local
plan and remains default deny.

---

# 9. Codegraph / ownership update

The existing codegraph predates the newly unlocked spectral anchor and cross-plane modules.

When implementing Phases 4–7:

- extend the point-guided/frontend task only with the smallest new read/write paths;
- optionally create narrowly scoped spectral tasks if the repository codegraph convention supports them;
- preserve default-deny behavior;
- do not unblock legacy `anchors/**`, `fields/**`, `routing/**`, reconstruction, training, or unrelated data packages.

Expected new files, subject to existing naming conventions:

```text
src/smagm/features/point_guided/triplane_projection.py
src/smagm/features/point_guided/swt_haar.py
src/smagm/features/point_guided/spectral_anchor.py
src/smagm/features/point_guided/spectral_query.py
src/smagm/features/point_guided/cross_plane_consistency.py

tests/features/point_guided/test_triplane_projection.py
tests/features/point_guided/test_swt_haar.py
tests/features/point_guided/test_spectral_anchor.py
tests/features/point_guided/test_spectral_query.py
tests/features/point_guided/test_cross_plane_consistency.py
```

Gate-C runtime files are owned only by the narrow `trajectory` codegraph task;
do not create Gate-D/E/F/G runtime files in that task.

---

# 10. Configuration contract

MAIN values should be explicit and centralized rather than spread as magic constants.

Target configuration semantics:

```text
num_semantic_classes = 3
num_points = 2048
support_radius_mm = 4.0
max_displacement_mm = 2.0

freeze_backbone = true
detach_backbone_features = true
spectral_tap = conv1_pre_maxpool

projection_mode = axis_local_weighted

wavelet_family = swt_haar
wavelet_levels = 2
wavelet_boundary = reflect
wavelet_band_order = [LL2,LH1,HL1,HH1,LH2,HL2,HH2]
wavelet_band_channels = 8
anchor_channels = 56
anchor_norm = none

spectral_query = bilinear
consistency_descriptor = ll2_energy12
consistency_similarity = pairwise_cosine
consistency_weighting = softmax_mean_agreement
spectral_fusion = reliability_weighted_concat
point_spectral_channels = 168
```

Do not expose arbitrary research knobs simply because the implementation could support them.

---

# 11. Ablation registry

Support only intentional ablations.

```text
MedicalNet tap:
  conv1_pre_maxpool   MAIN
  layer1              ABLATION

Backbone:
  frozen              MAIN
  fine_tuned          ABLATION

Feature detach:
  true                MAIN
  false               ABLATION

Point count:
  2048                MAIN
  3072                ABLATION

Directional context:
  center_only         ABLATION
  ±1mm                ABLATION
  ±1/2/3mm            MAIN

PoU affinity:
  spatial_only        ABLATION
  spatial_x_semantic  MAIN

Base projection:
  mean                ABLATION
  max                 ABLATION
  pointwise_weighted  ABLATION
  axis_local_weighted MAIN

Wavelet:
  standard decimated Haar   FUTURE BASELINE if explicitly added
  2-level SWT-Haar          MAIN

Anchor channel projection:
  shared 64→8 per band      MAIN

Anchor normalization:
  none                      MAIN
  band_gn                   STABILITY ABLATION

Cross-plane:
  naive_concat              BASELINE
  consistency_aware         MAIN
```

Do not automatically run every Cartesian product.

---

# 12. Scientific invariants

Every unblocked phase must preserve:

1. T1ce never enters observation input.
2. Observation channel order stays T1/T2/FLAIR.
3. MedicalNet checkpoint behavior stays fail-closed and provenance-aware.
4. Frozen BatchNorm statistics do not mutate.
5. MAIN spectral tap is after Conv1+BN+ReLU and before MaxPool.
6. MAIN shallow branch feature is detached.
7. Coarse semantic meaning is exactly normal brain / edema / tumor-core candidate.
8. Point displacement remains <= 2 mm from original center.
9. Point support radius remains exactly 4 mm.
10. PoU remains sparse and normalized.
11. No dense `[B,N,D,H,W]` point tensor.
12. No second heavy spectral encoder.
13. Base tri-plane projection remains axis-local scalar-weighted collapse.
14. SWT-Haar is 2-level, undecimated, reflect-padded, with exactly seven stored bands.
15. Wavelet band projection is one shared trainable `1×1`, 64→8.
16. Anchor plane size remains aligned with the corresponding base plane.
17. Anchor has 56 channels per plane in fixed band order.
18. MAIN anchor normalization is none.
19. `A` is fixed across future trajectory iterations but not detached from learning.
20. Gate B query is bilinear and geometry-aware.
21. Consistency descriptor is deterministic 24-d `[LL2,E1,E2]`.
22. Reliability is pairwise cosine → mean agreement → softmax.
23. No learned confidence MLP in MAIN.
24. Final point spectral evidence is reliability-weighted concat, 168-d.
25. Plane identity and band identity remain recoverable from `f_i^spec`.
26. No 104-d canonical-orientation fusion module is added at Gate B.
27. No `168→d` learned compression at Gate B MAIN.
28. Full `PointGuidedMRIModel.forward()` still refuses unresolved T1ce synthesis.
29. No silent reuse of legacy 3DGS reconstruction modules.
30. No trajectory implementation before Gate C is unlocked.

---

# 13. Final verification for all currently unblocked phases

After Phase 7:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Also report:

```text
current HEAD
files changed
tests changed/added
x shape
F_shallow shape
F_layer1 shape
F_deep shape
S_coarse shape
Bxy/Bxz/Byz shapes
7 raw SWT band shapes per plane
Axy/Axz/Ayz shapes
queried f_xy/f_xz/f_yz shape
q_xy/q_xz/q_yz shape
α shape and normalization
f_i^spec shape
whether a MedicalNet checkpoint was actually loaded during tests
frozen/detach behavior
parameter count of three axis-local scorers
parameter count of shared 1×1 band projector
confirmation Haar filters have zero trainable parameters
peak/approximate tensor sizes for N=2048 and N=3072 if easily measurable
next blocked research gate = Gate C
```

Do not claim reconstruction quality, clinical validity, lesion fidelity, or publication performance from software tests.

---

# 14. Phase completion log

Reconciled from the actual merged repository state during M6 governance review:

```text
Phase 0: COMPLETE — actual HEAD resolved at 6f754eda48de49fac33cdf87c135f9dbdadc76f6
Phase 1: COMPLETE — shared MedicalNet intermediate features
Phase 2: COMPLETE — freeze, detach, and tap controls
Phase 3: COMPLETE — exact three-class coarse semantic contract
Phase 4: COMPLETE — static axis-conditioned base tri-plane projector B
Phase 5: COMPLETE — one-pass semantic and B diagnostic composition
  implementation commit: bc6d2e60ae5d02aae11faaadd115be9b73a0687f
  merged repository state: 6f754eda48de49fac33cdf87c135f9dbdadc76f6
  historical software evidence: 94 point-guided tests, compileall, git diff --check

Research Gate A — SWT-Haar anchor: CLOSED / LOCKED
Phase 6: COMPLETE — fixed two-level SWT-Haar static spectral anchor A
  implementation worktree: feature/point-guided-spectral-anchor-phase6
  files: swt_haar.py, spectral_anchor.py, config.py, contracts.py, model.py,
         Phase-6 focused/integration tests, and current-boundary documentation
  software evidence: SWT tests 15 passed; spectral-anchor tests 6 passed;
                     frontend/boundary/triplane integration tests 48 passed;
                     full point-guided suite 121 passed; compileall, codegraph,
                     tests/test_codegraph.py, and git diff --check passed

Research Gate B — point query + cross-plane reliability: CLOSED / LOCKED
Phase 7: COMPLETE — geometry-aware bilinear query, deterministic 24-d
  descriptor, cosine/softmax reliability, and XY/XZ/YZ-preserving 168-d
  point spectral evidence
  implementation worktree: feature/point-guided-spectral-anchor-phase6
  files: spectral_query.py, cross_plane_consistency.py, contracts.py,
         model.py, Phase-7 focused/integration tests, and status documentation
  software evidence: query tests 20 passed; consistency tests 7 passed;
                     frontend/boundary/anchor integration tests 42 passed;
                     full point-guided suite 151 passed

Research Gate C — adaptive reward-cost trajectory: COMPLETE — C1-C7 only
Research Gate D — implicit decoder: COMPLETE — D1 only; explicit final-Z
chunked decoding, no generic-forward policy or supervision
Research Gate E — supervision: COMPLETE — E1–E9 only; target-after-inference
typed objective, no optimizer/training loop
Research Gate F – baseline training: F1/F2 COMPLETE; F3 tiny overfit and F4
full train/validation SOFTWARE READY – NOT YET EXECUTED ON SERVER
Research Gate G – G1-G4 SOFTWARE COMPLETE; trained-checkpoint and held-out
results PENDING SERVER EXPERIMENT
Gate H — default deny; no locked local plan
Human Gate: PENDING

Current Gate-C completion record (supersedes the pre-completion Gate-C entry
immediately above): COMPLETE — C1-C7 only; bounded dynamic-state diagnostics,
no decoder, target data, losses, or T1ce output; software evidence: full
point-guided suite 181 passed.

Gate-D completion record: COMPLETE — D1 only; source `decoder.py`, explicit
`forward_reconstruction`, and focused decoder tests; working tree based on
`965f29dc095253736a963e382d9a96ded39a0d86`; software evidence: 15 decoder
tests and full point-guided suite 198 passed. This historical D1 record
preceded Gate-E activation; no Gate-F/G behavior was added.

Gate-E completion record: COMPLETE — E1–E9 only; sources `losses.py`,
`reward_supervision.py`, `training_objective.py`, and the target-free context
seam in `model.py`. T1ce enters only after inference context construction;
there is no optimizer, scheduler, training loop, checkpoint, Gate-F, or Gate-G
behavior. Software evidence: focused E/C/D/boundary regression 72 passed;
full point-guided suite 223 passed; `compileall`, frontend/trajectory/decoder/
supervision codegraph scopes, and `tests/test_codegraph.py` (5 passed) all
passed; working and cached diff checks passed. Protected Phase-6 staged index
tree remained `28d32fe8906a44365f31cd5bd7a98b9621b05f71`.

Gate-F F1/F2 record: COMPLETE. The human decision classifies the existing
`point_refiner.offset_predictor` as a MAIN Gate-F trainable with 1,419
parameters and a baseline-optimizer membership. This resolves only optimizer
ownership: its architecture, observation-only inputs, deterministic point
initialization, and the hard displacement bound of at most 2 mm are unchanged.
The synthetic target-after-inference smoke completed forward, objective,
backward, and one Adam step with finite nonzero gradients and updates for every
authorized trainable, including the offset predictor. The additive
`server_pipeline` task now owns the full-volume data adapter, semantic
grounding, trainer, checkpoints, and evaluation CLI. Gate G G1-G4 software is
complete; experimental execution and held-out evaluation remain pending until
an F4 trained checkpoint and an untouched real test split exist.
Gate-G `candidate_evaluations` reports actual dense RewardNet scores; the
separate `eligible_candidate_evaluations` reports candidates still eligible
before exact no-revisit utility masking. Neither metric implies skipped dense
query/reward computation.
F3 tiny overfit: SOFTWARE READY — NOT YET EXECUTED ON SERVER. F4 full
train/validation: SOFTWARE READY — NOT YET EXECUTED ON SERVER. The additive
full-volume adapter is separate from the legacy sparse-plane adapter; no
training or validation result is claimed until the server commands run.
```

For every completed phase record:

```text
status:
HEAD:
files changed:
verification:
remaining assumptions:
```

Codex should update this log as it works.

---

# 15. Scope guard

Do not drift into:

- 3D Gaussian Splatting;
- Gaussian opacity/covariance;
- sparse-slice reconstruction;
- a second heavy encoder;
- a BraTS-specialist model that pre-solves the missing-modality problem;
- T1ce conditioning;
- local per-point FFT;
- full 3D DWT/3D wavelet evidence volume as replacement for the tri-plane anchor;
- decimated Haar as MAIN;
- DT-CWT as MAIN;
- dynamic mutation of spectral anchor A;
- transformer/cross-attention spectral fusion;
- learned reliability MLP at Gate B;
- hard plane selection;
- canonical 104-d orientation fusion at Gate B;
- learned 168-d compression at Gate B;
- decoder, reconstruction loss, or any Gate-D-and-later runtime before its explicit gate;
- legacy anchor/field/routing packages unless a later research decision explicitly reuses them.

Gate D D1 and Gate E E1-E9 are complete through their dedicated tasks. Gate F
F1/F2 are complete and F3/F4 software is ready for server execution. Gate G
G1-G4 software is complete; experimental execution and held-out testing remain
pending server evidence.
