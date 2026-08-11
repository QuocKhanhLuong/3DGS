# PLAN.md — CVPR 2027 Point-Guided MRI Reconstruction

## /goal

Build the next research-locked stages of the point-guided MRI frontend continuously from the current repository state.

Codex execution contract:

1. Resolve the smallest applicable codegraph scope first, then read `PLAN.md`
   within that declared scope.
2. Resolve actual HEAD before editing; never reset newer user changes.
3. Execute only a phase explicitly assigned by the active task. `AUTHORIZED —
   NOT IMPLEMENTED` grants scope but never triggers automatic progression.
4. For each phase: inspect the smallest relevant codegraph scope, implement only the locked design, add focused tests, run verification, fix failures, then update the phase log in this file.
5. Never invent a decision marked **OPEN / BLOCKED**.
6. Stop at the active task's terminal condition, at a research gate, or when a
   failure cannot be fixed without changing a locked scientific contract.
7. Preserve fail-closed behavior: no fake T1ce, no silent random/pretrained fallback, no hidden reuse of legacy 3DGS modules.
8. Do not implement trajectory/decoder/reconstruction until explicitly unlocked.

### M0 authority reconciliation

M0 policy explicitly authorized implementation Phases 1–5 in the sequence
below: shared MedicalNet feature exposure, detach/tap controls, the exact
three-class semantic contract, static feature-only base planes `Bxy/Bxz/Byz`,
and their diagnostic frontend composition. Their completed implementation
status is recorded in the phase log below. This authorization is bounded by
`AGENTS.md` and `CODEGRAPH.json`; it never authorizes a second encoder, legacy
reuse, target-derived conditioning, anchor `A`, cross-plane fusion, dynamic
trajectory, decoder, reconstruction loss, or T1ce synthesis. Phases 6+ remain
blocked by their corresponding research gates.

Immediate target:

```text
T1 / T2 / FLAIR
       │
       ▼
MedicalNet ResNet10 shared observation encoder
       │
       │ post Conv1 + BN + ReLU
       │ BEFORE MaxPool
       │
       ├───────────────────────────────┐
       │                               │
       │                               ▼
       │                         F_shallow.detach()
       │                               │
       │                        spectral branch
       │                               │
       │                 ┌─────────────┼─────────────┐
       │                 │             │             │
       │                 ▼             ▼             ▼
       │              Z-local       Y-local       X-local
       │              scorer        scorer        scorer
       │               1×1×3         1×3×1         3×1×1
       │                 │             │             │
       │              weights       weights       weights
       │                 │             │             │
       │            collapse Z    collapse Y    collapse X
       │                 │             │             │
       │                Bxy           Bxz           Byz
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

The base tri-plane projection above is the last newly locked spectral component. Wavelet details and exact cross-plane consistency fusion remain research gates.

---

# 0. Repository source of truth

Plan authored against:

```text
repository: QuocKhanhLuong/3DGS
branch: main
commit: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1
message: docs: add phased point-guided spectral frontend plan
```

Before implementation, compare current HEAD with this commit. If newer, inspect the diff, preserve compatible changes, and update this section.

---

# 1. Already implemented — do not redo

Current owner: `src/smagm/features/point_guided/`.

Already present:

- `PointGuidedConfig`
- RAS-mm / voxel geometry contracts
- full feature-only MedicalNet ResNet10 layout
- strict local checkpoint loading + SHA256 provenance
- deterministic 1-channel → 3-channel stem adaptation
- frozen backbone parameter policy
- frozen BatchNorm running-stat policy
- minimal `Conv3d(512,K,1)` semantic head
- soft full-resolution `S_coarse`
- quasi-uniform point initialization
- directional sampling at ±1/±2/±3 mm along XYZ
- small MLP offset predictor
- L2 displacement <= 2 mm from original center
- refined-center semantic sampling
- fixed 4 mm sphere support
- quadratic compact spatial affinity
- L1 semantic affinity
- multiplicative semantic-spatial affinity
- sparse normalized PoU
- frontend-only public forward contract
- fail-closed full `forward()` before T1ce synthesis

Do not rebuild these from scratch.

---

# 2. Locked architecture

## 2.1 Input

```text
x: [B,3,D,H,W]
0=T1, 1=T2, 2=FLAIR
```

T1ce is not an observation input.

## 2.2 Shared MedicalNet encoder

Single MedicalNet ResNet10 serves both branches.

Split point is locked:

```text
Conv1(7³,stride=2) → BN → ReLU → F_shallow
```

Spectral branch taps **after ReLU, before MaxPool**.

Semantic path continues:

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
- BatchNorm running stats stay frozen;
- exported branch features are detached/stop-gradient when configured.

Required ablation support:

```text
spectral_tap:
  conv1_pre_maxpool   MAIN
  layer1

backbone:
  frozen              MAIN
  fine_tuned

feature_detach:
  true                MAIN
  false
```

Branching and freezing are separate concepts.

## 2.4 Coarse semantic prior

Locked soft channels:

```text
0 normal brain
1 edema
2 tumor-core candidate
```

No separate enhancing class. No uncertainty class; derive uncertainty later from entropy if needed.

`S_coarse` is for point refinement, point semantic identity, and semantic-aware PoU. It must not become a permanent dense trajectory-conditioning branch.

## 2.5 Existing point contract

MAIN defaults remain:

```text
num_points = 2048
alternative = 3072
directional offsets = ±1, ±2, ±3 mm
support radius = 4 mm
max displacement = 2 mm
semantic distance = L1
spatial kernel = quadratic compact
affinity = spatial × semantic
PoU = sparse + normalized
```

No point split/merge/prune/radius/orientation/covariance learning and no dense `[B,N,D,H,W]`.

---

# 3. Newly locked base tri-plane projection

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

No separate residual spectral adapter. No `mean branch + learned branch` double path.

The axis-conditioned scorer starts as a mean-like projection by zero initialization.

For tensor-order clarity, PyTorch input is `[B,C,D,H,W]`:

```text
Bxy: collapse D/Z
score kernel = (3,1,1), padding=(1,0,0)
output = [B,C,H,W]

Bxz: collapse H/Y
score kernel = (1,3,1), padding=(0,1,0)
output = [B,C,D,W]

Byz: collapse W/X
score kernel = (1,1,3), padding=(0,0,1)
output = [B,C,D,H]
```

The conceptual 1×1×3 / 1×3×1 / 3×1×1 naming refers to XYZ orientation; implementation must document the DHW mapping explicitly.

### Uniform initialization

Zero initialize scorer weights and bias. Then:

```text
logits = 0
softmax(logits, collapse_axis) = uniform
```

so the learned weighted projection equals mean projection at initialization without adding a redundant residual mean branch.

Do not replace this with full 3D attention, channel-wise 5D attention, transformers, 3×3×3 encoder stacks, or hard argmax selection.

---

# 4. Required projection ablations

Support:

```text
projection_mode:
  mean
  max
  pointwise_weighted
  axis_local_weighted   MAIN
```

`pointwise_weighted` = `Conv3d(C,1,kernel=1)` + axis softmax + weighted sum.

`axis_local_weighted` = axis-local size-3 scorer + axis softmax + weighted sum.

Do not add unrelated ablations.

---

# 5. Wavelet spectral anchor

## LOCKED concept

```text
B = {Bxy,Bxz,Byz}
→ 2D wavelet per plane
→ A = {Axy,Axz,Ayz}
```

`A` is generated once per subject and held fixed through trajectory steps:

```text
A0 = A1 = ... = AK = A
```

Fixed means the anchor tensor is not iteratively mutated by trajectory. It does not automatically mean all anchor-producing parameters must be frozen in every future ablation.

## OPEN / BLOCKED

Not yet locked:

- wavelet family;
- decomposition level;
- padding/boundary mode;
- subband packing;
- normalization;
- post-DWT projection.

Wavelet is chosen over FFT/DCT, but Codex must not choose Haar/db2/etc. automatically.

---

# 6. Cross-plane spectral consistency

## LOCKED concept

A refined point `p_i=(x,y,z)` queries:

```text
Axy(x,y)
Axz(x,z)
Ayz(y,z)
```

Final main fusion must account for cross-plane consistency/reliability rather than blindly concatenate the three views.

## OPEN / BLOCKED

Still unresolved:

- consensus formulation;
- reliability score;
- shared projector;
- normalization;
- optional consistency loss.

Do not invent transformer/cross-attention.

---

# 7. Dynamic trajectory remains blocked

Future:

```text
A   = fixed spectral reference
P*  = refined sparse spatial/semantic carrier
Z_t = dynamic reconstruction tri-plane

Z0 → Z1 → ... → ZK
```

Still open: Z0, selector, updater, history, stopping, decoder, reconstruction losses.

No implementation beyond interfaces.

---

# 8. Implementation phases

## Phase 0 — Rebase plan on actual HEAD

**Status: COMPLETE — policy/architecture reconciliation only; no model implementation.**

### /phase-goal

Work from actual latest code and never reimplement already merged frontend work.

Tasks:

- [x] `git status --short`
- [x] `git rev-parse HEAD`
- [x] compare HEAD with `4ccffcef...`
- [x] inspect `CODEBASE.md`
- [x] inspect `CODEGRAPH.json`
- [x] run `python scripts/codegraph.py --task frontend`
- [x] inspect current point-guided public interfaces
- [x] update this plan source commit

Verify:

```bash
git diff --check
```

Stop after M0. Phase 1 requires an explicit active task.

---

## Phase 1 — Shared MedicalNet intermediate-feature API

**Status: COMPLETE**

### /phase-goal

Expose the pre-MaxPool spectral tap and Layer1 ablation tap without duplicating the backbone or breaking the semantic prior.

Tasks:

- [x] expose `Conv1→BN→ReLU` pre-MaxPool feature
- [x] expose Layer1 feature for ablation
- [x] preserve `forward_features()` final-feature compatibility
- [x] do not run the stem twice
- [x] keep checkpoint state-dict keys unchanged
- [x] keep 3-channel stem adaptation unchanged
- [x] document tensor shapes
- [x] test that the prepool tap is truly before pooling
- [x] test that the existing final feature output is unchanged in eval mode

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_semantic_prior.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
git diff --check
```

Stop after Phase 1. Phase 2 requires an explicit active task.

---

## Phase 2 — Frozen feature detach + ablation controls

**Status: COMPLETE**

### /phase-goal

Make the frozen observation-encoder contract explicit and reusable by semantic and spectral branches.

Tasks:

- [x] add config for `detach_backbone_features`
- [x] add config for spectral tap
- [x] MAIN tap = pre-MaxPool
- [x] Layer1 tap = ablation
- [x] preserve frozen BN eval behavior
- [x] detach branch features when configured
- [x] ensure semantic head still receives gradients
- [x] ensure future projection receives gradients
- [x] add frozen-vs-finetuned and detach-vs-nondetach support
- [x] add gradient-path tests

Required tests:

1. frozen backbone has no trainable parameters;
2. BN running stats do not change;
3. detached shallow feature cannot backprop into MedicalNet;
4. downstream trainable module can still receive gradients;
5. `detach=False` remains valid for later fine-tuning ablation.

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
git diff --check
```

Stop after Phase 2. Phase 3 requires an explicit active task.

---

## Phase 3 — Lock 3-class coarse semantics in code

**Status: COMPLETE**

### /phase-goal

Align the implementation with the final coarse semantic meaning consumed by refinement and PoU.

Tasks:

- [x] production/main semantic count = exactly 3
- [x] define names/order in one explicit contract
- [x] reject accidental 4/6-class main configuration
- [x] update semantic-prior tests
- [x] update point-descriptor shape tests
- [x] preserve softmax probabilities
- [x] do not add uncertainty channel
- [x] do not invent semantic supervision/loss here

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
git diff --check
```

Stop after Phase 3. Phase 4 requires an explicit active task.

---

## Phase 4 — Axis-conditioned base tri-plane projector

**Status: COMPLETE**

### /phase-goal

Implement `F_shallow.detach() → Bxy/Bxz/Byz` exactly as locked, without a second encoder or redundant residual projection.

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

Zero-init scorer kernels/biases.

Required ablation modes:

- [x] mean
- [x] max
- [x] pointwise_weighted
- [x] axis_local_weighted MAIN

Required tests:

- [x] exact plane shapes
- [x] softmax weights sum to 1 along collapsed axis
- [x] zero-init main projector equals mean projection within tolerance
- [x] orientation mapping correct using synthetic coordinate ramps
- [x] scorer logits are `[B,1,D,H,W]`, not channel-wise attention
- [x] gradients reach scorer parameters
- [x] detached MedicalNet feature blocks backbone gradients
- [x] no second encoder introduced

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Stop after Phase 4. Phase 5 requires an explicit active task.

---

## Phase 5 — Compose spectral branch through B only

**Status: COMPLETE — STOPPED AT RESEARCH GATE A**

### /phase-goal

Wire the shared MedicalNet tap and the base tri-plane projector into the frontend, stopping before wavelet.

Tasks:

- [x] compute shared MedicalNet stem once
- [x] route the configured selected shared feature to the static B branch
- [x] continue the same feature bundle through the semantic branch
- [x] respect detach config
- [x] produce Bxy/Bxz/Byz
- [x] expose base planes through typed `FrontendOutput.base_planes`
- [x] preserve existing point/refinement/PoU public outputs
- [x] do not call B a spectral anchor yet
- [x] do not add fake FFT/DWT
- [x] do not change point behavior

Tests:

- [x] shared backbone path does not perform two complete MedicalNet forwards
- [x] existing frontend invariants remain green
- [x] B planes deterministic in eval mode
- [x] frozen MedicalNet remains unchanged after projector-only optimizer step

Verify:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

After Phase 5, STOP at Research Gate A.

---

# 9. Research Gate A — Wavelet details

**Status: BLOCKED**

Need human/research decision for:

- wavelet family;
- level;
- boundary mode;
- band packing;
- normalization;
- post-DWT projection.

Already decided: 2D wavelet is applied on the three base planes and forms a fixed spectral tri-plane anchor.

## Phase 6 — Wavelet anchor A

**Status: BLOCKED BY GATE A**

Target only after unlock:

```text
Bxy → DWT → Axy
Bxz → DWT → Axz
Byz → DWT → Ayz
```

---

# 10. Research Gate B — Cross-plane consistency

**Status: BLOCKED**

Already decided: all three planes are queried at each refined 3D point and main fusion is consistency/reliability-aware.

Exact mechanism remains open.

## Phase 7 — Point spectral query + consistency fusion

**Status: BLOCKED BY GATE B**

No implementation until gate is resolved.

---

# 11. Research Gate C — Dynamic trajectory

**Status: BLOCKED**

Need decisions for Z0, selector, updater, history, stopping, decoder, losses.

Do not pull in legacy anchor/field/routing implementations automatically.

---

# 12. Codegraph / ownership update

M0 pre-authorizes the exact Phase 1–5 paths in the `frontend` and `tests`
tasks while preserving default deny. Do not broaden those paths or add a
separate spectral task merely to bypass their boundary. In particular, do not
unblock legacy `anchors/**`, `fields/**`, `routing/**`, reconstruction,
training, or data packages.

The reserved Phase 4 paths are:

```text
src/smagm/features/point_guided/triplane_projection.py
tests/features/point_guided/test_triplane_projection.py
```

Do not create wavelet, cross-plane fusion, dynamic trajectory, decoder, or
reconstruction files until the corresponding research gate is resolved.

---

# 13. Ablation registry

```text
MedicalNet tap:
  conv1_pre_maxpool   MAIN
  layer1

Backbone:
  frozen              MAIN
  fine_tuned

Feature detach:
  true                MAIN
  false

Point count:
  2048                MAIN
  3072

Directional context:
  center_only
  ±1mm
  ±1/2/3mm            MAIN

PoU affinity:
  spatial_only
  spatial_x_semantic  MAIN

Base projection:
  mean
  max
  pointwise_weighted
  axis_local_weighted MAIN

Spectral later:
  anchor_off
  anchor_on           MAIN after implementation

Cross-plane later:
  naive_concat        baseline only
  consistency_aware   MAIN after research lock
```

Support ablations; do not automatically run every Cartesian combination.

---

# 14. Scientific invariants

Every unblocked phase must preserve:

1. T1ce never enters observation input.
2. channel order stays T1/T2/FLAIR.
3. MedicalNet checkpoint behavior stays fail-closed/provenance-aware.
4. frozen BN statistics do not mutate.
5. main spectral tap is before MaxPool.
6. main shallow branch feature is detached.
7. coarse semantic meaning is exactly normal/edema/tumor-core candidate.
8. displacement <= 2 mm from original point center.
9. support radius = 4 mm.
10. PoU stays sparse and normalized.
11. no `[B,N,D,H,W]` dense point tensor.
12. no second heavy spectral encoder.
13. no fake spectral anchor before wavelet is locked.
14. full `PointGuidedMRIModel.forward()` still refuses unresolved T1ce synthesis.
15. no silent reuse of legacy 3DGS reconstruction modules.

---

# 15. Final verification for all currently unblocked phases

After Phase 5:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Report:

```text
current HEAD
files changed
tests changed/added
shapes: x, F_shallow, F_layer1, F_deep, S_coarse, Bxy/Bxz/Byz
whether a MedicalNet checkpoint was actually loaded during tests
frozen/detach behavior
parameter count of three axis-local scorers
next blocked research gate
```

Do not claim reconstruction or clinical performance from software tests.

---

# 16. Phase completion log

```text
Phase 0: COMPLETE — M0 policy/architecture reconciliation at d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1; policy/docs/permissions only
Phase 1: COMPLETE — shared typed MedicalNet shallow/Layer1/deep feature API; no later phase started
  status: COMPLETE
  HEAD: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1
  files changed: medicalnet_resnet10.py, semantic_prior.py, test_semantic_prior.py, PLAN.md
  verification: 13 focused tests passed; 45 point-guided tests passed; compileall and git diff --check passed
  remaining assumptions: Phase 2 detach/tap controls and all later phases remain unimplemented
Phase 2: COMPLETE — explicit shared-feature detach/tap and frozen/fine-tuned ablation controls; no later phase started
  status: COMPLETE
  HEAD: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1
  files changed: config.py, semantic_prior.py, test_semantic_prior.py, PLAN.md
  verification: 27 focused tests passed; 59 point-guided tests passed; compileall and git diff --check passed
  remaining assumptions: Phase 3 exact semantic-class contract and all later phases remain unimplemented
Phase 3: COMPLETE — exact ordered three-class coarse semantics; no later phase started
  status: COMPLETE
  HEAD: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1
  files changed: config.py, contracts.py, test_semantic_prior.py, test_points_refinement.py, test_frontend_forward.py, POINT_GUIDED_FRONTEND.md, PLAN.md
  verification: 32 semantic-prior focused tests passed; 9 refinement tests passed; 4 frontend smoke tests passed; 65 point-guided tests passed; compileall and git diff --check passed
  remaining assumptions: Phase 4 base-plane projection and all later phases remain unimplemented
Phase 4: COMPLETE — PLAN-locked static base tri-plane projector; no later phase started
  status: COMPLETE
  HEAD: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1
  files changed: config.py, triplane_projection.py, test_triplane_projection.py, PLAN.md
  verification: 15 focused tri-plane tests passed; 4 frontend smoke tests passed; 80 point-guided tests passed; compileall, codegraph, and git diff --check passed
  remaining assumptions: Phase 5 diagnostic frontend B composition and all later phases remain unimplemented
Phase 5: COMPLETE — shared one-pass MedicalNet composition with static typed diagnostic B planes; stopped at Research Gate A
  status: COMPLETE
  HEAD: d623d179e6626a54deefcf3f4c5a8d9e9f0a33c1 (working tree implementation)
  files changed: semantic_prior.py, model.py, contracts.py, test_frontend_forward.py, test_frontend_boundaries.py, POINT_GUIDED_FRONTEND.md, README.md, CODEBASE.md, PLAN.md
  verification: 18 frontend tests passed; 42 frontend/boundary/projector tests passed; 94 point-guided tests passed; compileall and git diff --check passed
  remaining assumptions: Research Gate A wavelet details and all Phase 6+ work remain blocked

Research Gate A — wavelet: BLOCKED
Phase 6: BLOCKED

Research Gate B — cross-plane consistency: BLOCKED
Phase 7: BLOCKED

Research Gate C — trajectory: BLOCKED
```

For every completed phase record:

```text
status:
HEAD:
files changed:
verification:
remaining assumptions:
```

---

# 17. Scope guard

Do not drift into:

- 3D Gaussian Splatting;
- Gaussian opacity/covariance;
- sparse-slice reconstruction;
- a second heavy encoder;
- full BraTS-specialist segmentation;
- T1ce conditioning;
- local per-point FFT;
- replacing the fixed spectral tri-plane with a full 3D wavelet evidence field;
- dynamic mutation of spectral anchor A;
- transformer selector/updater;
- trajectory before its research gate;
- legacy anchor/field/routing packages.

Resolve one locked research decision at a time.
