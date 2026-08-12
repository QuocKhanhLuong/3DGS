# PLAN_GATE_C_D_E.md — Reward–Cost Trajectory, Decoder, and Training

## /goal

Implement the next locked stages of the CVPR 2027 point-guided MRI reconstruction system after Gate A/B:

```text
Gate C — adaptive reward–cost trajectory over refined points
Gate D — lightweight implicit tri-plane decoder
Gate E — reconstruction, reward, and trajectory supervision
```

This file is an implementation specification, not a license to redesign the model.

Codex execution contract:

1. Read root `PLAN.md` first, then this file.
2. Resolve actual `main` HEAD and current repository state before editing.
3. Preserve any implementation already completed for Gate A/B and earlier point-guided stages.
4. Execute all phases in this file that are marked **UNBLOCKED** in order without asking for confirmation between phases.
5. For each phase: inspect the smallest relevant codegraph scope, implement the exact locked design, add focused tests, run verification, fix failures, and update the completion log.
6. Do not silently replace a locked method with a simpler alternative.
7. Hyperparameter examples marked **TUNABLE** are not scientific constants.
8. Do not introduce Flow Matching, Optimal Transport, RL/policy gradients, transformers, or a second heavy encoder in this gate.
9. Keep T1ce as training target only; T1ce must never enter inference observation input.
10. Stop after Gate E implementation and verification unless a later gate has been explicitly unlocked elsewhere.

---

# 0. Dependency context

This plan assumes the earlier pipeline has already established, or is implementing according to root `PLAN.md`:

```text
T1 / T2 / FLAIR
        │
        ▼
shared MedicalNet ResNet10
        │
post Conv1 + BN + ReLU, pre-MaxPool
        │
        ├──────── semantic branch ───────→ S_coarse
        │
        └──────── spectral branch
                     │
                F_shallow.detach()
                     │
          axis-conditioned projection
                     │
              Bxy / Bxz / Byz
                     │
             2-level SWT-Haar
                     │
        7 bands + shared 1×1 projection
                     │
               Axy/Axz/Ayz
              56 channels/plane
                     │
             point spectral query
                     │
      f_i^spec ∈ R^168 + reliability α_i
```

Refined points already carry at least:

```text
p_i*       refined 3D point center
π_i        3-class semantic probability
f_i^spec   168-d reliability-weighted spectral evidence
α_i        [α_xy, α_xz, α_yz]
```

The point system remains:

```text
MAIN point count = 2048
ablation = 3072
support radius r = 4 mm
max displacement = 2 mm from original center
sparse compact-support PoU semantics preserved
```

Do not rebuild these components in this file.

---

# 1. Gate C overview — Reward–Cost Trajectory Optimization

## LOCKED conceptual objective

The trajectory is not a free-form learned sequence and not Flow Matching.

It is an adaptive point-routing optimization process over the refined point set.

Let:

```text
τ = (i_1, i_2, ..., i_K)
```

be the ordered sequence of selected points.

The trajectory should prefer points with high expected reconstruction gain while penalizing travel, redundant local coverage, and unnecessary updates.

Conceptual objective:

```text
maximize cumulative reward
minus travel cost
minus overlap/redundancy cost
minus per-update cost
```

At each step `t`, recompute reward from the current dynamic state `Z_t`.

The route is therefore adaptive rather than precomputed once.

---

# 2. Gate C1 — Dynamic state initialization Z0

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Create the initial dynamic reconstruction tri-plane from the already available base observation tri-plane `B`, without introducing a second encoder or contaminating `Z0` with dense semantic/spectral conditioning.

## Locked meaning

```text
Z0 = latent reconstruction state
```

`Z0` is **not** an initial T1ce image estimate.

`B` remains an observation-derived base tri-plane and is not mutated by the trajectory.

`A` remains a fixed spectral evidence anchor and is not fused into `Z0`.

## Locked construction

For each plane `p ∈ {xy,xz,yz}`:

```text
B_p: 64 channels
  ↓
shared Conv2d 1×1
64 → 32
  ↓
Z_{p,0}: 32 channels
```

Use one shared `1×1 Conv2d` across all three orientations.

Shapes:

```text
Zxy_0 = [B,32,Hs,Ws]
Zxz_0 = [B,32,Ds,Ws]
Zyz_0 = [B,32,Ds,Hs]
```

Spatial grids must match the corresponding `B` and `A` planes.

## Locked exclusions

Do not:

- concatenate dense `S_coarse` into `Z0`;
- concatenate spectral anchor `A` into `Z0`;
- use 3×3/5×5 spatial encoder stacks here;
- add a second MedicalNet or U-Net;
- treat `Z0` as T1ce prediction;
- mutate `B` during trajectory.

## Gradient contract

MAIN frozen-backbone path remains:

```text
loss
 ↓
state initializer
 ↓
B
 ↓
axis projector
 X
F_shallow.detach()
 X
MedicalNet
```

The `Z0` initializer and axis projector remain trainable.

## TUNABLE ablation only

```text
state channels:
  16
  32   MAIN
  64
```

## Required tests

- exact three-plane shapes;
- initializer shared across planes;
- no `A` or `S_coarse` dependency in `Z0` construction;
- `B` tensor is not mutated;
- gradients reach state initializer;
- frozen MedicalNet remains gradient-isolated in MAIN.

---

# 3. Gate C2 — Dynamic reward R_i^t

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Estimate, for every candidate point at the current state, how useful an update at that point is expected to be.

Reward semantics are locked:

```text
R_i^t ≈ expected reconstruction improvement if point i is updated now
```

Reward must depend on the current state `Z_t`; it must not be static semantic/spectral saliency.

## Inputs

### Current dynamic-state query

At refined point `p_i*`, bilinearly query:

```text
Zxy_t(x,y) → 32-d
Zxz_t(x,z) → 32-d
Zyz_t(y,z) → 32-d
```

Concatenate with fixed block order:

```text
z_i^t = [z_xy | z_xz | z_yz] ∈ R^96
```

### Compact spectral selector descriptor

Reuse Gate B deterministic consistency descriptors:

```text
q_xy, q_xz, q_yz ∈ R^24
```

and reliability weights:

```text
α_i = [α_xy, α_xz, α_yz] ∈ R^3
```

Because `q` is orientation-insensitive, use reliability-weighted average:

```text
q_bar_i = α_xy q_xy + α_xz q_xz + α_yz q_yz
q_bar_i ∈ R^24
```

### Semantic input

```text
π_i ∈ R^3
```

No explicit XYZ coordinates in RewardNet MAIN.

## RewardNet

Locked MAIN architecture:

```text
input = [z_i^t, π_i, q_bar_i, α_i]
      = 96 + 3 + 24 + 3
      = 126-d

RewardNet:
126 → 64 → 1 → sigmoid
```

Output:

```text
R_i^t ∈ [0,1]
```

Use one shared RewardNet for all points and all trajectory steps.

## Locked exclusions

Do not use:

- full `f_i^spec=168` in RewardNet MAIN;
- explicit pathology hand-weights;
- explicit XYZ positional coordinates;
- transformer/cross-attention;
- route cost inside RewardNet;
- a separate learned selector score unrelated to reconstruction gain.

Full `f_i^spec` is reserved for the updater.

## Required tests

- reward output shape `[B,N]` or equivalent sparse batched form;
- reward range `[0,1]`;
- reward changes when `Z_t` changes with fixed point evidence;
- no explicit XYZ input;
- no route cost input;
- shared RewardNet parameters across points.

---

# 4. Gate C3 — Explicit routing cost

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Keep travel/redundancy/update expense explicit and interpretable rather than hiding them inside RewardNet.

## C3.1 Travel cost

If the previously selected point is `j`, define:

```text
C_travel(j,i) = ||p_j* - p_i*||_2 / r
```

with locked support radius:

```text
r = 4 mm
```

Use physical-space distance, not voxel-index distance.

For the first route step, use a deterministic initialization policy defined by implementation contract; do not invent a learned start token. A safe default is zero travel cost for the first selection.

## C3.2 Overlap/redundancy cost

Reuse the already meaningful point support geometry.

Locked compact form:

```text
C_overlap(i,H_t)
  = max over visited j of
    (1 - d_ij/(2r))_+^2
```

where:

```text
d_ij = ||p_i* - p_j*||_2
r = 4 mm
```

The overlap term is zero when supports cannot overlap (`d >= 2r`).

Use visited/update history only for explicit overlap accounting; do not add an unrelated recurrent history network.

## C3.3 Per-update step cost

Include a constant positive per-step cost:

```text
C_step = 1
```

scaled by a tunable coefficient.

Purpose: prevent the optimizer from visiting all points simply because tiny positive rewards remain.

## Utility

Locked utility:

```text
U_i^t
 = R_i^t
 - λ_d C_travel(j,i)
 - λ_o C_overlap(i,H_t)
 - λ_s
```

Keep meanings separate:

```text
Reward = expected reconstruction gain
Cost   = explicit routing/update expense
Utility = Reward - Cost
```

## TUNABLE hyperparameters

```text
λ_d > 0
λ_o > 0
λ_s > 0
```

Do not hardcode scientifically privileged values without experiment config.

## Required tests

- physical distance normalization by `r`;
- cost zero/positive behavior at expected distances;
- overlap becomes zero at `d >= 2r`;
- exact utility decomposition;
- RewardNet output is unchanged when only routing hyperparameters change.

---

# 5. Gate C4 — Adaptive route solver

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Choose the next point from current utility, apply one update, recompute state-dependent rewards, and continue.

## Locked MAIN solver

Use adaptive one-step receding-horizon routing:

```text
i_t = argmax_i U_i^t
```

After applying the selected update:

```text
Z_t → Z_{t+1}
```

recompute rewards for the next step:

```text
R_i^{t+1} = RewardNet(Z_{t+1}, point evidence)
```

Do not precompute a full route from `Z0` and keep it fixed.

## Stop rule

Primary learned/economic stopping condition:

```text
max_i U_i^t <= 0  → STOP
```

Also require a safety cap:

```text
t < K_max
```

`K_max` is **TUNABLE** and not yet scientifically fixed.

## Training selection

Inference:

```text
hard argmax
```

Training MAIN:

```text
straight-through soft selection
```

Use softmax utility weights for backward surrogate:

```text
w_i = softmax(U_i / temperature)
```

while forward behavior remains effectively hard selection.

The exact straight-through implementation must be unit-tested for forward one-hot/hard behavior and nonzero backward gradients.

## TUNABLE / ablation

MAIN:

```text
adaptive greedy 1-step
```

Later ablations only:

```text
lookahead depth 3
beam search
```

Do not implement beam search unless explicitly requested by an experiment phase.

## Locked exclusions

Do not add:

- RL policy gradient;
- Q-learning;
- PPO;
- Gumbel routing as MAIN;
- Flow Matching;
- global TSP/orienteering solve over all 2048 points at t=0;
- static ranking independent of `Z_t`.

## Required tests

- hard inference argmax correctness;
- stop when maximum utility <= 0;
- safety cap honored;
- reward recomputed after a state change;
- selected route can change after update;
- straight-through surrogate propagates gradient during training.

---

# 6. Gate C5 — Local updater

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Given the selected point, compute how each dynamic plane should change using full local state, full spectral evidence, semantic identity, and cross-plane reliability.

## Updater input

At selected point `i_t`:

```text
z_i^t       = 96-d current dynamic-state query
f_i^spec    = 168-d full Gate B spectral evidence
π_i         = 3-d semantic probability
α_i         = 3-d cross-plane reliability
```

Concatenate:

```text
x_i^upd ∈ R^270
```

## Locked MAIN UpdateNet

```text
270 → 128 → 96
```

Output packing:

```text
Δ_i^t = [Δ_xy | Δ_xz | Δ_yz]

Δ_xy ∈ R^32
Δ_xz ∈ R^32
Δ_yz ∈ R^32
```

Use one shared updater across points and trajectory steps.

Do not predict a scalar T1ce intensity here.

## Update scale

Use a bounded/small update parameterization or explicit scale factor so one step cannot arbitrarily overwrite a plane.

Exact numeric scale is **TUNABLE**.

## Required tests

- exact 270-d input contract;
- exact 96-d output packing;
- gradients reach UpdateNet;
- three 32-d plane corrections are distinct blocks;
- updater receives full `f_i^spec`, unlike RewardNet.

---

# 7. Gate C6 — Compact local write-back

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Write selected point correction to `Z_t` locally using the existing 4 mm point support geometry instead of modifying a single plane pixel or the whole plane.

## Locked kernel

Reuse compact quadratic support:

```text
k_i(x) = (1 - ||x-p_i||/r)_+^2
```

with:

```text
r = 4 mm
```

For each dynamic plane, project the point center to its corresponding 2D coordinate:

```text
XY → (x,y)
XZ → (x,z)
YZ → (y,z)
```

and apply the corresponding 32-d correction over the compact local support.

Conceptual update:

```text
Z_{p,t+1}(x)
 = Z_{p,t}(x)
 + η k_i(x) Δ_{p,i}^t
```

where `η` is **TUNABLE**.

## Multiple updates in one step

MAIN route selects one point per step.

If a later ablation batches multiple selected points, overlap must use PoU-style normalized aggregation:

```text
ΔZ(x)
 = sum_i k_i(x) Δ_i
   / (sum_i k_i(x) + eps)
```

Do not let overlapping contributions arbitrarily amplify update magnitude.

## Required tests

- support is zero outside radius;
- center receives maximal weight;
- write is local rather than full-plane;
- correct plane coordinate mapping;
- no accidental cross-plane shape swap;
- selected update changes `Z_t` but not `B` or `A`;
- optional multi-point aggregation stays normalized.

---

# 8. Gate C full operator

The locked trajectory loop is:

```text
P*, A, fixed point evidence
           │
           ▼
          Z0
           │
           ▼
   query all candidate points
           │
           ▼
     RewardNet → R_i^t
           │
      explicit costs
           │
           ▼
       U_i^t = R - C
           │
           ▼
      adaptive argmax
           │
           ▼
      selected point i_t
           │
           ▼
       UpdateNet
           │
           ▼
  compact 4 mm local write
           │
           ▼
         Z_{t+1}
           │
      recompute rewards
           │
           └──────── repeat
```

Stop when:

```text
max utility <= 0
OR
step == K_max
```

Locked scientific interpretation:

```text
points = candidate optimization sites
spectral anchor = fixed evidence
Z_t = current reconstruction state
reward = expected gain from updating a point now
cost = travel + redundancy + update expense
trajectory = adaptive reward–cost route
```

---

# 9. Gate D — Lightweight implicit tri-plane decoder

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Decode final dynamic state `Z_K` into full-resolution T1ce while keeping decoder capacity intentionally limited so reconstruction reasoning remains attributable to the trajectory.

## D1. Voxel query

For each full-resolution output voxel `(x,y,z)`:

```text
query Zxy_K at (x,y) → 32-d
query Zxz_K at (x,z) → 32-d
query Zyz_K at (y,z) → 32-d
```

Use bilinear interpolation on each 2D plane.

Concatenate in fixed order:

```text
[XY | XZ | YZ] → 96-d
```

The coordinate mapping must use geometry transforms rather than assuming a hardcoded stride division.

## D2. Decoder architecture

Locked MAIN:

```text
96 → 64 → 32 → 1
```

Suggested activation:

```text
SiLU
```

Use one shared voxel MLP for the entire volume.

Output:

```text
predicted absolute T1ce intensity
```

## D3. Absolute reconstruction MAIN

Locked MAIN:

```text
X_hat_T1ce(x) = Decoder(query(Z_K,x))
```

Do not add a direct T1 skip connection in MAIN.

A residual formulation:

```text
T1 + residual
```

is allowed only as an ablation.

## D4. Resolution

Do not expand tri-planes into a dense 3D feature volume before decoding.

Query the shallow-grid tri-planes at continuous coordinates corresponding to each full-resolution output voxel.

Decode in chunks for memory efficiency.

Chunk size is **TUNABLE/runtime-dependent**.

## Locked exclusions

Do not add:

- U-Net decoder;
- 3D ResNet decoder;
- transformer decoder;
- dense 3D feature expansion as MAIN;
- explicit XYZ positional encoding MAIN;
- direct spectral anchor bypass into final decoder MAIN;
- direct T1 image bypass MAIN.

## Required tests

- exact output shape `[B,1,D,H,W]`;
- bilinear query geometry correctness;
- fixed plane block order;
- chunked and unchunked outputs match within tolerance;
- decoder receives only `Z_K` query features in MAIN;
- no direct observation-image bypass.

---

# 10. Gate E — Reconstruction and trajectory supervision

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Train decoder/updater/state trajectory for accurate T1ce synthesis while giving RewardNet an explicit target corresponding to measured reconstruction improvement.

---

# 11. Gate E1 — Final reconstruction loss

Locked MAIN reconstruction objective:

```text
L_rec
 = L_charbonnier
 + λ_ssim L_ssim
 + λ_grad L_grad
```

Recommended initial coefficients:

```text
Charbonnier = 1.0
SSIM        = 0.2
Gradient    = 0.1
```

These are **INITIAL TUNABLE HYPERPARAMETERS**, not fixed scientific constants.

## Charbonnier

Use a numerically stable epsilon.

Conceptually:

```text
sqrt((prediction-target)^2 + eps^2)
```

## SSIM

Use structural similarity loss over valid reconstructed brain/output support.

## Gradient loss

Use image-gradient agreement to discourage oversmoothed boundaries:

```text
|∇X_hat - ∇X_gt|
```

## Locked exclusions MAIN

Do not initially add:

- GAN/adversarial loss;
- VGG perceptual loss;
- segmentation loss;
- pathology classification loss;
- extra wavelet reconstruction loss;
- cross-plane consistency loss;
- RL loss.

---

# 12. Gate E2 — Measured counterfactual reward target

**Status: LOCKED / UNBLOCKED**

## /phase-goal

Supervise RewardNet using an observable target: how much reconstruction improves if a candidate update is actually applied.

For sampled candidate point `i` at state `Z_t`:

1. compute reconstruction loss before hypothetical update;
2. apply the candidate update to form a hypothetical state;
3. compute reconstruction loss after update;
4. measure improvement;
5. detach the measured target;
6. regress RewardNet to it.

Basic normalized target:

```text
R_i* = clip(
  (L_before - L_after) / (L_before + eps),
  0,
  1
)
```

Reward prediction:

```text
R_i^t = RewardNet(...)
```

Reward loss:

```text
L_reward = SmoothL1(R_i^t, stopgrad(R_i*))
```

## Locked semantic meaning

Reward target measures reconstruction gain only.

Do not subtract travel/overlap/step cost inside `R_i*`.

Costs remain explicit at route-utility level.

---

# 13. Gate E3 — Counterfactual candidate sampling

Testing all ~2048 candidates with a hypothetical updater+decoder pass per training step is not required.

Use a sampled candidate subset during reward supervision.

Suggested initial range:

```text
M = 32 to 64 candidates
```

This is **TUNABLE**.

The subset should contain a mixture of:

- actually selected point;
- some high predicted-reward candidates;
- some random candidates.

Do not use only top-reward points, or RewardNet will receive poor calibration for low/medium values.

Inference still evaluates cheap RewardNet over the full candidate set.

---

# 14. Gate E4 — Local gain and tri-plane spill safeguard

**Status: LOCKED / UNBLOCKED**

## Motivation

A compact write on a 2D tri-plane can influence output voxels along fibers sharing the same projected plane coordinate.

Reward supervision must therefore not credit a local improvement that causes severe collateral degradation elsewhere.

## Local gain

Measure reconstruction change in the 3D local region around the point support.

```text
ΔL_local = L_local_before - L_local_after
```

## Spill/collateral penalty

Sample affected nonlocal voxels aligned with modified plane coordinates, including conceptual fibers such as:

```text
same XY, different z
same XZ, different y
same YZ, different x
```

Define positive degradation:

```text
ΔL_spill = max(0, L_spill_after - L_spill_before)
```

Measured gain:

```text
G_i^t = ΔL_local - β ΔL_spill
```

Normalize/clamp this measured gain into the `[0,1]` reward target.

`β` is **TUNABLE**.

The implementation should avoid decoding the entire volume for every counterfactual candidate; query/decode only the local and sampled collateral voxels needed for the estimate.

---

# 15. Gate E5 — Local step supervision

**Status: LOCKED / UNBLOCKED**

Do not wait until final `Z_K` for all updater gradients.

After selected update:

```text
Z_t → Z_{t+1}
```

compute reconstruction loss over the affected/local region.

Trajectory local loss:

```text
L_local
 = mean_t L_rec_local(D(Z_{t+1}), X_gt)
```

This gives UpdateNet direct supervision at each trajectory step.

---

# 16. Gate E6 — Monotonic improvement penalty

**Status: LOCKED / UNBLOCKED**

Because the mechanism is explicitly an optimization trajectory, penalize steps that worsen reconstruction.

Use a hinge:

```text
L_mono
 = mean_t max(0, ell_{t+1} - ell_t)
```

where `ell_t` is a consistent local/trajectory reconstruction error measurement.

Do not require mathematically exact monotonicity as a hard constraint; use a soft penalty.

---

# 17. Gate E7 — Update magnitude regularization

**Status: LOCKED / UNBLOCKED**

Penalize unnecessarily large corrections:

```text
L_delta
 = mean_t ||Δ_i^t||_2^2
```

Use a small coefficient.

Purpose: lightweight trust-region behavior and trajectory stability.

---

# 18. Gate E8 — Total loss

Locked structure:

```text
L_total
 = L_rec
 + λ_local L_local
 + λ_R L_reward
 + λ_M L_mono
 + λ_delta L_delta
```

Recommended starting values only:

```text
λ_local = 0.5
λ_R     = 1.0
λ_M     = 0.2
λ_delta = 1e-4
```

All four are **TUNABLE HYPERPARAMETERS**.

Do not encode these values as immutable architecture constants.

---

# 19. Gate E9 — Selection gradient contract

Inference selection is hard:

```text
i_t = argmax U_i^t
```

Training MAIN uses a straight-through softmax surrogate.

The route solver itself has no separate learned parameters.

RewardNet receives direct `L_reward` supervision.

Updater/state/decoder receive reconstruction-driven gradients through the selected/hypothetical paths.

Do not introduce policy-gradient training unless a future explicit experiment unlocks it.

---

# 20. Recommended implementation phases

## Phase C0 — Resolve actual repository state

**Status: UNBLOCKED**

Tasks:

- [ ] read root `PLAN.md`;
- [ ] resolve current HEAD;
- [ ] inspect point-guided codegraph scope;
- [ ] detect which Gate A/B phases are already implemented;
- [ ] preserve any newer user code;
- [ ] identify exact internal contracts for `B`, `A`, `P*`, `f_spec`, `α`;
- [ ] do not reimplement completed work.

Verify:

```bash
git diff --check
```

---

## Phase C1 — Dynamic tri-plane state initializer

**Status: UNBLOCKED**

Implement:

```text
B → shared 1×1 Conv2d 64→32 → Z0
```

Add focused tests from Section 2.

---

## Phase C2 — Dynamic-state point query + RewardNet

**Status: UNBLOCKED**

Implement:

- [ ] bilinear `Z_t` query at all refined points;
- [ ] `q_bar` construction;
- [ ] exact 126-d reward descriptor;
- [ ] shared `126→64→1→sigmoid` RewardNet;
- [ ] typed reward outputs.

Do not implement routing cost inside RewardNet.

---

## Phase C3 — Routing costs + utility

**Status: UNBLOCKED**

Implement:

- [ ] normalized physical travel cost;
- [ ] compact overlap cost;
- [ ] per-step cost;
- [ ] exact utility decomposition;
- [ ] configuration for `λ_d, λ_o, λ_s`.

---

## Phase C4 — Adaptive route solver

**Status: UNBLOCKED**

Implement:

- [ ] hard argmax inference;
- [ ] straight-through softmax training;
- [ ] utility<=0 stopping;
- [ ] `K_max` safety cap config;
- [ ] visited-point history only as needed for overlap;
- [ ] reward recomputation after every applied update.

Do not implement global route search.

---

## Phase C5 — UpdateNet

**Status: UNBLOCKED**

Implement exact MAIN:

```text
270 → 128 → 96
```

with fixed output packing:

```text
32 XY | 32 XZ | 32 YZ
```

---

## Phase C6 — Compact local write-back

**Status: UNBLOCKED**

Implement 4 mm compact quadratic write with correct tri-plane geometry.

Preserve `A` and `B` as immutable observation/evidence tensors.

---

## Phase C7 — End-to-end trajectory loop

**Status: UNBLOCKED**

Compose:

```text
Z0
→ reward
→ utility
→ select
→ update
→ local write
→ Z1
→ recompute
→ ...
→ ZK
```

Return diagnostics sufficient for tests/training:

```text
selected indices
route length
per-step reward
per-step travel cost
per-step overlap cost
per-step utility
per-step update norm
stop reason
```

Do not expose unnecessary dense candidate tensors as permanent public API.

---

## Phase D1 — Implicit tri-plane decoder

**Status: UNBLOCKED**

Implement:

```text
Z_K tri-plane query
→ 96-d
→ MLP 96→64→32→1
→ absolute T1ce
```

Support chunked full-volume decoding.

---

## Phase E1 — Core reconstruction losses

**Status: UNBLOCKED**

Implement:

- [ ] Charbonnier;
- [ ] SSIM;
- [ ] gradient loss;
- [ ] configurable coefficients.

---

## Phase E2 — Counterfactual reward supervision

**Status: UNBLOCKED**

Implement:

- [ ] sampled candidate subset;
- [ ] hypothetical update path;
- [ ] before/after local reconstruction evaluation;
- [ ] sampled spill/collateral evaluation;
- [ ] detached normalized reward target;
- [ ] SmoothL1 RewardNet loss.

Keep this training-only path out of inference.

---

## Phase E3 — Trajectory regularization

**Status: UNBLOCKED**

Implement:

- [ ] per-step local reconstruction loss;
- [ ] monotonic hinge loss;
- [ ] update magnitude regularization;
- [ ] total loss composition.

---

## Phase E4 — Integration verification

**Status: UNBLOCKED**

Verify all Gate C/D/E contracts together.

Required end-to-end synthetic test should demonstrate:

1. `Z0` has three 32-channel planes;
2. reward is state-dependent;
3. cost affects route choice without changing raw reward;
4. one selected point produces local `Z` modification only;
5. the next route decision can change after update;
6. stopping works on nonpositive utility;
7. decoder reconstructs correct full-resolution tensor shape;
8. reward target is detached;
9. gradients reach RewardNet through `L_reward`;
10. gradients reach UpdateNet/state initializer/decoder through reconstruction losses;
11. frozen MedicalNet remains isolated in MAIN;
12. T1ce is never passed into inference observation path.

---

# 21. Suggested modules

Use existing repository naming/style where possible rather than forcing these exact filenames.

If no equivalent files exist, reasonable focused modules are:

```text
src/smagm/features/point_guided/state_init.py
src/smagm/features/point_guided/reward.py
src/smagm/features/point_guided/trajectory_cost.py
src/smagm/features/point_guided/trajectory_solver.py
src/smagm/features/point_guided/updater.py
src/smagm/features/point_guided/writeback.py
src/smagm/features/point_guided/trajectory.py
src/smagm/features/point_guided/decoder.py
src/smagm/features/point_guided/losses.py
src/smagm/features/point_guided/reward_supervision.py
```

Tests should mirror repository conventions under:

```text
tests/features/point_guided/
```

Do not create duplicate modules if equivalent code already exists.

---

# 22. Config contract

Expose scientific/tunable controls explicitly rather than burying constants.

At minimum support:

```text
state_channels = 32                 MAIN
reward_hidden = 64                  MAIN
updater_hidden = 128                MAIN
support_radius_mm = 4.0             LOCKED

lambda_travel                        TUNABLE
lambda_overlap                       TUNABLE
lambda_step                          TUNABLE
update_scale                         TUNABLE
K_max                                TUNABLE
selection_temperature                TUNABLE

counterfactual_candidates            TUNABLE, initial 32–64
spill_weight_beta                    TUNABLE

loss_charbonnier_weight              initial 1.0
loss_ssim_weight                     initial 0.2
loss_gradient_weight                 initial 0.1
lambda_local                         initial 0.5
lambda_reward                        initial 1.0
lambda_monotonic                     initial 0.2
lambda_delta                         initial 1e-4
```

Validation must reject nonsensical negative cost/loss coefficients where positivity is required.

---

# 23. Ablation registry

Support targeted ablations without running all Cartesian combinations.

```text
Z0 channels:
  16
  32 MAIN
  64

Routing:
  reward_only
  reward_minus_travel
  reward_minus_travel_overlap
  full_reward_cost MAIN

Solver:
  static ranking baseline
  adaptive greedy MAIN
  lookahead-3 future ablation

Reward:
  static evidence baseline
  dynamic state-aware MAIN

Update:
  single-cell write baseline
  compact 4mm write MAIN

Decoder:
  absolute MAIN
  residual-to-T1 ablation

Reward supervision:
  indirect only baseline
  counterfactual measured gain MAIN

Trajectory loss:
  final-only baseline
  local+monotonic MAIN
```

Do not add unrelated ablations automatically.

---

# 24. Scientific invariants

Every Gate C/D/E implementation must preserve:

1. T1ce is target only and never inference observation input.
2. Observation channel order remains T1/T2/FLAIR.
3. `A` remains fixed spectral evidence during trajectory.
4. `B` remains unmutated base observation tri-plane.
5. `Z_t` is the only dynamic tri-plane state.
6. `Z0` comes from `B`, not from `A` or dense `S_coarse`.
7. Semantic information enters trajectory through point descriptors, not dense permanent conditioning.
8. Reward means expected reconstruction improvement only.
9. Routing costs stay explicit and separate from RewardNet.
10. Reward is state-dependent.
11. Route is recomputed adaptively after state updates.
12. Support radius remains 4 mm.
13. Local write uses compact support.
14. RewardNet uses compact spectral summary, not full `f_spec` MAIN.
15. UpdateNet uses full `f_spec`.
16. Decoder remains lightweight and reads only `Z_K` MAIN.
17. No direct T1 bypass MAIN.
18. No dense 3D decoder MAIN.
19. No Flow Matching/OT/RL introduced in this gate.
20. Frozen MedicalNet remains gradient-isolated in MAIN.

---

# 25. Scope guard

Do not drift into:

- Flow Matching;
- Optimal Transport coupling;
- reinforcement learning;
- PPO/Q-learning/policy gradients;
- TSP solver over all points;
- transformer router;
- transformer decoder;
- second MedicalNet;
- U-Net target decoder;
- 3D Gaussian Splatting;
- Gaussian opacity/covariance;
- full dense `[B,N,D,H,W]` point tensors;
- target T1ce leakage into inference;
- dense `S_coarse` trajectory conditioning;
- spectral anchor mutation;
- rebuilding Gate A/B from scratch.

---

# 26. Final verification

After all Gate C/D/E phases:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

Report at completion:

```text
HEAD
files changed
new tests
Z0 shapes
RewardNet parameter count
UpdateNet parameter count
decoder parameter count
route stop behavior
straight-through selection behavior
counterfactual reward calibration sanity check
local-write support sanity check
decoder full-volume output shape
loss values on synthetic batch
frozen/detach gradient verification
remaining tunable hyperparameters
next unresolved research gate
```

Do not claim clinical or reconstruction-quality improvement from software tests alone.

---

# 27. Phase completion log

Initialize as:

```text
C0 repository audit: PENDING
C1 Z0 initializer: PENDING
C2 RewardNet: PENDING
C3 costs/utility: PENDING
C4 adaptive solver: PENDING
C5 UpdateNet: PENDING
C6 local write: PENDING
C7 trajectory composition: PENDING

D1 implicit decoder: COMPLETE — chunked final-Z-only geometry-aware query and
shared 96->64->32->1 SiLU MLP; 15 focused tests, 198 full point-guided tests

E1 reconstruction losses: PENDING
E2 counterfactual reward supervision: PENDING
E3 trajectory regularization: PENDING
E4 integration verification: PENDING
```

For every completed phase append:

```text
status:
HEAD:
files changed:
verification:
remaining assumptions:
```

---

# 28. Locked end-state of this plan

After successful completion, the model segment covered by this file should be conceptually:

```text
T1/T2/FLAIR
     │
     ▼
shared observation representation
     │
     ├──────── fixed spectral anchor A
     │
     ├──────── refined points P*
     │
     └──────── base tri-plane B
                    │
                    ▼
          shared 1×1 state init
                    │
                    ▼
                   Z0
                    │
          ┌─────────┴─────────┐
          │ adaptive loop     │
          │                   │
          │ RewardNet         │
          │   ↓               │
          │ reward - costs    │
          │   ↓               │
          │ select point      │
          │   ↓               │
          │ UpdateNet         │
          │   ↓               │
          │ compact write     │
          │   ↓               │
          └── Z_{t+1} ────────┘
                    │
                    ▼
                   ZK
                    │
                    ▼
          implicit tri-plane decoder
                    │
                    ▼
              predicted T1ce
```

Training additionally uses GT T1ce only for reconstruction/counterfactual supervision:

```text
GT T1ce
   │
   ├─ final reconstruction loss
   ├─ local step loss
   ├─ counterfactual measured reward
   └─ monotonic trajectory supervision
```

GT T1ce has no inference path into the model.
