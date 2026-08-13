# PLAN_GATE_F_G.md — Baseline Training and Inference

## /goal

Implement and run the first full baseline of the locked point-guided MRI reconstruction system, then debug only from observed failures.

Do not introduce extra curriculum, alternating optimization, beam search, RL, Flow Matching, or other training machinery unless the baseline exposes a concrete failure that requires it.

Codex execution contract:

1. Read root `PLAN.md`, then `PLAN_GATE_C_D_E.md`, then this file.
2. Resolve actual repository HEAD and preserve already-completed work.
3. Implement the full locked model first.
4. Run smoke/unit checks.
5. Run a tiny overfit sanity test.
6. Run the first full train/val/test baseline.
7. Report real failures and metrics before changing the scientific design.
8. Do not silently replace locked components with simpler alternatives.
9. T1ce is target/supervision only and never an inference observation input.

---

# 1. Locked baseline scope

The first baseline is the complete currently locked pipeline:

```text
T1 / T2 / FLAIR
        ↓
shared MedicalNet ResNet10
        ↓
S_coarse + refined points P*
        ↓
axis-conditioned base tri-plane B
        ↓
2-level SWT-Haar spectral anchor A
        ↓
point spectral query + cross-plane reliability
        ↓
f_i^spec
        ↓
Z0 initializer
        ↓
Reward–Cost trajectory
        │
        ├─ RewardNet
        ├─ travel / overlap / step cost
        ├─ greedy point selection
        ├─ UpdateNet
        └─ compact local write-back
        ↓
ZK
        ↓
implicit tri-plane decoder
        ↓
T1ce
```

MAIN frozen/fixed components:

```text
MedicalNet backbone   frozen
SWT-Haar filters      fixed
```

MAIN trainable components:

```text
semantic head if not already pretrained/locked
axis-conditioned projector
spectral shared 1×1 projection
Z0 initializer
RewardNet
UpdateNet
implicit decoder
```

Do not add a second heavy encoder.

---

# 2. Gate F — Baseline training strategy

## F1 — Implement full model first

**Status: UNBLOCKED**

Before training, ensure the full forward path exists from T1/T2/FLAIR to predicted T1ce.

Required checks:

- [ ] exact tensor shapes match earlier plans;
- [ ] no T1ce leakage into observation path;
- [ ] MedicalNet remains frozen in MAIN;
- [ ] SWT filters are fixed;
- [ ] spectral anchor A is fixed across trajectory steps;
- [ ] refined points and support radius remain unchanged;
- [ ] RewardNet produces scalar reward per candidate point;
- [ ] cost remains explicit and separate from RewardNet;
- [ ] UpdateNet produces local tri-plane corrections;
- [ ] decoder reads only ZK and outputs T1ce.

Do not begin full training until this forward path passes smoke tests.

---

## F2 — Smoke / unit test

**Status: UNBLOCKED AFTER F1**

Run one or a few synthetic/real mini-batches and verify:

- [ ] forward completes;
- [ ] backward completes;
- [ ] no NaN/Inf;
- [ ] frozen modules receive no gradients;
- [ ] trainable modules do receive gradients;
- [ ] reward, utility, selected point, update magnitude, and stop logic are finite;
- [ ] trajectory cannot revisit an exact selected point in MAIN;
- [ ] local write-back remains bounded inside configured support;
- [ ] decoder output shape is `[B,1,D,H,W]`.

This is an engineering gate only, not a quality claim.

---

## F3 — Tiny overfit sanity test

**Status: UNBLOCKED AFTER F2**

Use 1–2 training subjects only.

Goal:

```text
The complete baseline must clearly reduce training reconstruction loss.
```

Track at minimum:

```text
reconstruction loss
PSNR
SSIM
MAE
mean update magnitude
mean predicted reward
trajectory length K_used
```

Pass condition:

- reconstruction loss decreases substantially;
- predictions become visibly closer to target;
- no reward/update collapse;
- no trajectory loop;
- selected points are not all identical across iterations/subjects.

If this fails, debug before any full run.

---

## F4 — Full baseline train / validation

**Status: UNBLOCKED AFTER F3**

Train the complete baseline end-to-end using the locked Gate E objective:

```text
L = L_rec
  + λ_local L_local
  + λ_R L_reward
  + λ_mono L_mono
  + λ_delta L_delta
```

with:

```text
L_rec = Charbonnier + 0.2 * SSIM + 0.1 * gradient
```

Reward supervision remains the measured counterfactual improvement defined in Gate E.

Initial loss weights from Gate E are starting values only and may be tuned on validation, never on test.

Validation is used for:

```text
checkpoint selection
loss-weight tuning
reward-cost λ tuning
K_max tuning if needed
basic early stopping
```

Do not add extra training stages unless a concrete failure appears.

---

# 3. Gate G — Baseline inference policy

## G1 — Deterministic greedy trajectory

**Status: UNBLOCKED**

Initialize:

```text
Z = Z0
visited = empty set
```

At each step compute predicted reward for available points and explicit utility:

```text
U_i = R_i
    - λ_d * C_travel(i)
    - λ_o * C_overlap(i)
    - λ_s
```

First step:

```text
travel cost = 0
overlap cost = 0
```

Select:

```text
i* = argmax_i U_i
```

MAIN inference is hard and deterministic.

No soft selection at inference.

---

## G2 — Update rule

For the selected point:

```text
query current Z
+ full f_i^spec
+ semantic π_i
+ reliability α_i
        ↓
UpdateNet
        ↓
ΔZ_xy / ΔZ_xz / ΔZ_yz
        ↓
compact 4 mm local write-back
        ↓
Z ← Z + ΔZ
```

Then mark the exact selected point as visited and recompute reward/utility for the next step.

MAIN:

```text
1 point per step
no exact-point revisit
```

---

## G3 — Stopping

Stop when either:

```text
max_i U_i <= 0
```

or:

```text
K_used == K_max
```

Initial MAIN operational value:

```text
K_max = 64
```

Ablation / validation candidates:

```text
32
64
128
```

`K_max=64` is not a scientific constant.

Initial validation-tunable routing costs:

```text
λ_d = 0.05
λ_o = 0.20
λ_s = 0.05
```

These values must be tuned only on validation if changed.

---

## G4 — Final decode

Do not decode the full T1ce volume at every trajectory step.

After trajectory stops:

```text
ZK
 ↓
implicit tri-plane decoder
 ↓
full T1ce volume
```

Full-volume decode happens once at the end.

---

# 4. Full baseline test protocol

After training and checkpoint selection, freeze everything and evaluate the full pipeline on the held-out test set.

Report at minimum:

```text
PSNR
SSIM
MAE
NMSE if already supported
```

Also log trajectory statistics per subject:

```text
K_used
stop reason
path length
mean / max reward
mean / max utility
mean update magnitude
number of candidate points evaluated
```

Save qualitative outputs for at least representative easy, medium, and failure cases.

No test-set hyperparameter tuning.

---

# 5. Debug policy after first baseline

Only add complexity in response to observed failure.

Examples:

```text
If updater cannot improve local regions:
    debug UpdateNet / write-back / local supervision.

If RewardNet ranking is poor:
    debug counterfactual reward calibration.

If route loops or clusters too tightly:
    inspect overlap/travel cost.

If trajectory always hits K_max:
    inspect reward calibration / λ_s / stopping.

If trajectory stops immediately:
    inspect reward scale / cost scale.

If decoder reconstructs poorly even when Z improves:
    debug decoder capacity / coordinate mapping.

If training is unstable:
    only then consider staged warm-up or alternating optimization.
```

Do not introduce curriculum, RL, beam search, Flow Matching, or a more complex optimizer without evidence from the baseline.

---

# 6. Verification commands

Use repository-native commands where available. At minimum:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided
PYTHONPATH=src .venv/bin/python -m compileall -q src/smagm/features/point_guided
git diff --check
```

For training code, add a dedicated smoke command and a tiny-overfit command that can be rerun reproducibly.

The exact CLI names may follow existing repository conventions, but the capabilities above are required.

---

# 7. Completion log

```text
F1 full model forward: COMPLETE — current target-free context and explicit APIs
F2 smoke/unit test: COMPLETE — synthetic forward/objective/backward/update
F3 tiny overfit: SOFTWARE READY — NOT YET EXECUTED ON SERVER
F4 full train/val baseline: SOFTWARE READY — NOT YET EXECUTED ON SERVER

G1 deterministic route: COMPLETE — target-free hard baseline inference
G2 update loop: COMPLETE — shared Gate-C UpdateNet/write-back path
G3 stopping: COMPLETE — exact no-revisit and bounded stopping diagnostics
G4 final decode: COMPLETE — strict final-Z-only decoder path

Full held-out test: PENDING SERVER EXPERIMENT
```

For each completed item record:

```text
status:
HEAD:
config/checkpoint:
command:
result:
failures:
next action:
```

---

# 8. Baseline success criterion

The first goal is not SOTA.

The first goal is a scientifically valid, fully runnable baseline where:

1. T1/T2/FLAIR → T1ce works end-to-end;
2. the trajectory executes deterministically;
3. reward and explicit routing costs affect point order;
4. selected updates modify Z locally as designed;
5. reconstruction metrics can be measured on train/val/test;
6. failures are observable and debuggable.

Only after this baseline is reproducible should training/routing complexity be increased.
