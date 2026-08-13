# AGENTS.md

## Active objective

Implement only the locked point-guided frontend for `T1/T2/FLAIR -> T1ce`
research. The target-free executable inference path reaches one shared MedicalNet traversal, coarse
semantics, deterministic initial points, bounded point refinement, point-centre
semantics, sparse semantic-aware PoU, static diagnostic base planes
`Bxy/Bxz/Byz`, the static spectral anchor `A`, typed point spectral evidence
`f_spec`, bounded Gate-C dynamic tri-plane trajectory, and the explicit
Gate-D final-Z-only decoder. Generic `forward()` remains fail-closed; Gate-E
is a separate target-after-inference objective, not an inference/training-loop
endpoint.

Repository authority is deliberately staged:

- Phases 1-7 are **IMPLEMENTED**.
- Gate A is **CLOSED / LOCKED**; Phase 6 is **COMPLETE**.
- Gate B is **CLOSED / LOCKED**; Phase 7 is **COMPLETE**.
- Gate C C1–C7 is **COMPLETE**: only the bounded dynamic tri-plane,
  reward-cost, selection, update, write-back, and diagnostic trajectory scope
  is implemented.
- Gate D D1 is **COMPLETE**: a chunked geometry-aware implicit decoder reads
  only final dynamic `Z` through the explicit reconstruction API.
- Gate E E1–E9 is **COMPLETE**: target T1ce enters only after the target-free
  inference context inside a bounded typed objective. Gate F F1/F2 are
  complete and F3/F4 software is ready for server execution; no tiny-overfit
  or full-train result is claimed yet. Gate G G1-G4 software is complete;
  trained-checkpoint and held-out evaluation evidence remains pending server
  execution.
  Gate H is default deny and has no locked local plan.

Completion of a phase does not authorize later work. A task must explicitly
name its active gate. Server-ready software does not count as a Gate F
experiment; Gate G trained-checkpoint/held-out evidence still requires server
execution.
`B` is a feature-only base projection. It is not the
spectral anchor `A`, point spectral evidence `f_spec`, a dynamic tri-plane,
or a reconstruction path.

## Scope and navigation

- Modify files only inside this repository. Do not delete, edit, or install
  into paths outside it.
- Start each task with `python scripts/codegraph.py --task <task-name>`.
  Read only that task's entrypoints and declared read paths unless a focused
  import or failing test requires expansion.
- Keep `src/smagm/features/point_guided/` additive. Do not replace the legacy
  E0/E1/E2 encoder or alter legacy anchor/field/training systems while working
  on this frontend.
- The existing BraTS21 implementation is a sparse-plane protocol. It is a
  reference for provenance and affine semantics, not a volume-loader API.
- Authorization of a phase never authorizes automatic progression into a later
  phase. `AUTHORIZED - NOT IMPLEMENTED` is never equivalent to done.

## Scientific contracts

- Input is `[B, 3, D, H, W]` ordered `(T1, T2, FLAIR)`; physical points are
  canonical RAS `XYZ` in millimetres and volume indices are `[d, h, w]`.
- Keep voxel spacing explicit; default `(1.0, 1.0, 1.0)` is not a hard-coded
  assumption.
- The MedicalNet ResNet10 prior is frozen by default, has only a minimal
  semantic head, and never downloads or pretends to load weights.
- The Phase 3 production semantic contract is exactly `(normal brain, edema,
  tumor-core candidate)` and is implemented.
- Initial points are deterministic, quasi-uniform, mask/geometry dependent,
  and value independent. Refined displacement is always relative to the
  original point and bounded by `2 mm`; supports are fixed `4 mm` spheres.
- PoU uses compact local neighborhoods only, exact L1 semantic compatibility,
  the locked quadratic spatial kernel, and explicit empty-support failure.
- No learned radius, covariance, Gaussian, point topology, U-Net, attention,
  transformer, local FFT, target-pixel access, or inference reconstruction loss may be
  introduced. The completed Gate-C trajectory remains limited to its locked
  bounded reward-cost routing scope.
- T1ce, ground truth, segmentation labels, and any target-derived value are
  forbidden from every frontend computation: input, semantic prior, point
  initialization/refinement, support construction, base-plane projection,
  spectral anchor construction, spectral query, geometry handling, routing,
  and stopping. An optional mask is legal only with non-target-derived
  provenance; this policy does not create a data adapter. T1ce may be used only
  after prediction in the completed Gate-E supervision objective only. It remains
  forbidden from all inference APIs, route decisions, and Gate-F/G work.
- Phases 1-7 may use one shared MedicalNet forward only. The base-plane branch
  may apply only the PLAN-locked axis-conditioned collapse to its existing
  `[B, C, D, H, W]` shallow feature and produce documented `XY`, `XZ`, and
  `YZ` base planes. It may not introduce a second encoder, FFT/DCT alternative,
  learned 3-D support, or a new coordinate convention. RAS `XYZ` millimetres,
  `[d, h, w]`, spacing, and affine semantics remain unchanged.

## Phase 6 implementation: fixed spectral anchor A

Phase 6 implements only a fixed two-level, 2-D stationary/undecimated Haar
transform of each static base plane and the static spectral anchor `A`. The
MAIN contract is fixed normalized Haar filters `L = [1, 1] / sqrt(2)` and
`H = [1, -1] / sqrt(2)`, reflect padding, stride one without downsampling,
and the exact stored band order:

```text
LL2, LH1, HL1, HH1, LH2, HL2, HH2
```

`LL1` is intermediate only. For a plane `[B, C, H, W]`, every stored band
retains `[B, C, H, W]`. A single shared `Conv2d(64, 8, 1)` may be applied
identically to every band and plane, yielding 56 channels per plane in that
same order: `Axy [B,56,H,W]`, `Axz [B,56,D,W]`, and `Ayz [B,56,D,H]`.
MAIN normalization is none; the only retained optional ablation is
`band_gn = GroupNorm(7, 56)`. Conv2d bias is an implementation-detail choice
unless PLAN later locks parameter count or bias behavior explicitly.

Band names are fixed by tensor axes, not an external-library convention. For
`[B, C, H, W]`, the first filter symbol addresses H/the row axis and the
second addresses W/the column axis: `LL = low H, low W`, `LH = low H, high
W`, `HL = high H, low W`, and `HH = high H, high W`. Thus XY maps row/column
to Y/X, XZ to Z/X, and YZ to Z/Y. PLAN wins if it explicitly changes this
convention.

Reflect is the only MAIN padding. The Phase 6 implementation must fail closed with a
clear `ValueError` or typed failure when a required plane dimension is one;
it must not silently substitute zero, replicate, or circular padding.
Implementation remains PyTorch-only: fixed buffers/constants plus grouped
`Conv2d` or equivalent PyTorch operations. `pywt`, PyWavelets,
`pytorch_wavelets`, and `kymatio` remain prohibited. Fixed Haar filters are
non-trainable buffers/constants; only the shared band projector and existing
base-plane scorers may evolve in the model state. Learned Haar filters,
independent band/plane projectors, external hidden spectral networks, and
`torch.fft` remain forbidden.

## Phase 7 implementation: geometry-aware point spectral evidence

Phase 7 implements only deterministic feature-grid geometry bookkeeping and
point-level spectral evidence from the implemented `A`. The helper derives
shallow/B/A grid shape, affine/centre mapping, RAS-mm-to-feature-grid
coordinates, grid-sample coordinates, and plane coordinates from
`VolumeGeometry` plus the actual convolution/pooling spatial transform. It
supports rotation, shear, anisotropic spacing, and translation; hard-coded
`coordinate / 2` is forbidden. Learned registration, learned coordinate
transforms, deformation fields, target-derived coordinate correction, and
semantic-derived coordinate warps are forbidden.

For refined `p_i*` in RAS-mm, Phase 7 bilinearly queries `Axy(x,y)`,
`Axz(x,z)`, and `Ayz(y,z)` to obtain `f_xy`, `f_xz`, and `f_yz`, each 56-d.
It derives only the deterministic descriptor
`q = concat([LL2, E1, E2])` (24-d), where
`E1 = sqrt(LH1^2 + HL1^2 + HH1^2 + eps)` and
`E2 = sqrt(LH2^2 + HL2^2 + HH2^2 + eps)`. Three pairwise cosine agreements,
mean agreement per plane, and softmax reliability weight the raw plane
features. The only MAIN packing is
`concat([alpha_xy * f_xy, alpha_xz * f_xz, alpha_yz * f_yz])`, exactly 168-d
with XY, XZ, then YZ provenance. No nearest-neighbour/patch/sphere query,
second encoder, transformer, cross-attention, learned confidence MLP, hard
plane drop, 104-d canonical fusion, or learned `168->64` compression is
implemented.

## Completed Gate C: adaptive reward–cost trajectory (C1–C7 only)

The dedicated post-Phase-7 authorities are `PLAN_GATE_C_D_E.md` and
`PLAN_GATE_F_G.md`; they supersede the historical root-PLAN statement that
Gate C is complete with only: C1 shared `B -> Z0` state initialization; C2
parameter-free state queries plus the locked RewardNet; C3 explicit
travel/overlap/step costs and utility; C4 adaptive solver; C5 shared UpdateNet;
C6 compact physical 4-mm write-back; and C7 trajectory composition. It reuses
fixed Phase-7 `q`, reliability, and `f_spec` exactly once per integrated
frontend call. No target-derived data may enter any Gate-C module.

## Completed Gate D: lightweight implicit tri-plane decoder (D1 only)

Gate D D1 decodes only the final typed Gate-C `DynamicTriPlanes` through
the shared geometry-aware XY/XZ/YZ query and the locked `96 -> 64 -> 32 -> 1`
SiLU MLP. It returns an absolute prediction only through the narrow explicit
decoder API, in bounded chunks, and must not add an observation/B/A/spectral
bypass, positional encoding, target input, loss, or training behavior.
`forward_frontend`, `forward_trajectory`, and generic `forward` retain their
existing meanings.

## Completed Gate E: reconstruction and trajectory supervision (E1–E9 only)

Gate E may receive a typed target T1ce only after the target-free frontend,
Gate-C trajectory, and Gate-D prediction are already computed. It owns the
reconstruction objective, measured counterfactual RewardNet supervision,
sampled local/spill evaluation, local-step supervision, monotonic hinge,
update magnitude regularization, and their typed total objective. It must
reuse the one decoder, UpdateNet, and 4-mm writeback; it must not alter
target-free inference, RewardNet inputs, route utility, or Gate-C stopping.

Gate F F1/F2 baseline readiness is complete under its dedicated
`baseline_training` ownership. The additive server pipeline makes F3/F4
software ready without claiming that either experiment has run. Gate G G1-G4
software is complete under the server pipeline and target-free inference
ownership; experimental execution, a trained checkpoint claim, held-out
evaluation, and Gate H remain pending/default-deny.
Gate-G diagnostics distinguish actual dense RewardNet score count
(`candidate_evaluations`) from pre-mask exact-no-revisit eligibility
(`eligible_candidate_evaluations`); unavailable candidates are still densely
queried/scored by the current implementation.
Legacy `anchors`, `fields`, `memory`,
`routing`, reconstruction, training, evaluation, CLI, and data systems remain
outside the point-guided trajectory; the word "anchor" does not authorize
their reuse.

### Gate-F F1 trainable-set clarification

`point_refiner.offset_predictor` is a **MAIN Gate-F trainable** with 1,419
parameters and is a baseline-optimizer member. This is optimizer ownership
only: its existing architecture, observation-only inputs, deterministic point
initialization, and the hard displacement bound of at most 2 mm are unchanged.
T1ce remains target-after-inference only; reconstruction gradients may reach
the existing predictor but target-derived inputs may not.

## Verification

Add small CPU synthetic tests under `tests/features/point_guided/`. Run the
smallest affected test file first, then the frontend smoke test, `compileall`,
and `git diff --check`. A pass is software evidence only, not reconstruction,
clinical, or novelty validation.
