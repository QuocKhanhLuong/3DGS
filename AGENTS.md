# AGENTS.md

## Active objective

Implement only the locked point-guided frontend for `T1/T2/FLAIR -> T1ce`
research. The current executable frontend ends after coarse semantics,
deterministic initial points, bounded point refinement, point-centre semantics,
and a sparse semantic-aware PoU. It must never return a fake T1ce volume.

`PLAN.md` Phases 1–5 are authorized next implementation scope, but are not
implemented merely by this authorization: shared MedicalNet intermediate
features, explicit feature detach/tap controls, the three-class semantic
contract, static base planes `Bxy/Bxz/Byz`, and their diagnostic frontend
composition may be added only in that order. `B` is a feature-only base
projection, not wavelet spectral anchor `A`, cross-plane fusion, a dynamic
tri-plane, or a reconstruction path.

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
  phase. An active task must explicitly name the authorized phase, and
  `AUTHORIZED — NOT IMPLEMENTED` is never equivalent to done.

## Scientific contracts

- Input is `[B, 3, D, H, W]` ordered `(T1, T2, FLAIR)`; physical points are
  canonical RAS `XYZ` in millimetres and volume indices are `[d, h, w]`.
- Keep voxel spacing explicit; default `(1.0, 1.0, 1.0)` is not a hard-coded
  assumption.
- The MedicalNet ResNet10 prior is frozen by default, has only a minimal
  semantic head, and never downloads or pretends to load weights.
- The Phase 3 production semantic contract is exactly `(normal brain, edema,
  tumor-core candidate)`. Until Phase 3 code is completed, this is an approved
  pending implementation contract rather than a claim about the current
  configurable class count.
- Initial points are deterministic, quasi-uniform, mask/geometry dependent,
  and value independent. Refined displacement is always relative to the
  original point and bounded by `2 mm`; supports are fixed `4 mm` spheres.
- PoU uses compact local neighborhoods only, exact L1 semantic compatibility,
  the locked quadratic spatial kernel, and explicit empty-support failure.
- No learned radius, covariance, Gaussian, point topology, U-Net, attention,
  transformer, local FFT, target-pixel access, routing, or reconstruction
  loss may be introduced.
- T1ce, ground truth, segmentation labels, and any target-derived value are
  forbidden from every frontend computation: input, semantic prior, point
  initialization/refinement, support construction, base-plane projection,
  routing, and stopping. An optional mask is legal only with non-target-derived
  provenance; this policy does not create a data adapter. T1ce may be used only
  after prediction in separately approved training loss, validation,
  evaluation, or metric code, none of which is authorized for Phases 1–5.
- Phase 1–5 may use one shared MedicalNet forward only. The base-plane branch
  may apply only the PLAN-locked axis-conditioned collapse to its existing
  `[B, C, D, H, W]` shallow feature and produce documented `XY`, `XZ`, and
  `YZ` base planes. It may not introduce a second encoder, FFT/DCT alternative,
  learned 3-D support, or a new coordinate convention. RAS `XYZ` millimetres,
  `[d, h, w]`, spacing, and affine semantics remain unchanged.

## Unresolved boundary

Wavelet spectral anchor `A` (including all DWT choices), cross-plane point
query/fusion, dynamic reconstruction tri-plane `Z`, selector, updater, history,
stopping, decoder, reconstruction loss, and full T1ce synthesis are
research-gated, type-only interfaces. Do not add defaults or behavior to them
until the corresponding research decision is supplied.

## Verification

Add small CPU synthetic tests under `tests/features/point_guided/`. Run the
smallest affected test file first, then the frontend smoke test, `compileall`,
and `git diff --check`. A pass is software evidence only, not reconstruction,
clinical, or novelty validation.
