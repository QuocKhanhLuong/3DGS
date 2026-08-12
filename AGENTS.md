# AGENTS.md

## Active objective

Implement only the locked point-guided frontend for `T1/T2/FLAIR -> T1ce`
research. The executable frontend currently ends after one shared MedicalNet
traversal, coarse semantics, deterministic initial points, bounded point
refinement, point-centre semantics, sparse semantic-aware PoU, and static
diagnostic base planes `Bxy/Bxz/Byz`. It must never return a fake T1ce volume.

Repository authority is deliberately staged:

- Phases 1-5 are **IMPLEMENTED**.
- Gate A is **CLOSED / LOCKED**; Phase 6 is **AUTHORIZED / IMPLEMENTABLE**,
  but **NOT IMPLEMENTED**.
- Gate B is **CLOSED / LOCKED**; Phase 7 is **AUTHORIZED / IMPLEMENTABLE**,
  but **NOT IMPLEMENTED**.
- Gate C is **BLOCKED** and remains default-deny.

Authorization is not implementation and does not begin work automatically. An
active task must explicitly name its authorized phase; Phase 6 precedes Phase
7. `B` is a feature-only base projection. It is not the as-yet-unimplemented
spectral anchor `A`, point spectral evidence `f_spec`, a dynamic tri-plane, or
a reconstruction path.

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
  transformer, local FFT, target-pixel access, routing, or reconstruction
  loss may be introduced. Phase 6/7 authorization does not relax these bans.
- T1ce, ground truth, segmentation labels, and any target-derived value are
  forbidden from every frontend computation: input, semantic prior, point
  initialization/refinement, support construction, base-plane projection,
  spectral anchor construction, spectral query, geometry handling, routing,
  and stopping. An optional mask is legal only with non-target-derived
  provenance; this policy does not create a data adapter. T1ce may be used only
  after prediction in separately approved training loss, validation,
  evaluation, or metric code, none of which is authorized here.
- Phases 1-7 may use one shared MedicalNet forward only. The base-plane branch
  may apply only the PLAN-locked axis-conditioned collapse to its existing
  `[B, C, D, H, W]` shallow feature and produce documented `XY`, `XZ`, and
  `YZ` base planes. It may not introduce a second encoder, FFT/DCT alternative,
  learned 3-D support, or a new coordinate convention. RAS `XYZ` millimetres,
  `[d, h, w]`, spacing, and affine semantics remain unchanged.

## Phase 6 authorization: fixed spectral anchor A

Phase 6 may add only a fixed two-level, 2-D stationary/undecimated Haar
transform of each static base plane and the static spectral anchor `A`. It is
not implemented by this authorization. The MAIN contract is fixed normalized
Haar filters `L = [1, 1] / sqrt(2)` and `H = [1, -1] / sqrt(2)`, reflect
padding, stride one without downsampling, and the exact stored band order:

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

Reflect is the only MAIN padding. Future Phase 6 code must fail closed with a
clear `ValueError` or typed failure when a required plane dimension is one;
it must not silently substitute zero, replicate, or circular padding.
Implementation remains PyTorch-only: fixed buffers/constants plus grouped
`Conv2d` or equivalent PyTorch operations. `pywt`, PyWavelets,
`pytorch_wavelets`, and `kymatio` remain prohibited. Fixed Haar filters are
non-trainable buffers/constants; only the shared band projector and existing
base-plane scorers may evolve in the model state. Learned Haar filters,
independent band/plane projectors, external hidden spectral networks, and
`torch.fft` remain forbidden.

## Phase 7 authorization: geometry-aware point spectral evidence

Phase 7 may add only deterministic feature-grid geometry bookkeeping and
point-level spectral evidence from the still-unimplemented `A`. It is not
implemented by this authorization. A minimal helper may derive shallow/B/A
grid shape, affine/centre mapping, RAS-mm-to-feature-grid coordinates,
grid-sample coordinates, and plane coordinates from `VolumeGeometry` plus the
actual convolution/pooling spatial transform. It must support rotation, shear,
anisotropic spacing, and translation; hard-coded `coordinate / 2` is
forbidden. Learned registration, learned coordinate transforms, deformation
fields, target-derived coordinate correction, and semantic-derived coordinate
warps are forbidden.

For refined `p_i*` in RAS-mm, Phase 7 may bilinearly query `Axy(x,y)`,
`Axz(x,z)`, and `Ayz(y,z)` to obtain `f_xy`, `f_xz`, and `f_yz`, each 56-d.
It may derive only the deterministic descriptor
`q = concat([LL2, E1, E2])` (24-d), where
`E1 = sqrt(LH1^2 + HL1^2 + HH1^2 + eps)` and
`E2 = sqrt(LH2^2 + HL2^2 + HH2^2 + eps)`. Three pairwise cosine agreements,
mean agreement per plane, and softmax reliability may weight the raw plane
features. The only MAIN packing is
`concat([alpha_xy * f_xy, alpha_xz * f_xz, alpha_yz * f_yz])`, exactly 168-d
with XY, XZ, then YZ provenance. No nearest-neighbour/patch/sphere query,
second encoder, transformer, cross-attention, learned confidence MLP, hard
plane drop, 104-d canonical fusion, or learned `168->64` compression is
authorized.

## Gate C: unresolved boundary

Dynamic `Z0`/`Z_t`, selector, top-k routing, point revisit, updater, scatter
or overlap handling, history, stopping, decoder, reconstruction/spectral/
pathology loss, training pipeline, and T1ce synthesis remain research-gated,
type-only interfaces. Do not add defaults or behavior to them. Legacy
`anchors`, `fields`, `memory`, `routing`, reconstruction, training, evaluation,
CLI, and data systems remain outside the frontend; the word "anchor" does not
authorize their reuse.

## Verification

Add small CPU synthetic tests under `tests/features/point_guided/`. Run the
smallest affected test file first, then the frontend smoke test, `compileall`,
and `git diff --check`. A pass is software evidence only, not reconstruction,
clinical, or novelty validation.
