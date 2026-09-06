# Point-guided multimodal MRI frontend

## Repository authority

The repository has three explicit states, which must not be conflated:

- **Implemented:** PLAN Phases 1-7, ending in typed static base planes
  `Bxy/Bxz/Byz`, the typed static SWT-Haar anchor `Axy/Axz/Ayz`, typed
  geometry-aware point spectral evidence `f_spec`, Gate-C C1-C7 bounded
  dynamic tri-plane trajectory diagnostics, and Gate-D D1 final-Z-only
  chunked implicit decoding.
- **Implemented supervision boundary:** Gate E E1-E9, with target permitted
  only after a target-free context. **Next / inactive:** Gate F. **Blocked /
  default-deny:** Gate G and the final inference policy.

The frontend exposes `f_spec` as typed diagnostic evidence at the
already-refined points. An explicitly configured Gate-C call uses it only for
bounded dynamic-state trajectory diagnostics; the explicit Gate-D endpoint
then decodes final Z only. Gate E subsequently applies only a separate
target-after-inference objective; it does not alter those inference paths.

## Current boundary

The implemented boundary is:

```text
T1 / T2 / FLAIR volume
        -> one shared MedicalNet ResNet10 traversal
           -> deep feature -> coarse semantic prior -> bounded refined points
                            -> semantic-aware compact-support PoU
           -> configured selected feature -> static diagnostic Bxy/Bxz/Byz
                                            -> fixed SWT-Haar -> static Axy/Axz/Ayz
           -> refined points + derived feature-grid geometry
                                            -> bilinear XY/XZ/YZ query
                                            -> reliability-weighted `f_spec` [B,N,168]
                                            -> optional bounded Gate-C `Z` trajectory diagnostics
```

The input tensor order is `[B, 3, D, H, W]` with channels `(T1, T2, FLAIR)`.
T1ce is rejected at this boundary rather than treated as a fourth channel.
Tensor indices are `[d, h, w]`; physical coordinates are canonical `XYZ` in
millimetres. `spacing_mm` is ordered `(x, y, z)`, matching `(w, h, d)`. A
batch must already share one registered grid/affine; per-patient affine
adaption belongs to a future data-adapter decision.

`PointGuidedMRIModel.forward_frontend` returns soft semantic probabilities,
initial and refined physical point centres, bounded displacements, point-centre
semantics, a sparse PoU edge list, and typed static `BaseTriPlanes`. The
projector consumes the Phase-2 selected shared map once, then the fixed SWT
branch returns a typed static `SpectralAnchor`. The Phase-7 branch queries
that anchor only at the already-refined points and returns typed reliability
and `f_spec`; it does not feed back into point/refinement/PoU. It does not
synthesize T1ce.

## Implemented locked frontend scope

`PLAN.md` Phases 1–7 are implemented engineering work. They do not authorize
full reconstruction:

1. expose one shared MedicalNet pre-MaxPool shallow feature and an optional
   Layer1 ablation feature;
2. add explicit detach/tap and frozen/fine-tuned ablation controls;
3. lock the production coarse semantic state to `normal brain`, `edema`, and
   `tumor-core candidate`;
4. project the configured selected shared feature into static base planes
   `Bxy`, `Bxz`, and `Byz`; and
5. expose those base planes as typed diagnostic frontend data; and
6. derive the fixed two-level SWT-Haar static anchor `Axy`, `Axz`, and `Ayz`
   with one shared 64-to-8 per-band projector; and
7. derive selected feature-grid geometry, bilinearly query `A` at refined
   RAS-mm points, and emit deterministic 168-d `f_spec`; and
8. when explicitly configured, run Gate C C1-C7 bounded dynamic-state
   reward-cost trajectory diagnostics without decoder or target data.

`B` is a feature-only base projection, not wavelet spectral anchor `A`,
cross-plane fusion, a dynamic tri-plane, or a decoder input. The shared
MedicalNet forward must not run twice and no second encoder is permitted.
Its Phase 4 implementation may use only PLAN's locked axis-conditioned
collapse; it may not introduce an FFT/DCT alternative, learned 3-D support,
or an unapproved residual spectral adapter.
For `[B, C, D, H, W]`, `Bxy` collapses `D/Z` to `[B, C, H, W]`, `Bxz`
collapses `H/Y` to `[B, C, D, W]`, and `Byz` collapses `W/X` to
`[B, C, D, H]`. These names preserve the existing RAS `XYZ` and tensor `DHW`
mapping; they create no separate coordinate convention.

## Locked rules

- The coarse prior uses the full [MedicalNet ResNet10 layout](https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py)
  and a minimal `1x1x1` semantic head. A supplied local one-channel checkpoint
  is SHA-checked when a digest is supplied and adapted to three channels by
  deterministic replication/averaging; no checkpoint is downloaded implicitly.
  A shape-compatible local file is recorded as a custom checkpoint, never
  called verified official pretrained weights. No official digest is bundled
  in this scaffold.
- The Phase 3 production semantic contract is exactly three soft classes:
  `normal brain`, `edema`, and `tumor-core candidate`. It is implemented and
  fail-closed: `PointGuidedConfig` accepts only this class count, and the
  public frontend output requires the corresponding three-channel tensors.
- Initial point selection depends only on geometry and the optional brain mask.
  It is deterministic, quasi-uniform, and produces the configured count. For
  a voxel mask, a point is legal when its nearest voxel centre is a valid mask
  element; candidates remain sub-voxel and non-Cartesian.
- Point displacements are measured from the original centre and projected to
  at most `2 mm`. Supports are fixed `4 mm` spheres.
- Directional context samples `+/-` RAS axes at `1`, `2`, and `3 mm` using
  trilinear interpolation. It uses centre-relative values, never a cropped
  3-D patch.
- PoU edges are computed only inside local sphere bounds. No
  `[B, N, D, H, W]` allocation is permitted.
- Semantic affinity is `1 - 0.5 * L1`; spatial affinity is
  `(1 - distance / radius)^2` inside the sphere; their product is normalized
  over contributors at each covered voxel. A voxel with local spatial support
  but a zero semantic-affinity denominator is emitted in the sparse
  `unsupported_*` record rather than silently normalized away. If this leaves
  no positive edge anywhere, construction raises `EmptySparseSupportError`
  and attaches those sparse unsupported records; it never returns an invalid
  all-zero PoU object.
- T1ce, ground truth, segmentation labels, and every target-derived value are
  forbidden from the observation input and all frontend branches, including
  the semantic prior, points, PoU, and authorized base-plane projection. An
  optional brain mask is admissible only with non-target-derived provenance.

This implementation is a sparse software-contract reference. It has no
default-scale (`N=2048` or `N=3072`) runtime or memory-performance evidence,
and makes no throughput, reconstruction-quality, or clinical claim.

## Implemented: Gate A / Phase 6 static anchor A

Phase 6 implements only the static spectral-anchor branch:

```text
Bxy/Bxz/Byz
  -> fixed two-level 2-D stationary/undecimated Haar
  -> seven same-grid bands per plane
  -> one shared 1x1 Conv2d(64 -> 8), applied per band and plane
  -> Axy/Axz/Ayz (56 channels each)
```

The MAIN transform uses fixed normalized filters
`L = [1, 1] / sqrt(2)` and `H = [1, -1] / sqrt(2)`, stride one, appropriate
stationary level-2 dilation, no downsampling, and reflect padding. It stores
exactly this order:

```text
LL2, LH1, HL1, HH1, LH2, HL2, HH2
```

`LL1` is an intermediate approximation, not an eighth output. For an input
plane `[B,C,H,W]`, every stored band remains `[B,C,H,W]`; the static anchors
are `Axy [B,56,H,W]`, `Axz [B,56,D,W]`, and `Ayz [B,56,D,H]`. MAIN
normalization is none. The only retained optional ablation is
`band_gn = GroupNorm(7,56)`, which must default off. The `Conv2d` bias is an
implementation-detail choice until PLAN explicitly locks parameter count or
bias behavior.

Band names have a locked axis convention. In `[B,C,H,W]`, the first filter
symbol is H/the row axis and the second is W/the column axis:

```text
LL = low H,  low W
LH = low H,  high W
HL = high H, low W
HH = high H, high W
```

Physically, XY has H/row = Y and W/column = X; XZ has H/row = Z and W/column
= X; YZ has H/row = Z and W/column = Y. This convention governs synthetic
orientation tests unless PLAN later explicitly changes it.

Phase 6 is PyTorch-only: fixed Haar tensors must be buffers/constants and use
grouped `Conv2d` or equivalent PyTorch tensor operations. `pywt`, PyWavelets,
`pytorch_wavelets`, and `kymatio` are prohibited, as are learned Haar filters,
seven independent band projectors, three per-plane projectors, hidden spectral
networks, and `torch.fft`. The shared band projector remains trainable; fixed
filters do not. The existing B scorers remain the only other authorized
upstream trainable state.

Reflect padding is locked rather than best-effort. The implementation validates
required plane dimensions and raises a clear `ValueError` or typed failure when
a required dimension is one. It does not silently use zero, replicate, or
circular padding.

## Implemented: Gate B / Phase 7 point spectral evidence

Phase 7 deterministically derives the selected feature-grid geometry from the
live convolution/pooling chain and the full source affine, then queries the
implemented static anchors at refined physical `p_i*` in canonical RAS-mm:

```text
p_i* + input VolumeGeometry + actual convolution/pooling spatial transform
  -> shallow/B/A feature-grid geometry
  -> bilinear Axy(x,y), Axz(x,z), Ayz(y,z)
  -> f_xy, f_xz, f_yz in R^56
  -> deterministic reliability
  -> f_spec in R^168
  -> typed `f_spec` [B,N,168]
  -> optional bounded Gate-C trajectory diagnostics
  -> final Z only -> chunked 96-d implicit decoder -> absolute prediction
  -> optional Gate-E target-after-inference objective
  -> STOP before Gate F
```

The minimal helper may derive feature-grid shape, affine/centre mapping,
RAS-mm-to-feature-grid voxel coordinates, grid-sample coordinates, and plane
coordinates. It must support rotated, sheared, anisotropically spaced, and
translated input affines. Hard-coded feature coordinates such as
`original_coordinate / 2` are forbidden. This is deterministic geometry
bookkeeping, not learned registration: learned coordinate transforms,
deformation fields, target-derived coordinate correction, and semantic-derived
warps are prohibited.

Each query is single-point bilinear interpolation, not nearest-neighbour,
patch pooling, sphere pooling, or dense point-to-plane tensors. SWT preserves
the B-plane grids, so it introduces no extra coordinate rescaling. The raw
56-d plane features remain intact. Only for reliability, their seven 8-d band
blocks may form the deterministic 24-d descriptor:

```text
E1 = sqrt(LH1^2 + HL1^2 + HH1^2 + eps)
E2 = sqrt(LH2^2 + HL2^2 + HH2^2 + eps)
q  = concat([LL2, E1, E2]) in R^24
```

The only MAIN reliability rule is three pairwise cosine similarities, mean
agreement per plane, then softmax, with nonnegative weights summing to one.
The only MAIN packing is:

```text
concat([
  alpha_xy * f_xy,
  alpha_xz * f_xz,
  alpha_yz * f_yz,
]) in R^168
```

The 168-d layout permanently preserves XY, then XZ, then YZ 56-d blocks and
the Gate-A band order within each block. No second encoder, transformer,
cross-attention, confidence MLP, hard plane drop, 104-d canonical fusion, or
learned `168 -> 64` compression is authorized. A majority-consistency failure
mode is known: two incorrect agreeing planes can outvote one correct outlier;
do not silently add a learned judge or semantic prior to change this rule.

## Implemented: Gate C C1-C7 bounded dynamic trajectory

With an explicit `TrajectoryConfig`, `forward_trajectory` first runs the
shared Phase-1-7 frontend once, then initializes three 32-channel `Z` planes
from static B, bilinearly queries only Z at refined points, scores the locked
126-d reward descriptor, applies explicit travel/overlap/step utility, makes a
hard or straight-through adaptive selection, writes a bounded 4-mm local
correction, and returns only final Z plus compact route diagnostics. B, A,
refined points, point semantics, `q`, reliability, `f_spec`, and feature-grid
geometry remain fixed for the route. The explicit Gate-D endpoint then reads
only final Z with the same geometry; it is not a target lookup.

Gate D D1 is complete only as the explicit final-Z decoder. Gate E adds a
separate target-after-inference supervision objective; Gate F training is
next/inactive and Gate G/final-inference policy remain default-deny. No
Gate-C/D path may read targets, open a dataset
path, or persist patient state. This does not authorize legacy `smagm.anchors`,
`smagm.fields`, `smagm.memory`, `smagm.routing`, reconstruction, training,
evaluation, CLI, or data packages.

## BraTS21 boundary

The existing BraTS21 code is a legal sparse-plane preparation system, not a
`[B, C, D, H, W]` volume loader. A future adapter must make the `T1/T2/FLAIR`
channel order explicit, convert its source `[x, y, z]` and affine convention to
the frontend's `[D, H, W]` / RAS-mm contract, and preserve any target-reveal
policy outside this frontend. In particular, the existing BraTS21 patient
object exposes T1ce and therefore must not be passed to the frontend or treated
as a legal point-guided volume adapter.

The optional brain mask controls initial candidate legality and PoU support.
The locked refiner currently guarantees only that a refined centre remains in
the registered volume box; it does not claim a new sub-voxel brain-mask
segmentation policy.

## Navigation rule

Use `python scripts/codegraph.py --task <frontend|medicalnet|data-boundary-audit|tests|quality>`
before opening files. The policy grants the smallest task-specific read/write
set and explicitly blocks legacy routing, training, anchor, field, memory, and
evaluation packages from frontend work.

## Additive PFGR-Lite W1-W5 boundary and ledger

PFGR-Lite is an additive, separately versioned implementation under
`src/smagm/features/point_guided/pfgr_lite/`. Its W1 composition wraps a
legacy `PointGuidedMRIModel` created without `TrajectoryConfig`, calls the
private four-result shared-feature seam exactly once, and computes
graph-preserving `Z0` while ordered observations and the shallow/Layer1/deep
`MedicalNetFeatures` bundle are alive. The context stores only target-free
typed frontend evidence, mask/geometry provenance, and owned 32-channel
initial planes; `initialize_state` clones those planes without detaching.

The static heads are explicitly versioned `b0_legacy_v1`,
`b1_multiscale_v1`, `b2_ordered_multiscale_v1`, and `b_light_ordered_v1`.
B1 and B2 instantiate equal-width/equal-depth source-slot projections; B1
feeds those slots with explicit zeros while B2 uses ordered source samples at
live affine-aware feature-cell centres. Static geometry derives centre
lattices from every live Conv/Pool/residual transform and supports full RAS
rotation, shear, anisotropic spacing, translation, and uneven shapes. The
existing prepool B/A branch remains unchanged and no target/output bypass is
available.

`PFGRLiteModel.decode_final` deliberately requires W2's canonical
`PFGRQueryLattice` injection. W2 must provide an object with
`build(output_geometry, feature_geometry, query_dtype, build_chunk_size)` and
the resulting `query(state.planes, voxel_ids_dhw, chunk_size=...) -> [Q,96]`
(or `[B,Q,96]`) callable; the existing decoder MLP is reused on those query
features. There is no silent legacy-query fallback, so this standalone W1
package remains target-free until the explicit W2 dependency checkpoint.

PFGR declarations/configurations are frozen under `pfgr-lite-config-v1`,
with N=2048, state width 32, correction width 96, write scale 0.1, fixed
4-mm support/2-mm displacement, FP32 production, FP64 test mode, and policy
budgets 0/1/2/4. `ProducerCompatibility` excludes V architecture/weights,
fit settings, calibration, CLI/metrics, and a bare Git SHA; `ValueFitIdentity`
and `CalibrationIdentity` bind those separately. Synthetic/untrained
MedicalNet provenance is reported honestly and never mints a real-data claim.

The user-authorized PFGR implementation spans W1-W5 under this package. W1
owns the declarations/static seam described above; W2 owns the canonical
query lattice, footprint, sparse writer, and teacher; W3 owns stages, bank,
and value fitting; W4 owns immutable proposals, policy, calibration,
checkpoint, and inference; W5 owns CLI, metrics, benchmark, oracle
diagnostics, and the runbook. The completion ledger is intentionally
additive: W1 fix-round software is pending independent re-review, while W2-W5
implementation and experimental evidence remain pending. Navigation does not
permanently block those future package paths; each dispatch still enforces its
exclusive file ownership.

W1 CPU fixtures may opt into `PFGRLiteConfig(engineering_only=True)` with a
reduced point count to keep tests bounded. Production manifests reject reduced
N and declare N=2048. The first implementation processes one subject at a time
(B=1), serializing subjects when full batch metadata is unavailable; producer
compatibility hashes describe frozen global algorithms/components, whereas
subject affine/shape, resolved observation mask, and observed values remain in
context/action/replay identities.
