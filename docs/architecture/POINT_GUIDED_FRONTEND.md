# Point-guided multimodal MRI frontend

## Current boundary

The implemented boundary is only:

```text
T1 / T2 / FLAIR volume -> coarse semantic prior -> bounded refined points
-> semantic-aware compact-support PoU
```

The input tensor order is `[B, 3, D, H, W]` with channels `(T1, T2, FLAIR)`.
T1ce is rejected at this boundary rather than treated as a fourth channel.
Tensor indices are `[d, h, w]`; physical coordinates are canonical `XYZ` in
millimetres. `spacing_mm` is ordered `(x, y, z)`, matching `(w, h, d)`. A
batch must already share one registered grid/affine; per-patient affine
adaption belongs to a future data-adapter decision.

`PointGuidedMRIModel.forward_frontend` returns soft semantic probabilities,
initial and refined physical point centres, bounded displacements, point-centre
semantics, and a sparse PoU edge list. It does not synthesize T1ce.

## Locked rules

- The coarse prior uses the full [MedicalNet ResNet10 layout](https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py)
  and a minimal `1x1x1` semantic head. A supplied local one-channel checkpoint
  is SHA-checked when a digest is supplied and adapted to three channels by
  deterministic replication/averaging; no checkpoint is downloaded implicitly.
  A shape-compatible local file is recorded as a custom checkpoint, never
  called verified official pretrained weights. No official digest is bundled
  in this scaffold.
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

This implementation is a sparse software-contract reference. It has no
default-scale (`N=2048` or `N=3072`) runtime or memory-performance evidence,
and makes no throughput, reconstruction-quality, or clinical claim.

## Intentional non-implementations

Spectral anchors, dynamic tri-planes, trajectory selection and updates,
history, stopping, final decoding, and reconstruction losses are interfaces
only. They must not read a target, open a dataset path, mutate patient state,
or return a fake T1ce volume.

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
