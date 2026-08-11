# AGENTS.md

## Active objective

Implement only the locked point-guided frontend for `T1/T2/FLAIR -> T1ce`
research. The frontend ends after coarse semantics, deterministic initial
points, bounded point refinement, point-centre semantics, and a sparse
semantic-aware PoU. It must never return a fake T1ce volume.

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

## Scientific contracts

- Input is `[B, 3, D, H, W]` ordered `(T1, T2, FLAIR)`; physical points are
  canonical RAS `XYZ` in millimetres and volume indices are `[d, h, w]`.
- Keep voxel spacing explicit; default `(1.0, 1.0, 1.0)` is not a hard-coded
  assumption.
- The MedicalNet ResNet10 prior is frozen by default, has only a minimal
  semantic head, and never downloads or pretends to load weights.
- Initial points are deterministic, quasi-uniform, mask/geometry dependent,
  and value independent. Refined displacement is always relative to the
  original point and bounded by `2 mm`; supports are fixed `4 mm` spheres.
- PoU uses compact local neighborhoods only, exact L1 semantic compatibility,
  the locked quadratic spatial kernel, and explicit empty-support failure.
- No learned radius, covariance, Gaussian, point topology, U-Net, attention,
  transformer, local FFT, target-pixel access, routing, or reconstruction
  loss may be introduced.

## Unresolved boundary

Spectral anchor, dynamic tri-plane, selector, updater, history, stopping,
decoder, and reconstruction loss modules are type-only interfaces. Do not add
defaults or behavior to them until a new research decision is supplied.

## Verification

Add small CPU synthetic tests under `tests/features/point_guided/`. Run the
smallest affected test file first, then the frontend smoke test, `compileall`,
and `git diff --check`. A pass is software evidence only, not reconstruction,
clinical, or novelty validation.
