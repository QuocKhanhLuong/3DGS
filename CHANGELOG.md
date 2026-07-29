# Changelog

## Unreleased — 2026-07-29

### Changed

- Realigned the project and AgenTeam roles from a CVPR-first active-routing
  thesis to the ISBI 2027 permanently sparse support-anchor Gaussian
  representation thesis, with active routing deferred until after static
  representation gates.
- Replaced the reconstruction implementation order with human-gated T0 through
  T5 tranches and marked T0 complete, T0.5/T1 next, and T2+ unauthorized in the
  current run.
- Removed root README links to deleted legacy knowledge and architecture files.

### Added

- ISBI strategy precedence, full-mechanism novelty-collision analysis,
  permanently sparse training protocol, and T0.5/T1 interface design delta.
- PyTorch CPU-first technology stack (`Python >=3.10`, NumPy interoperability,
  `torch`, and a `pytest` test extra) for the T0 reference operator.
- Verified macOS arm64/Python 3.10 CPU dependency lock for the T0 reference
  environment under `requirements/`.
- Reproducibility record for the verified synthetic T0 CPU validation under
  `docs/reproducibility/`.
- `env.yaml` fast Conda bootstrap for Linux x86_64/RTX 4070, pinned to
  Python 3.10 and the official PyTorch 2.12.1 CUDA 12.6 wheel; GPU parity
  remains a server-side verification gate.
- `smagm.contracts.coordinates`: canonical RAS-mm source-affine, physical-plane,
  and target-grid contracts.
- `smagm.contracts.observation`: immutable sparse manifests, patient split
  validation, commit/reveal access control, and deterministic opened-file audit.
- `smagm.contracts.observations`: compatibility export for observation contracts.
- `smagm.gaussians` and `smagm.contracts.gaussians`: validated general-SPD
  Gaussian tensor batch contract and compatibility export.
- `smagm.renderer` and `smagm.render.plane`: differentiable analytic thin-plane
  and sampled finite-slab normalized additive Gaussian MRI renderer with
  configurable PSF profiles and compatibility export.
- `smagm.data.manifest`, plus package `__init__` modules under `smagm`,
  `smagm.contracts`, `smagm.data`, and `smagm.render`, for focused top-level
  imports and stable T0 module paths.
- Manifest-bound file access and content-addressed open-audit records, so files
  inside the data root but outside the sparse manifest remain inaccessible.
- Private payload-integrity digests excluded from public observation metadata,
  canonical manifest serialization, and the pre-commit manifest hash.
- Private raw payload access, required real-observation source provenance,
  cohort-wide patient split validation, target-budget enforcement, single-use
  reveal capabilities, exact decimal budget accounting, and deterministic
  commit/reveal/open event auditing.
- PSF/slab quadrature over depth-wise normalized latent intensity, with an
  independent multi-Gaussian regression oracle that distinguishes it from
  density-weighted numerator/support integration.
- Weighted PSF support coverage with an explicit configurable acceptance
  threshold and per-pixel `supported_psf_mass` diagnostics.
- Per-render Gaussian tensor revalidation and dtype-aware numeric bounds for
  covariance, amplitude, appearance, coordinates, and support thresholds.
