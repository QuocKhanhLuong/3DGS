# Changelog

## Unreleased — 2026-07-31

### Governance and phase-gate synchronization

- documented `CODEBASE.md` as the final theory-to-code blueprint and added the
  T0–T5 phase-gate quality layer;
- added the machine-readable checklist, evidence runner, ignored local reports,
  and clean-commit/dirty-state semantics;
- synchronized AgenTeam role workflow and narrowed the executable pipeline to
  the authorized T1-B software tranche;
- synchronized strategy, README, reconstruction, and Codex status documents;
- recorded the explicit T1-B Human Gate `PASS` decision and the later Human
  authorization to implement only T1-C, without authorizing T2+;
- made no scientific-validation claim; T1-B `PASS` is a software-gate decision
  only, while retrospective T0/T0.5/T1-A gates remain unrecorded.
- added the canonical active-document index and documentation lifecycle and
  retention policy;
- removed superseded CVPR-first, pre-code health, meeting, proofread, and
  completed implementation-plan documents from the active tree;
- normalized retained reconstruction theory to the static ISBI thesis and
  added document-link and stale-status quality checks;
- no runtime scientific behavior changed in this documentation cleanup.

### Added

- T1-C manifest-bound NumPy plane decoding, exact context-only normalization,
  deterministic matched episode schedules, declared registration records, and
  explicit missing-modality policy.
- Supported-mask reconstruction losses, legal receipt-gated episode
  orchestration, gradient/precision/accumulation policy, safe checkpoint
  round-trips, immutable provenance, and a CPU synthetic training CLI.
- T1-C legality, loss, E0/E1/E2 autograd/fairness, independent-state,
  checkpoint, and provenance tests. The tranche remains pending Human Gate and
  makes no reconstruction-quality claim.
- T1-C hardening now adds modality-aware one-target episode sampling,
  minimum-scale normalization fallbacks, decoded-input cache identity,
  config-driven run artifacts, typed objective stages, accumulation-safe exact
  resume, and expanded fairness/provenance bindings. It remains an
  implementation candidate under review, not a scientific validation result.
- T1-C checkpoint resume now rejects mismatched or tampered run, manifest,
  split, and schedule bindings; checkpoint cadence and selection are executed
  from resolved configuration; provenance includes opened-file-ledger,
  preprocessing, dependency, and artifact-digest bindings. This remains
  software-contract evidence only.

- T1-B teacher-free E0/E1/E2 encoder contracts with common structural,
  appearance, bounded-reliability, per-item geometry, and explicit topology
  outputs. E1/E2 use only small standard PyTorch micro-CNN layers.
- Explicit masked teacher-free structural consistency, appearance sensitivity,
  reliability regularization, variance-floor diagnostics, and registered
  cross-modality rejection.
- Fail-closed in-memory feature-cache keys bound to observation, source plane,
  encoder/configuration/state, preprocessing, transform, valid topology, dtype,
  and output channels. Target-derived cache insertion is rejected.
- CPU synthetic `smagm.cli.t1b` demonstrations for E0, E1, and E2.

### Changed

- T1-A feature maps now bind one source-plane transform and modality ID per
  batch item; the fixed-support sampler selects only the requested item's
  geometry and provenance.
- T1-A local covariance factors are rotated into canonical RAS with a typed
  differentiable Cholesky epsilon before construction of `GaussianBatch`.
- Analytic gradient magnitude is exactly zero for zero gradients, and analytic
  validity is topology-only erosion by the largest local-contrast support.

## Historical Unreleased Notes — 2026-07-29

### Changed

- Tightened the in-review T1-A analytic-to-Gaussian reference: analytic
  differential channels now respect declared in-plane millimetre spacing,
  support selection uses only the declared valid feature topology (never
  learned reliability values), and Gaussian centre offsets are mapped through
  each support plane's canonical RAS basis before the one-time gauge-safe
  runtime conversion.
- Locked T1-A feature geometry to half-pixel `align_corners=False` output
  strides 1, 2, or 4, including explicit invalid right/bottom padded centres
  for odd input shapes. Public fixed-support construction now requires an
  orthonormal per-support `(u, v, signed-normal)` RAS basis.
- Bound T1-A feature grids and fixed supports to the exact canonical source
  plane (including observation provenance), carried its SHA-256 signature into
  Gaussian primitive provenance, and moved deterministic `max_points`
  truncation after valid/non-padded row-major candidate filtering. Sampled
  evidence tensors remain attached to their feature-map autograd graph.
- Hardened bound feature-grid public geometry helpers so an explicit plane must
  canonically match the transform-bound source plane; separately constructed
  canonical equivalents remain valid.

- Realigned the project and AgenTeam roles from a CVPR-first active-routing
  thesis to the ISBI 2027 permanently sparse support-anchor Gaussian
  representation thesis, with active routing deferred until after static
  representation gates.
- Replaced the reconstruction implementation order with human-gated T0 through
  T5 tranches and marked T0 complete, T0.5/T1 next, and T2+ unauthorized in the
  current run.
- Removed root README links to deleted legacy knowledge and architecture files.

### Added

- T0.5 legal episodic-training contracts: role-free immutable sparse
  availability manifests, canonical immutable episode assignments, and a
  receipt-gated `EpisodeLedger` whose pure renderer remains outside ledger
  state.  Prediction receipts are minted only by `PredictionRegistrar` from a
  detached audit copy of an actual `RenderResult`.
- Tightened T0.5 receipt provenance: only `EpisodeController` may invoke the
  pure renderer for a committed target, using a factory-frozen gauge-provenanced
  live Gaussian state that is rehashed before rendering; training ledgers now
  reject sealed audit cohorts.
- Added factory-only patient split registries for the four declared cohort
  labels and persistent prediction receipt records with Gaussian-state and
  renderer-output-schema provenance.
- Exact-`Decimal` deployment acquisition schedules and ledgers, separate from
  zero-cost offline episode role assignment, with bootstrap and subsequent
  observation charging by immutable modality/plane cost key.
- Per-conversion differentiable
  `MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE` raw-Gaussian factory
  provenance; direct validated `GaussianBatch` construction remains a T0
  compatibility path.
- Ubuntu CPU GitHub Actions checks for pytest, compileall, and clean-diff
  validation.
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
