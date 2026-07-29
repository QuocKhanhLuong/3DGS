# CVPR Internal Health and Next-Tranche Design

Date: 2026-07-29  
Role: Architect  
Branch audited: `main` at `d79fbf7`  
Scope: read-only comparison of all tracked documentation against the current codebase, plus a bounded design for the next implementation tranche

> Scope correction: the project owner subsequently confirmed that the four
> root design documents are deprecated. The synthesized deep-dive and all
> implementation decisions use only `docs/` as authoritative. Any comparison
> to deprecated root documents below is retained only as raw specialist context.

## Executive decision

The repository is not implementation-ready for the full four-phase system. The current `main` branch is a documentation-only research specification: it has no `src/`, tests, dependency manifest, dataset schema, experiment configuration, or runnable baseline.

The newest reconstruction package is materially more coherent than the older root documents, but it still assumes several unresolved contracts at exactly the boundaries most likely to invalidate a medical reconstruction paper: observation legality, physical coordinates, source-affine provenance, tensor shapes, differentiability, uncertainty semantics, and evaluation isolation.

The recommended next tranche is therefore:

> **T0 — Leakage-safe physical-plane observation and differentiable Gaussian rendering kernel**

T0 should implement only:

1. immutable sparse-manifest and commit/reveal contracts;
2. canonical physical-coordinate and source-affine contracts;
3. explicit observation, plane, target-grid, and Gaussian tensor interfaces;
4. a pure-PyTorch reference physical-plane Gaussian renderer;
5. synthetic analytic, coordinate-invariance, leakage, and gradient tests.

This is the smallest scientifically coherent executable unit because it establishes the legal observation operator and the differentiable reconstruction operator on which every later encoder, field, Gaussian-memory, and routing claim depends. It is not itself a CVPR result.

Do not implement the encoder, true SDF, adaptive topology, learned uncertainty, multi-wave routing, custom CUDA, or full-volume export in T0.

## Sources and precedence

### Documents reviewed

Every tracked file under `docs/` was read, together with:

- `README.md`;
- `architecture.md`;
- `pipeline.md`;
- `MASTER_KNOWLEDGE.md`;
- `KNOWLEDGE_PACKAGE.md`.

No `docs/research/` or `docs/strategies/` directory exists on `main`, so there was no current research report or PM strategy artifact to cross-reference.

### Precedence decision

The authoritative order should be:

1. `docs/reconstruction/PROOFREAD_NOTES.md`;
2. `docs/reconstruction/README.md`;
3. `docs/reconstruction/FULL_FLOW.md`;
4. the four phase specifications and five module specifications;
5. root `README.md`;
6. `architecture.md`, `pipeline.md`, `MASTER_KNOWLEDGE.md`, and `KNOWLEDGE_PACKAGE.md` as legacy context only.

This ordering follows the recent git history: commits `459236c` through `d79fbf7` intentionally changed Phase 1 to teacher-free permanently sparse training and changed the primary task to reconstruction. The legacy root documents were not reconciled in those commits.

## Verified repository state

At the time of final inspection:

- `main` and `origin/main` both pointed to `d79fbf7`;
- all tracked content was Markdown;
- no tracked source, tests, package/dependency manifest, configuration, data schema, or CI existed;
- tracked files were clean;
- `.agenteam/` and `.codex/` were untracked orchestration artifacts;
- recent commits were documentation commits only.

Consequently, documented module coverage on current `main` is:

| Documented area | Current executable implementation | Tests | Status |
|---|---:|---:|---|
| Sparse manifest and legal observation provider | none | none | blocker |
| Coordinate/source-affine contracts | none | none | blocker |
| Analytic differential scaffold | none | none | missing |
| Evidence encoder | none | none | missing |
| Anchor-local field | none | none | missing |
| Structural Gaussian memory | none | none | missing |
| Volumetric appearance memory | none | none | missing |
| Physical-plane renderer | none | none | blocker |
| Render-before-update assimilation | none | none | missing |
| Uncertainty/observability | none | none | missing |
| Candidate graph and routing | none | none | missing |
| Convergence controller | none | none | missing |
| Volume reconstruction/export | none | none | missing |
| Experiment and reproducibility harness | none | none | blocker |

## Design drift and contradictions

### 1. Primary task drift — blocker for paper scope

The current reconstruction package locks full multi-sequence 3D reconstruction as the primary task. The legacy documents still recommend tumor segmentation as primary and held-out reconstruction as auxiliary:

- `MASTER_KNOWLEDGE.md` presents segmentation as the recommended task;
- `KNOWLEDGE_PACKAGE.md` presents segmentation as the first-paper task;
- `architecture.md` says segmentation can be primary for the first implementation;
- `pipeline.md` prioritizes segmentation loss and segmentation heads.

Decision: the current task is reconstruction. Legacy task language must not drive implementation or paper claims. Segmentation may be an audit/downstream metric, not a training requirement for T0 or the first reconstruction baseline.

### 2. Training-data drift — blocker for validity

The current package requires permanently sparse patient manifests and forbids complete-volume targets in the main training path. The legacy pipeline says complete registered volumes exist as hidden supervision and suggests sampling held-out slices from them during representation training.

These are different scientific regimes:

- **per-episode sparsity from a densely available training volume**;
- **permanently sparse training supervision**.

They must not be described as equivalent. Main-path loaders must enforce the latter. Dense volumes may exist only in an isolated audit process or explicitly privileged ablation.

### 3. Implementation order is internally circular — high

The current README orders encoder work before the physical-plane renderer and Gaussian state, but the teacher-free encoder gate requires downstream sparse target-plane reconstruction under matched anchor/Gaussian logic. The encoder cannot pass its scientific gate without a trustworthy downstream observation operator.

Decision: build and verify the legal data/geometry/renderer substrate before interpreting encoder comparisons. An encoder-only benchmark with an unrelated 2D head can be a diagnostic but cannot pass the documented Phase-1 gate.

### 4. “SDF” is not yet a valid locked property — high

The module and phase documents correctly caution that the field may only be a structural level-set field. Elsewhere, the project name, module names, formulas, and claims still use SDF unconditionally.

Missing evidence includes:

- a defined sign convention;
- sign or surface supervision;
- Eikonal validation;
- gradient-norm statistics;
- distance calibration against known geometry;
- robustness at blended field boundaries.

Decision: code APIs should use `StructuralField` until signed-distance tests pass. `SDF` may remain the hypothesis and ablation label.

### 5. Structural covariance has two incompatible contracts — high

`architecture.md` defines tangent-plane isotropy with two scales:

\[
\Sigma=\sigma_t^2(I-nn^\top)+\sigma_n^2nn^\top.
\]

`SDF_GAUSSIAN_MEMORY.md` defines two independent tangent scales plus one normal scale:

\[
\Sigma=R\operatorname{diag}(\sigma_{t1}^2,\sigma_{t2}^2,\sigma_n^2)R^\top.
\]

Decision for T0: support a general SPD covariance through a stable factorization in the renderer contract, while using log diagonal scales plus a frame for the first structural-Gaussian constructor. Tangent isotropy versus anisotropy is a later ablation, not two simultaneous core definitions.

### 6. Gaussian “opacity” and MRI composition are conflated — high

The documents sometimes call \(\alpha_i\) opacity-like, but the renderer explicitly rejects camera-style front-to-back alpha compositing and uses normalized additive blending.

Decision: call this value `support_amplitude` or `mixture_weight` in the MRI core. Do not inherit visibility/occlusion semantics from perspective 3DGS.

### 7. Plane axes are underspecified and transpose-prone — blocker

The plane formula uses

\[
x(u,v)=o+u\Delta_u r+v\Delta_v c,
\]

while images are stored as `[H, W]`. The terms “row direction” and “column direction” are frequently reversed across DICOM APIs and array-index conventions.

Decision: the executable contract must avoid `row`/`column` as primary axis names. Use:

- `axis_u`: direction of increasing tensor width index;
- `axis_v`: direction of increasing tensor height index;
- `spacing_u`, `spacing_v`;
- `pixel_center_origin`;
- `normal = normalize(axis_u × axis_v)`.

Metadata adapters may expose source-specific row/column labels, but they must convert into this canonical contract and record the source convention.

### 8. Coordinate convention is absent — blocker

The docs do not lock:

- canonical RAS versus DICOM LPS;
- millimeters versus voxels;
- voxel-center versus voxel-corner origin;
- affine direction;
- array axis order;
- normal sign;
- permitted shear;
- handedness validation;
- registration transform direction.

Without these choices, a renderer can pass image-space tests while reconstructing mirrored, transposed, or offset anatomy.

Decision: use canonical patient-space RAS millimeters, pixel-center origins, homogeneous source-to-canonical transforms, and independently validated signed slice axes. Preserve source convention and source affine as provenance.

### 9. Physical plane and volume grid contracts do not round-trip — high

`PhysicalPlane`, `ObservationMeta`, and `TargetGrid` are described separately, but no rule maps:

- tensor `[D,H,W]` indices;
- `shape_xyz`;
- affine columns;
- plane `(u,v)` indices;
- thickness/slab bounds.

Decision: define one tested `index_to_world` convention and derive planes and target grids from it. Never carry both an affine and independently editable direction/spacing/origin fields without agreement validation.

### 10. Uncertainty is a list of heuristics, not a statistical quantity — high

The current state mixes distance, coverage, missing modality, disagreement, residuals, propagation depth, and optional learned confidence. No unit, range, monotonicity, calibration target, or aggregation semantics are specified.

Decision: T0 returns `support_mass` and `unsupported_mask`, not “calibrated uncertainty.” A later uncertainty tranche must separately name epistemic support, residual variance, modality missingness, and calibrated predictive error.

### 11. Learned routing supervision contradicts permanent sparsity — high

The documents recognize this issue: dense counterfactual gain for every candidate is unavailable in the main training regime. Oracle routing requires an isolated dense simulator/audit protocol.

Decision: the first active baseline must be metadata-only and analytic. Learned utility and receding-horizon routing are deferred until the data source for counterfactual rewards is declared.

### 12. Global and patient state notation is ambiguous — medium

Symbols such as `C_t`, `M_t`, `R_t`, and `U_t` are reused for feature cache, coverage, modality state, residual/predicted gain, redundancy, and uncertainty. This will leak into incompatible tensor interfaces.

Decision: executable records use descriptive names, never single-letter state fields.

### 13. “One encoder execution per committed slice” needs mode semantics — medium

The cache is immutable at new-patient inference because weights are frozen. During offline training, encoder weights change after optimization steps, so cached feature maps cannot safely persist across optimizer steps.

Decision:

- training cache lifetime: one forward/episode only;
- inference cache lifetime: patient session, keyed by content digest and encoder-version hash;
- no detached cache on a training gradient path.

### 14. Scope is too broad for one CVPR claim — high

The design currently combines:

- a teacher-free structural encoder;
- local implicit fields;
- low-DoF dual-bank Gaussians;
- adaptive topology;
- uncertainty;
- active multi-wave routing;
- convergence-based stopping;
- full multi-modality reconstruction.

This risks a paper in which no component is isolated deeply enough. The encoder should be framed as enabling infrastructure unless experiments prove it is a standalone contribution. The central paper thesis should remain the coupling between a physically constrained sparse representation and budgeted observability.

## Coordinate and geometry contract

T0 should lock the following semantics.

### Canonical frame

- patient-space coordinates: RAS;
- unit: millimeters;
- homogeneous coordinates use column vectors;
- `source_to_canonical` maps a source physical coordinate into canonical RAS;
- source convention is an enum such as `DICOM_LPS`, `NIFTI_RAS`, or `CANONICAL_RAS`;
- conversions preserve the original source affine and convention for audit.

### Plane

```text
PhysicalPlane
├── pixel_center_origin_ras_mm: [3]
├── axis_u_ras: [3]   # increasing W index
├── axis_v_ras: [3]   # increasing H index
├── spacing_uv_mm: [2]
├── thickness_mm: scalar
├── shape_hw: [2]
├── signed_normal_ras: [3]
├── source_transform
└── observation_id
```

Required invariants:

- all values finite;
- positive spacing and thickness;
- axes unit length and mutually orthogonal within tolerance;
- signed normal unit length;
- `axis_u × axis_v` agrees in sign with the independently sourced slice axis;
- source affine origin and in-plane axes agree with the canonical plane;
- `shape_hw` contains positive integers;
- the pixel at tensor index `[v,u]` maps to

\[
x=o+u\,spacing_u\,axis_u+v\,spacing_v\,axis_v.
\]

If source metadata encodes voxel corners, the adapter must explicitly shift to pixel centers.

### Volume grid

```text
TargetGrid
├── index_to_ras_mm: [4,4]
├── shape_dhw: [3]
├── modality_ids
└── normalization_records
```

The mapping from tensor index `[d,h,w]` to homogeneous index must be explicit and tested. `shape_xyz` should not appear in code because it confuses physical axes with tensor axes.

### Gaussian state

```text
GaussianBatch
├── center_ras_mm: [N,3]
├── covariance_factor: [N,3,3]
├── log_support_amplitude: [N,1]
├── appearance: [N,M]
├── appearance_valid: [N,M]
├── primitive_kind: [N]
└── primitive_id: [N]
```

The covariance is

\[
\Sigma=LL^\top+\epsilon I.
\]

Use triangular solves or Cholesky-based solves; do not form `inverse(Σ)` in the hot path. Scales and amplitudes should use bounded positive parameterizations.

Patient-specific tensors must not be registered as global trainable parameters at inference.

## Tensor interface decisions

### Observation batch

```text
image:                 [B,1,H,W] floating
valid_mask:            [B,1,H,W] bool
modality_index:        [B] int64
plane origins:         [B,3]
plane axes_u/axes_v:   [B,3]
spacing_uv_mm:         [B,2]
thickness_mm:          [B]
observation_ids:       length B immutable IDs
normalization record:  metadata, not silently discarded
```

Variable shapes should first be handled by same-shape batches or explicit padding plus a valid mask. A ragged abstraction is unnecessary in T0.

### Evidence encoder, next tranche

```text
analytic_channels: [B,C_phi,H,W]
Z_str:             [B,C_str,H/s,W/s]
Z_app:             [B,C_app,H/s,W/s]
reliability:       [B,1,H/s,W/s] optional
```

The feature-grid-to-plane transform must be returned with the features. `align_corners` and half-pixel semantics for `grid_sample` must be fixed and tested rather than left to library defaults.

### Renderer

```text
render_plane(gaussians, planes, modality_index)
    -> intensity:        [B,1,H,W]
       support_mass:     [B,1,H,W]
       unsupported_mask: [B,1,H,W] bool
       diagnostics:      bounded non-gradient metadata
```

The first implementation may require all planes in a batch to share `H,W`. Chunking is allowed along pixels and Gaussians as long as outputs and gradients match the unchunked reference within tolerance.

### Numeric policy

- geometry validation and source-affine conversion: float64 on CPU;
- model/render tensors: float32 by default;
- gradient tests: float64;
- mixed precision: deferred until the float32 reference passes;
- epsilon values: named configuration fields, not anonymous literals.

## Differentiability contract

### Required gradient path

For later Phase 1, the following path must be differentiable:

```text
target-plane loss
→ normalized Gaussian composition
→ Gaussian appearance/amplitude/covariance/center
→ anchor/local-field outputs
→ sampled evidence maps
→ evidence encoder
```

T0 must verify the renderer portion of that path.

### Explicit boundaries

- Fixed analytic derivative filters should be implemented as fixed tensor convolutions so gradients can reach input pixels when needed.
- Source metadata validation, manifest lookup, commit/reveal, modality IDs, candidate selection, culling index construction, topology operations, and routing are discrete and non-differentiable.
- Conditional on a selected neighbor set, Gaussian weights and composition remain differentiable.
- Hard culling can create gradient discontinuities at support boundaries; T0 tests equivalence where the culling margin includes all non-negligible support.
- Adaptive birth/split/prune is excluded from the initial training graph.
- If field normals later come from \(\nabla F\), training losses that backpropagate through normals require `create_graph=True` and therefore second-order derivatives.
- Normalization by support mass must expose unsupported pixels rather than convert them into confident zeros.
- Training caches must retain autograd history for their episode; inference caches must be detached.

### Renderer mathematics to settle

For an infinitesimally thin plane, use a documented conditional or marginal Gaussian-plane formulation consistently. The current document names conditional covariance while also applying a normal-distance gate. The amplitude normalization must be derived so that the same 3D Gaussian does not arbitrarily change energy under plane rotation.

T0 should implement:

1. a thin-plane analytic reference;
2. a sampled finite-slab reference;
3. an optional analytic finite-slab path only after it matches sampled quadrature.

No perspective projection or camera alpha-compositing dependency is appropriate.

## Dependency assessment

There is no dependency manifest, so versions, vulnerabilities, CUDA compatibility, or maintenance status cannot currently be audited. This absence is itself a blocker.

Recommended minimal dependency boundary:

| Dependency | Initial role | Risk/decision |
|---|---|---|
| Python | runtime | pin one supported minor range |
| PyTorch | tensor/autograd/reference renderer | high compatibility risk across CUDA; pin and record CPU/CUDA matrix |
| NumPy | serialization/test interop | keep out of differentiable hot path |
| nibabel | NIfTI affine I/O adapter | use only at boundary; canonical contract remains internal |
| pytest | tests | required |
| hypothesis | coordinate/property testing | recommended but optional if dependency minimization is critical |

Deferred dependencies:

- MONAI: useful for medical transforms/metrics but can obscure affine and interpolation semantics; do not make core geometry depend on it initially.
- SimpleITK/pydicom: add only when actual input formats require them.
- SciPy: reference integration or spatial indexing only; no gradient path.
- NetworkX or graph libraries: defer until analytic routing.
- `gsplat` or perspective 3DGS CUDA renderers: do not reuse for the physical MRI operator without a mathematical equivalence proof.
- custom CUDA kernels: defer until the pure-PyTorch reference and gradient tests are stable.

## Tech debt and severity

| Area | Severity | Finding |
|---|---|---|
| Executable repository | blocker | no source, tests, dependencies, configs, or CI |
| Data legality | blocker | no enforceable manifest/commit/reveal boundary |
| Coordinate provenance | blocker | no canonical convention or affine agreement checks |
| Observation operator | blocker | no renderer or analytic reference |
| Scientific evaluation | blocker | no dataset/split/audit protocol or runnable baseline |
| Documentation consistency | high | current reconstruction task conflicts with legacy segmentation/dense-supervision docs |
| Tensor/API contracts | high | shapes, dtypes, devices, masks, batching, and axis semantics not executable |
| Differentiability | high | discrete operations and gradient boundaries not specified in code/tests |
| Uncertainty | high | heuristic components are conflated with calibrated predictive uncertainty |
| Identifiability | high | hidden pathology cannot be guaranteed from insufficient sparse observations |
| Reproducibility | high | no configs, seeds, environment lock, run manifest, or artifact schema |
| Paper scope | high | too many contributions and ablations for one first implementation |
| SDF naming | medium/high | SDF claims precede signed-distance evidence |
| Computational feasibility | medium/high | local fields, dual banks, routing, and adaptive topology have no complexity budget |
| Cache lifecycle | medium | training and inference cache semantics differ but are not encoded |
| Symbol/record naming | medium | reused symbols invite state-interface errors |

## Scientific and CVPR feasibility blockers

### Data and audit

Before a CVPR-level result, the project needs:

- a declared dataset and license;
- patient-level train/validation/test partitions;
- a fixed procedure for creating or receiving permanently sparse manifests;
- a physically isolated fully sampled audit set;
- scanner/site/vendor metadata if domain-shift claims are intended;
- registration quality and failure criteria;
- missing-modality policy;
- artifact and invalid-content policy.

If no held-out fully sampled audit volumes exist, full-volume fidelity cannot be claimed quantitatively.

### Baselines

At minimum, later experiments need:

- nearest/linear/spline sparse-slice interpolation;
- fixed uniform and random observation selection;
- a matched shallow 2D/2.5D reconstruction baseline;
- a coordinate/implicit field baseline;
- free Gaussian versus constrained Gaussian;
- analytic scaffold only versus raw shallow CNN versus scaffold plus micro-CNN;
- uncertainty-greedy versus single-wave versus balanced multi-wave only after routing exists.

All comparisons must use identical patient splits, legal observations, budgets, target grids, and audit processes.

### Metrics and statistics

Later experiment infrastructure should report:

- MAE, NMSE, PSNR, SSIM/MS-SSIM, and NCC;
- edge/gradient/frequency-band error;
- lesion- or ROI-sensitive fidelity where evaluation labels exist;
- quality-budget AUC and target-quality budget;
- latency, peak VRAM, primitive count, cache bytes, and total preprocessing cost;
- calibration curves only for explicitly defined uncertainty;
- patient-level confidence intervals;
- multiple training seeds for learned variants;
- failure counts, including `INSUFFICIENTLY_OBSERVED`.

### Reproducibility

Every run should serialize:

- git commit and dirty state;
- environment/package lock;
- full resolved configuration;
- random seeds;
- dataset and sparse-manifest hashes;
- patient split hash;
- context/target IDs;
- every file-open audit event relevant to leakage;
- coordinate convention and normalization record;
- checkpoint and output hashes;
- per-stage runtime and memory.

## Implementation approaches

### Approach A — Contract-first physical-plane kernel

Build the sparse-manifest boundary, geometry contracts, Gaussian state, pure-PyTorch renderer, and synthetic tests first.

Advantages:

- resolves the highest-risk validity issues early;
- creates a trustworthy gradient path for every later Phase-1 experiment;
- supports synthetic analytic validation without waiting for real data;
- limits the first implementation to a reviewable module;
- makes later optimized renderers testable against a reference.

Trade-offs:

- does not yet test encoder novelty or reconstruction quality on patients;
- front-loads contract work that may look less impressive in a demo;
- requires careful math for thin planes and finite slabs.

### Approach B — Encoder-first Phase-1 proxy benchmark

Build the sparse loader, analytic channels, E0/E1/E2 encoders, and a temporary lightweight target-plane prediction head.

Advantages:

- produces early representation-learning curves;
- tests collapse, alignment, and modality agreement quickly;
- may identify whether the micro-CNN is viable before complex state work.

Trade-offs:

- the proxy head may reward shortcuts unrelated to the locked anchor/Gaussian system;
- it cannot satisfy the documented “matched downstream pipeline” Phase-1 gate;
- geometry and leakage bugs can contaminate conclusions;
- the temporary head is likely throwaway code.

### Approach C — Full four-phase interface skeleton

Create stubs and preliminary implementations for provider, encoder, field, Gaussian memory, renderer, router, assimilation, convergence, and export.

Advantages:

- makes the entire dependency graph visible;
- enables early end-to-end smoke execution.

Trade-offs:

- maximizes simultaneous ambiguity;
- encourages placeholder behavior to be mistaken for evidence;
- makes gradient and coordinate bugs difficult to localize;
- creates large review and refactor cost;
- violates the proofread document’s own recommendation not to code the full system before the Phase-1 gate.

## Recommendation

Choose **Approach A**.

It is the only approach that produces a scientifically meaningful executable contract without prematurely choosing the encoder, SDF, topology, router, or acceleration strategy. Approach B should follow immediately as T1, using the verified T0 renderer rather than a throwaway predictor. Approach C should be rejected for now.

## T0 file and ownership plan

The Dev implementation plan should account for all of the following files. Names may be adjusted to repository conventions, but responsibilities should not be merged into one large module.

```text
pyproject.toml
src/smagm/__init__.py
src/smagm/contracts/__init__.py
src/smagm/contracts/coordinates.py
src/smagm/contracts/observations.py
src/smagm/contracts/gaussians.py
src/smagm/data/__init__.py
src/smagm/data/manifest.py
src/smagm/render/__init__.py
src/smagm/render/plane.py
tests/contracts/test_coordinates.py
tests/contracts/test_observations.py
tests/contracts/test_gaussians.py
tests/data/test_manifest_legality.py
tests/render/test_plane_renderer_analytic.py
tests/render/test_plane_renderer_coordinates.py
tests/render/test_plane_renderer_gradients.py
tests/render/test_plane_renderer_chunking.py
tests/integration/test_context_target_reveal_barrier.py
tests/integration/test_synthetic_sparse_plane_roundtrip.py
```

Optional test fixtures should live under `tests/fixtures/` and contain only synthetic, redistributable data.

## T0 acceptance criteria

### Contract tests

- immutable records reject nonfinite values and invalid shapes;
- LPS-to-RAS and RAS identity landmarks round-trip;
- singular or malformed affines fail clearly;
- plane/source origin and axes must agree;
- a flipped independently sourced signed normal is rejected;
- `[v,u]` pixel-center mapping is explicit and correct;
- equivalent coordinate-frame transforms render equivalent images;
- serialization is canonical and includes units/convention.

### Legality tests

- the main provider cannot open a non-manifest path;
- target metadata can be requested without target pixels;
- target pixels cannot be requested before a committed reveal token;
- context and target IDs are disjoint;
- patient splits cannot occur at slice level;
- the file-open audit log is deterministic and hashable.

### Renderer tests

- one isotropic Gaussian on an aligned plane matches an analytic image;
- rotated anisotropic Gaussian behavior matches an independent reference;
- finite-slab quadrature converges as samples increase;
- empty support produces a stable output plus `unsupported_mask`;
- normalized composition remains finite for tiny support mass;
- chunked and unchunked forward results agree;
- known synthetic Gaussian fields re-render reproducibly.

### Gradient tests

- `torch.autograd.gradcheck` passes for center, covariance factor, support amplitude, and appearance in float64;
- finite differences agree away from hard-culling boundaries;
- chunked and unchunked gradients agree;
- no target tensor enters a context-built graph before reveal;
- no unintended `.detach()`, NumPy conversion, or in-place mutation cuts the path.

### Reproducibility tests

- deterministic synthetic cases under a recorded seed;
- package imports in a clean subprocess;
- CPU test suite passes without a GPU;
- optional CUDA parity is reported separately, not required to establish the reference.

## T1 after T0

Only after T0 passes should T1 implement:

1. fixed analytic differential channels;
2. E0 analytic-only baseline;
3. E1 raw shallow CNN;
4. E2 analytic scaffold plus teacher-free micro-CNN;
5. structural warm-up diagnostics;
6. a minimal fixed-topology Gaussian state for sparse context-to-target rendering;
7. matched-FLOP and matched-downstream comparisons.

T1 must not add learned routing, adaptive topology, a true-SDF claim, or a large appearance decoder.

## Risks and mitigations

| Risk | Level | Mitigation |
|---|---|---|
| Coordinate convention bug yields plausible but wrong anatomy | blocker | canonical RAS-mm contract, independent source-affine validation, transform-invariance tests |
| Hidden target leakage invalidates sparse claim | blocker | provider capability boundary, commit/reveal token, file-open audit, isolated evaluation process |
| Renderer math mismatches physical MRI slice formation | high | thin-plane and finite-slab references, state exact approximation in paper |
| Discrete culling/topology breaks gradients | high | dense reference first, generous culling margin, topology deferred |
| Sparse training is insufficient for hidden pathology | high | calibrated unsupported regions, audit evaluation, narrow claims |
| Encoder learns generic edges | high | downstream E0/E1/E2 matched comparison in T1 |
| Cross-modality loss suppresses private lesions | high | align structural branch only, reliability weighting, ROI audit |
| Custom kernel changes results | medium/high | optimized implementation must match reference forward and backward |
| Dual-bank model becomes too large | medium | primitive and byte budgets before T2 |
| Paper contribution becomes diffuse | high | treat encoder as enabling component unless independently compelling |

## YAGNI and scope-control flags

The following are premature before T0 and T1 evidence:

- balanced multi-wave routing;
- D*-style incremental graph repair;
- learned candidate utility;
- receding-horizon or reinforcement learning;
- adaptive birth/split/merge/prune;
- second-order SDF optimization;
- curvature-aligned covariance;
- learned uncertainty calibration;
- custom CUDA rasterization;
- NIfTI/DICOM support beyond the first selected data format;
- full-volume export UI or visualization;
- teacher/pretrained upper-bound encoders;
- multi-dataset domain generalization;
- segmentation training heads.

Each may become justified later, but implementing them now would expand scope without resolving the current blockers.

## Decisions

1. Treat `docs/reconstruction/*` as authoritative over legacy root documents.
2. Keep reconstruction as the primary task.
3. Enforce permanently sparse main training and isolated dense audit evaluation.
4. Use canonical RAS millimeters and pixel-center plane semantics.
5. Preserve and independently validate source-affine provenance.
6. Use normalized additive MRI composition, not perspective alpha compositing.
7. Call the geometry a structural field until SDF behavior is measured.
8. Build a pure-PyTorch physical-plane reference before acceleration.
9. Make T0 a contract-and-renderer tranche; make encoder comparison T1.
10. Defer routing, topology, learned uncertainty, and full export.

## Constraints

- no unqueried pixels before a ledger commit;
- no non-manifest file access in main training;
- no audit volume in mutable patient state;
- no free ambiguity between tensor axes and physical axes;
- no unvalidated duplicate affine/plane metadata;
- no silent unsupported-voxel fill;
- no patient-specific state registered as global inference weights;
- no claim of global route optimality or uniquely recoverable hidden pathology;
- no CVPR-level claim until patient-level audit experiments and baselines exist.

## Next steps

1. PM/Researcher should reconcile this internal assessment with external literature and select a narrow paper thesis, dataset, and audit protocol.
2. Dev should translate T0 into a line-level implementation plan using the file list and acceptance criteria above.
3. QA should make coordinate, leakage, analytic renderer, and gradient checks blocking gates.
4. Reviewer should reject T1 if T0 coordinate invariance or leakage tests fail.
5. After T0, run the E0/E1/E2 T1 benchmark before any SDF, routing, or topology expansion.
