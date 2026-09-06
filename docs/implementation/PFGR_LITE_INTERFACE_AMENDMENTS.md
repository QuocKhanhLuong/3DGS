# PFGR-Lite interface amendments — verified Astra companion v1

Date: 2026-09-07 (Asia/Ho_Chi_Minh). This companion is authoritative together
with `PFGR_LITE_IMPLEMENTATION_PLAN.md` at SHA-256
`5cd2cd9ecc720aee6b2463de6dcfbf5f4b0a8b5c48723409e5c29af515de7985`.
It resolves findings B1–B4 from `msg_bcd26de7962f` and the coordinator's
scoped-dependency concern. The original plan remains unchanged; this file is
the only phase-zero amendment owned by this dispatch.

The plan plus this amendment are **ACCEPTED for implementation**, conditional
on the software gates below. This is not scientific acceptance: no trained
checkpoint, real-data result, CUDA result, reconstruction claim, or clinical
claim is implied.

## 1. Authority and bounded scope

W1 owns the shared declarations and the single-traversal/static seam. W2 owns
the PFGR query lattice, footprint, sparse query-delta, and detached teacher.
W3 owns stages, bank, and value fitting. W4 owns proposal identity, policy,
calibration, checkpoint, and inference. W5 owns CLI, metrics, benchmark,
oracle diagnostics, and runbook. A worker may consume a declaration below but
may not redefine it; a required signature change is an amendment request to
W1 and the coordinator.

All implementation workers are dispatched as `gpt-5.6-luna` with reasoning
`max`; each launch receipt must be checked for that exact pair. Astra owns
architecture and review only. No new backbone, PFGR-Full head, RL objective,
long training, or automatic final-cohort evaluation is authorized.

## 2. B1 — one traversal and graph-preserving Z0 ownership

`encode_observations` is the only PFGR entrypoint that consumes observations.
It validates ordered `x: Tensor[B,3,D,H,W]`, builds `VolumeGeometry`, and calls
the existing `SemanticPrior.extract_intermediate_features(x)` exactly once.
While that `MedicalNetFeatures(shallow, layer1, deep)` bundle and the ordered
`x` are live, it must compute the complete target-free frontend and the chosen
static head's `initial_planes` Z0. The head receives the ordered source and
the true shallow/layer1/deep lattices; it may not recover either through a
second traversal or a module cache.

W1's concrete seam is:

```python
def encode_observations(
    self,
    x: Tensor,                         # [B,3,D,H,W], T1/T2/FLAIR
    brain_mask: Tensor | None,
    geometry: VolumeGeometry,
) -> ObservationContext:
    ...

def initialize_state(self, context: ObservationContext) -> PFGRState:
    ...
```

`ObservationContext` is a frozen metadata object with owned, finite tensors:

```python
ObservationContext(
    context_id: str,
    frontend: FrontendOutput,
    q_bar: Tensor,                         # [B,N,24]
    feature_geometry: FeatureGridGeometry, # final Z/query lattice
    initial_planes: DynamicTriPlanes,      # XY/XZ/YZ, 32 channels
    producer: ProducerDependencies,
    source_provenance: SourceProvenance,
)
```

The context stores no target, segmentation, `ValidatedTargetContext`, oracle,
teacher, or full `MedicalNetFeatures` bundle. `initial_planes` are graph-
preserving owned tensors produced by the static head while ordered `x` and
the feature bundle are present. `initialize_state` creates a state-owned
`DynamicTriPlanes` with `tensor.clone()` for each plane; it must not call
`.detach()`, `.data`, `inference_mode`, or `torch.no_grad()`. Cloning is
required to prevent state writes from mutating the context and must leave
`grad_fn` connected in S0/S1. The local feature bundle is released after Z0
construction; no feature-volume retention is permitted merely to initialize
state later. Teacher/bank snapshots are the separate detached-copy boundary.

`PFGRState` has the following minimum fields:

```python
PFGRState(
    planes: DynamicTriPlanes,
    context_id: str,
    state_version: int,                    # 0 at initialization
    state_digest: str,
    producer: ProducerCompatibility,
    role: Literal["deployment", "training_behavior"],
)
```

The state constructor rejects context/geometry/producer mismatches, nonfinite
values, and in-place tensor changes detected by the owned digest/version guard.
`decode_final(state, context, *, chunk_size)` receives only final `state` and
typed geometry/query metadata; target-free APIs never receive a target.

Required W1 tests include one forward hook proving exactly one MedicalNet
traversal; B1/B2 ordered-source distinguishability; state-plane clone
graph/gradient preservation; state/context write isolation; and unchanged
legacy `forward_frontend`, `forward_trajectory`, Gate-E, and state-dict
behavior at fixed seeds.

## 3. B2 — canonical PFGR query lattice and complete footprint

The PFGR path adopts one explicitly versioned source-voxel-to-feature query
lattice. The legacy decoder/query path remains unchanged. This removes the
omitted-axis ambiguity exposed by the prior probe while avoiding a full plane
clone for every candidate.

W2 owns these concrete contracts in `footprint.py`/`sparse_write.py`:

```python
PFGRQueryLattice.build(
    output_geometry: VolumeGeometry,
    feature_geometry: FeatureGridGeometry,
    *,
    query_dtype: torch.dtype,       # production fp32; fp64 test mode
    build_chunk_size: int,
) -> PFGRQueryLattice

PFGRQueryLattice.query(
    state: DynamicTriPlanes,
    voxel_ids_dhw: Tensor,          # [Q,3] integer output centres
    *,
    chunk_size: int,
) -> Tensor:                         # [Q,96], XY|XZ|YZ

build_footprint(
    lattice: PFGRQueryLattice,
    action: ActionProposal,
    *,
    chunk_size: int,
) -> SparseFootprint

query_write_delta(
    lattice: PFGRQueryLattice,
    footprint: SparseFootprint,
    voxel_ids_dhw: Tensor,
    delta: Tensor,                   # [96], actual stored proposal delta
) -> Tensor:                         # [Q,96]
```

The lattice is derived from the live `FeatureGridGeometry` and the output
`VolumeGeometry` using the existing `[w,h,d,1] -> RAS XYZ mm` convention and
the actual Conv/Pool centre transform. It records a query-version string,
both geometry hashes, output/feature shapes, and dtype. For every output voxel
and each XY/XZ/YZ plane it stores the four integer bilinear neighbours, their
weights, and valid zero-padding flags using the exact half-voxel
`align_corners=False` formula. No border clamping, epsilon pruning, physical
axis substitution, or relaxed tolerance is allowed.

`PFGRQueryLattice.query` is the canonical PFGR query used by PFGR final
decoding, the independent full-write reference, and sparse query-delta. The
shared operation is:

```text
q_before(v) = lattice.query(Z, v)
dq(v)        = [s_xy(v)*delta_xy,
               s_xz(v)*delta_xz,
               s_yz(v)*delta_yz]
q_after(v)   = q_before(v) + dq(v)
prediction   = D.mlp(q_after(v))
```

The MLP is still exactly `96 -> 64 -> 32 -> 1`; only the query mechanics are
shared. `reference_full_write` may materialize a full hypothetical plane only
as a parity oracle. The optimized teacher never needs one.

Footprint construction first obtains the actual positive retained plane-node
weights from the existing writer: full affine RAS distance, fixed 4-mm
quadratic kernel, clipped local windows, and strict `weight > 0`. It then
looks up the inverse lattice index for every node's positive bilinear stencil.
The footprint is the union of unique output voxel IDs across all three planes;
`multiplicity(v)` is the number of plane supports containing that voxel. A
fixed write support is not a sphere, cube, omitted-axis fibre, or target mask.
An exactly zero stored delta may be retained for a diagnostic no-op, but an
adaptive action with no retained support is illegal.

The lattice may be built once per `(output_geometry, feature_geometry,
query_version, query_dtype)` and cached/chunked. Its inverse node-to-voxel
index is constructed once per geometry, never once per candidate. If the
declared memory bound prevents indexed materialization, W2 must use an exact
full-output voxel scan with the same stencils as a correctness fallback. The
fallback is legal but records `footprint_mode="full_scan_fallback"`, scanned
voxel count, bytes, and latency; it cannot be reported as a sparse-speed
result. No candidate may silently use a mathematical envelope that has not
been certified against the production stencils.

The MAIN exact support and fixed-Q sampling domains are explicit:

```text
F_phi       = unique output voxels reached by plane phi's positive stencils
n_phi       = |F_phi|
S           = n_xy + n_xz + n_yz
c(v)        = count of planes whose F_phi contains v
p(v)        = c(v)/S, for v in F_xy union F_xz union F_yz
```

For `iid_fixed_q`, choose a plane with probability `n_phi/S`, then a voxel
uniformly from that plane's unique support. Every union voxel has `p(v)>0`.
For Q independent draws with replacement:

```text
g_hat = mean_q( mask(v_q) * [rho(before_q)-rho(after_q)] / (M*p(v_q)) )
```

where `M=sum(mask)>0` is the fixed whole-volume denominator. Mask rejection is
forbidden. Duplicate draws remain draws even when query results are cached;
report `Q_draws`, `unique_query_voxels`, `n_phi`, union size, multiplicity,
valid masked contributions, variance, and SE separately. `exact_footprint`
evaluates each unique union voxel once and has zero sampling uncertainty.
Screening samples and winner-confirmation samples use separate seeds/records;
an optionally stopped screen is never labelled an unbiased MAIN gain.

The old probe (`msg_bcd26de7962f`) found omitted-axis strict-support
mismatches under a rotated/sheared translated affine, even though direct
query-delta parity was within FP64/FP32 tolerances. This amendment therefore
requires lattice identity shared by final/reference/teacher, positive stencil
membership, and the indexed/fallback accounting above. The prior four-case
CPU probe remains feasibility evidence only: maximum query/prediction errors
were approximately `8.2e-16/5.6e-17` in FP64 and `3.9e-7/3.0e-8` in FP32,
with correction-gradient errors below `2.2e-11`; no production evaluator or
speed claim follows from it.

W2 parity tests must compare independent full-write and sparse query outputs,
signed gains, and correction gradients under interior/border, fractional,
anisotropic, rotated/sheared, translated, overlap, mask-hole, and zero-support
fixtures. Required tolerances remain FP64 `atol=1e-10, rtol=1e-9` and FP32
`atol=1e-6, rtol=1e-5`; a near-boundary mismatch blocks implementation until
the shared lattice or exact fallback is corrected.

## 4. B3 — bounded lazy-import ownership and target boundary

W1 additionally owns the parent
`src/smagm/features/point_guided/__init__.py` and the import seam in the
existing `src/smagm/features/point_guided/model.py`. This is a bounded import
repair, not a legacy behavior change.

The parent package keeps the same `__all__` and public names but resolves
`PointGuidedMRIModel` through a module-level lazy `__getattr__` (or equivalent
explicit lazy map). Importing contracts/configuration or
`smagm.features.point_guided.pfgr_lite` must not eagerly import `model`,
`training_objective`, `reward_supervision`, `oracle`, `bank`, `data`, or CLI
modules. `model.py` moves its Gate-E imports behind `TYPE_CHECKING` and
function-local imports in `forward_training_context`/
`compute_training_objective`; future annotations preserve public type hints.
Direct legacy imports still resolve the same objects and preserve existing
results, signatures, and exceptions once the Gate-E method is called.

The import-boundary test must run in a fresh subprocess and assert:

```text
import smagm.features.point_guided
assert target-bearing Gate-E/teacher modules are absent
from smagm.features.point_guided import PointGuidedMRIModel
assert target-bearing modules are still absent until a Gate-E method is called
```

PFGR inference additionally rejects target-bearing objects at type/role
validation. `ObservationContext`, `PFGRState`, `ActionProposalBatch`, and
`InferenceBundle` contain no target or segmentation references. Only W2's
`ValidatedTargetContext` and `measure_actions(...)` may accept T1ce, and only
after a sealed target-free `CompletedBehaviorTrace` exists. Replacing,
deleting, or poisoning the target after observation/trace construction must
not change deployment proposals, route, or prediction. If a legacy import
cannot be made lazy without changing a public behavior, document that exact
module as an exception; do not claim stronger import exclusion.

## 5. B4 — producer compatibility versus value-fit identity

W1 declares two distinct hashes. `ProducerCompatibility` covers only inputs
that can change a state, proposal, writer, query, or measured label:

```python
ProducerCompatibility(
    schema_version,
    observation_normalization_hash,
    geometry/query_version_hash,
    medicalnet_provenance_hash,
    frozen_bn_hash,
    static_head_hash,
    semantic_head_hash,
    point_refiner_hash,
    spectral_projector_hash,
    state_initializer_hash,
    updater_hash,
    decoder_hash,
    writer_hash,
    candidate_geometry_hash,
    label_definition_hash,
)
```

It explicitly excludes V architecture/weights, V optimizer/loss/fit
settings, calibration parameters, policy thresholds, W5 CLI/runbook/metric
code, and a bare whole-repository Git SHA. Source SHA/config are retained as
provenance fields for reproduction; relevant component hashes above are the
compatibility identity. A source change that alters a producer must change
that component's canonical state/config/version hash.

`ProducerDependencies` is the metadata envelope named by the original plan;
its `compatibility` member is exactly one `ProducerCompatibility` instance,
and its `source_provenance` member is `SourceProvenance`. The envelope may
carry diagnostic dependency details, but bank matching uses only the canonical
`ProducerCompatibility.digest`; workers must not create a second producer
compatibility hash.

`ValueFitIdentity` is separate:

```python
ValueFitIdentity(
    schema_version,
    input_variant,                 # 126 | 222 | 270 | 366
    architecture_hash,
    weights_hash,
    fit_config_hash,
    bank_manifest_hash,
    gain_scale_hash,
)
```

`ValueBankManifest` stores `producer_compatibility_hash`, label-definition
hash, split/role hashes, fixed gain-scale provenance, and row/shard hashes.
V126/V222/V270/V366 must load and fit from the same manifest and rows; changing
only V architecture/weights or refitting V does not invalidate the bank.
Changing any producer, writer/query lattice, geometry, normalization, mask,
label objective, or split/role membership rejects the bank. Calibration binds
both the producer-compatibility hash and one exact `ValueFitIdentity`; a V-only
change therefore keeps the bank valid but invalidates adaptive calibration
until recalibration. Tests must prove both directions.

## 6. W1 authoritative declarations

W1 is the sole declaration owner for these names; consumers import them rather
than creating parallel dataclasses:

```text
PFGRLiteConfig
StaticSynthesisConfig
PFGRPolicyConfig
ValueModelConfig
EffectTeacherConfig
ObservationContext
PFGRState
ActionProposalBatch
ActionProposal
Decision
CompletedBehaviorTrace
SparseFootprint
GainLabel
GainCalibration
ProducerCompatibility
ProducerDependencies
SourceProvenance
ValueFitIdentity
ValueBankManifest
OperationCounters
StageState
InferenceBundle
ResumeState
PFGRRouteResult
```

The aggregate `PFGRLiteConfig` contains exactly
`static: StaticSynthesisConfig`, `policy: PFGRPolicyConfig`,
`value: ValueModelConfig`, and `teacher: EffectTeacherConfig`, plus protocol
version and operational chunk/device fields. Frozen choices remain:
`candidate_count=2048`, `state_channels=32`, `correction_channels=96`,
`write_scale=0.1`, `support_radius_mm=4.0`, decoder `(96,64,32,1)`, FP32
production, CPU FP64 test mode, policy budgets `{0,1,2,4}`, MAIN revisit
`allow`, lowest point-ID ties, signed raw gain, MSE/fixed training-bank scale,
and teacher modes `exact_footprint`/`iid_fixed_q` (screening is diagnostic).

`ActionProposalBatch` owns ordered tensors `point_ids[B,N]`,
`points_ras_mm[B,N,3]`, `v126[B,N,126]`, `o270[B,N,270]`,
`delta[B,N,96]`, `legal[B,N]`, plus context IDs, state versions/digests,
producer versions, writer/query hashes, and a proposal digest. Execution must
gather the stored `delta` row; it may not call U again. `Decision` stores the
selected ID or `-1`, active flags, raw/calibrated/conservative values,
allowance, margin, compute cost, policy hash, and stop code; it never stores a
label or target. These constructors validate batch/order/device/dtype and
finite values before any worker-specific logic runs.

## 7. Initial implementation checkpoint and dispatch order

The first dispatch is W1 only, on the new PFGR package and the bounded legacy
import seam. W1's exclusive additions are:

```text
src/smagm/features/point_guided/pfgr_lite/__init__.py
src/smagm/features/point_guided/pfgr_lite/types.py
src/smagm/features/point_guided/pfgr_lite/config.py
src/smagm/features/point_guided/pfgr_lite/provenance.py
src/smagm/features/point_guided/pfgr_lite/static_geometry.py
src/smagm/features/point_guided/pfgr_lite/static_synthesis.py
src/smagm/features/point_guided/pfgr_lite/model.py
src/smagm/features/point_guided/__init__.py
src/smagm/features/point_guided/model.py       # private seam/lazy Gate-E imports only
tests/features/point_guided/pfgr_lite/test_contracts.py
tests/features/point_guided/pfgr_lite/test_base.py
tests/features/point_guided/pfgr_lite/test_provenance.py
```

At that checkpoint W1 runs the three focused tests, the existing frontend
forward test, package `compileall`, `git diff --check`, and the fresh-process
import guard. It must also run a tiny hook-count/gradient fixture for one
traversal and graph-preserving Z0. W2 and W4 may then proceed in parallel on
their exclusive files; W3 starts after W2/W4 contracts are present; W5 starts
after W3/W4 service APIs; the coordinator runs the integrated CPU gates; Astra
performs the independent implementation review. No worker commits another
worker's path, and no implementation worker claims scientific success.

No remaining architecture blocker exists for this bounded implementation once
W1 accepts these declarations and W2 implements the canonical lattice/fallback
contract. Any failure of the focused tests, stale hash, target-boundary leak,
or numerical parity is a blocking software issue requiring an owned fix;
scientific headroom and real/CUDA evidence remain pending by design.
