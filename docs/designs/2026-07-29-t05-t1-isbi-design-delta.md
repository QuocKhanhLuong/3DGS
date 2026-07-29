# T0.5/T1 ISBI Design Delta — Legal Episodes and Fixed Gaussian Attribution

Date: 2026-07-29
Status: Stage-1 architecture contract; implementation blocked by Human Gate 1
Owner: Architect / geometry and medical forward-model lead
Inputs:

- `docs/strategies/2026-07-29-isbi-realignment.md`
- `docs/research/2026-07-29-isbi-full-mechanism-collision.md`
- `docs/protocols/permanently_sparse_training.md`
- `docs/plans/2026-07-29-t05-t1-teacher-free-encoder-fixed-gaussian-baseline.md`

## 1. Decision and scope

This design changes the executable boundary from a permanent
`CONTEXT`/`TARGET` observation label to a fixed sparse availability set plus an
immutable episode assignment. It also makes a registered pre-reveal prediction
a required capability for target access.

After T0.5 passes, T1 tests only this attribution question:

> Does compact teacher-free evidence improve context-to-target reconstruction
> over analytic-only and raw shallow-CNN evidence when observations, episode
> assignments, physical supports, primitive count, renderer, optimization
> opportunity, and compute accounting are matched?

T0 remains infrastructure. T1 uses deterministic supports and fixed-topology
Gaussians as a downstream bridge. This document does not authorize or define
executable placeholders for an anchor-local field, learned Gaussian birth,
anchor–Gaussian propagation, adaptive topology, routing, or full-volume export.

The current renderer is named the:

> **through-plane profile-aware Gaussian reference renderer**

It is not described as `PSF-correct`, `scanner-accurate PSF`, or a `complete
physical MRI forward operator`.

## 2. Current-to-approved delta

| Concern | Current T0 behavior | Approved T0.5/T1 behavior |
|---|---|---|
| Availability and role | `ObservationMeta.access_level` permanently stores `CONTEXT` or `TARGET` | `SparseAvailabilityManifest` stores legal observations; `EpisodeAssignment` stores temporary roles |
| Manifest hash | Includes permanent access level | Availability hash contains no episodic role |
| Target reveal | `commit_target()` immediately returns a reveal capability | Commit returns a commit capability; reveal requires a registered, matching, single-use prediction receipt |
| Budget | Target role and target commitment are coupled to a float-facing budget | Offline episode roles cost zero; a separate deployment ledger uses canonical `Decimal` amounts |
| Support classification | Local support threshold may change under a common log-amplitude shift | Phase-1 uses mean-centered per-patient log amplitude before rendering |
| Encoder coordinates | No feature-grid geometry contract | Every output carries an explicit pixel-center `FeatureGridToPlaneTransform` |
| Gaussian parameters | Public contract validates an already-positive Cholesky factor | T1 constructor maps raw tensors to a valid factor, bounded centers, and gauge-fixed amplitudes |
| Target gradient | No episodic trainer exists | The held prediction tensor remains live for loss/backward; receipt hashing uses a detached audit copy only |

`AccessLevel`, `SparseManifest`, and `ObservationLedger` may remain as explicitly
named T0 migration adapters. No new Phase-1 module may import them as the source
of episode role.

## 3. Coordinate and hashing conventions

All medical geometry remains in canonical RAS millimetres. Homogeneous
transforms multiply column vectors. Plane tensors are `[H,W]`, indexed
`[v,u]`, and the input pixel centre is:

\[
\mathbf x(v,u)=\mathbf o+
u\,\Delta_u\mathbf r+
v\,\Delta_v\mathbf c.
\]

The signed plane normal remains independently validated against the source
affine slice axis. It must not be inferred only from
\(\mathbf r\times\mathbf c\).

Canonical records use:

- UTF-8 JSON;
- sorted object keys;
- separators `(",", ":")`;
- sorted unique observation-ID tuples where set order has no scientific
  meaning;
- canonical decimal strings for acquisition cost;
- SHA-256 over the encoded canonical JSON.

Hashes identify scientific inputs; they are not security claims. Payload
digests remain private to the bound provider and audit machinery and do not
enter target metadata or model inputs.

## 4. T0.5 observation interfaces

### 4.1 Availability metadata and manifest

The new availability record has no training role:

```python
@dataclass(frozen=True)
class AvailabilityObservationMeta:
    observation_id: str
    patient_id: str
    split: str
    relative_path: str
    modality_id: str
    plane: PhysicalPlane
    is_synthetic: bool
    acquisition_cost_key: str | None
    registration_record_id: str | None
```

```python
@dataclass(frozen=True)
class SparseAvailabilityManifest:
    entries: tuple[AvailabilityObservationMeta, ...]
    manifest_id: str
    # Private provider binding, excluded from canonical public metadata.
    integrity_digests: InitVar[Mapping[str, str]]

    @property
    def manifest_hash(self) -> str: ...
    def metadata(self, observation_id: str) -> AvailabilityObservationMeta: ...
```

Required invariants:

- observation IDs and manifest-relative paths are unique;
- paths are non-empty, relative, and cannot escape the bound root;
- every patient has exactly one patient-level split across all manifests;
- every non-synthetic plane retains source-affine provenance;
- manifest entries and public metadata are immutable;
- the canonical manifest hash contains availability and physical provenance,
  but no context/target role and no private content digest;
- changing a role in an episode never changes the availability manifest;
- a payload mismatch, symlink escape, unknown ID, or unbound path fails before
  bytes or a successful open-audit row are returned.

Migration rule: legacy `ObservationMeta.access_level` may be read only by a
compatibility adapter that constructs an explicit assignment. It is excluded
from the new availability hash and cannot be consulted by T1 loaders, caches,
support builders, or trainers.

### 4.2 Immutable episode assignment

```python
@dataclass(frozen=True)
class EpisodeAssignment:
    episode_id: str
    manifest_hash: str
    patient_id: str
    context_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    assignment_hash: str = field(init=False)

    @classmethod
    def create(
        cls,
        manifest: SparseAvailabilityManifest,
        *,
        episode_id: str,
        patient_id: str,
        context_ids: Iterable[str],
        target_ids: Iterable[str],
    ) -> "EpisodeAssignment": ...
```

Construction normalizes IDs to sorted tuples and rejects duplicates. The
canonical assignment payload is exactly:

```json
{
  "context_ids": ["..."],
  "episode_id": "...",
  "manifest_hash": "...",
  "patient_id": "...",
  "target_ids": ["..."]
}
```

`assignment_hash` is SHA-256 of that canonical payload and is recomputed, not
accepted from callers.

Required invariants:

- `episode_id`, `manifest_hash`, and `patient_id` are non-empty;
- the supplied manifest hash equals the bound availability manifest;
- context and target IDs are individually unique and mutually disjoint;
- every ID exists in that manifest;
- every selected observation belongs to `patient_id`;
- tuples and the record are immutable after construction;
- reconstruction episodes require at least one context and one target, enforced
  by the trainer even if the generic record later permits diagnostic
  context-only episodes;
- one manifest may produce any number of differently hashed assignments;
- assignment creation and role changes consume no acquisition budget.

### 4.3 State version

`state_version` is a non-empty canonical SHA-256 identifier for the patient
state used to render a target. For T1 it binds at least:

```text
patient_id
assignment_hash
sorted context observation IDs
opened-context audit hash
encoder/config version
deterministic support-manifest hash
Gaussian-state tensor digest/version
```

It is frozen before target commitment. It contains no target pixels,
target-derived statistic, audit pixel, or revealed target digest.

### 4.4 Episode ledger

```python
class EpisodeLedger:
    def __init__(
        self,
        manifest: SparseAvailabilityManifest,
        assignment: EpisodeAssignment,
        root: Path,
    ) -> None: ...

    def metadata(self, observation_id: str) -> AvailabilityObservationMeta: ...
    def open_context(self, observation_id: str) -> bytes: ...
    def commit_target(
        self,
        target_id: str,
        state_version: str,
    ) -> TargetCommitCapability: ...
    def register_prediction_receipt(
        self,
        commit_capability: TargetCommitCapability,
        *,
        render_evidence: RenderedPredictionCapability,
        target_id: str,
        state_version: str,
        plane_hash: str,
        renderer_version: str,
        prediction_digest: str,
    ) -> PredictionReceiptCapability: ...
    def reveal_target(
        self,
        target_id: str,
        receipt_capability: PredictionReceiptCapability,
    ) -> bytes: ...
```

`open_context()` accepts only an ID in `assignment.context_ids`.
`metadata()` may expose immutable plane and modality metadata for an assigned
target, but never its pixels, labels, content digest, normalization statistic,
or cached features.

The ledger owns a fresh opaque nonce. Commit and receipt capabilities bind that
nonce and an unpredictable secret; their `repr` exposes no secret. Capabilities
are in-process scientific-validity contracts, not an OS security sandbox.

### 4.5 Prediction receipt

A raw caller-provided digest is not evidence that rendering occurred. The
reference path therefore defines:

```python
@dataclass(frozen=True)
class FrozenPatientState:
    state_version: str
    # Opaque, immutable tensor/state binding validated when frozen.
    state_binding: FrozenStateBinding

class RenderedPredictionCapability:
    """Opaque and constructible only by the committed-render adapter."""

def render_committed_target(
    *,
    commit_capability: TargetCommitCapability,
    frozen_state: FrozenPatientState,
    renderer: ThroughPlaneProfileAwareGaussianReferenceRenderer,
) -> tuple[RenderResult, RenderedPredictionCapability]: ...
```

`render_committed_target()` obtains the target plane only from the bound commit,
requires `frozen_state.state_version` to equal the committed version, calls the
registered reference renderer, and mints an opaque evidence capability from
the returned tensors. The capability binds ledger, episode, assignment,
target, state version, plane hash, renderer implementation/config version,
output-schema version, and prediction digest. Its constructor is private to the
adapter; a raw string, direct `RenderResult`, or caller-built lookalike is
rejected.

`register_prediction_receipt()` consumes this renderer-minted capability and
checks every explicit scalar argument against it. This preserves the externally
auditable receipt fields while preventing an arbitrary well-formed digest or
renderer-version string from satisfying the barrier. The capability is a
scientific-validity mechanism inside the reference process, not a claim of
protection against hostile code or operating-system compromise.

The auditable receipt record binds:

```python
@dataclass(frozen=True)
class PredictionReceiptRecord:
    ledger_id: str
    episode_id: str
    assignment_hash: str
    target_id: str
    state_version: str
    plane_hash: str
    renderer_version: str
    prediction_digest: str
    commit_sequence: int
    receipt_sequence: int
```

`plane_hash` is recomputed by the ledger from the manifest-bound target
`PhysicalPlane.canonical_json()`. The caller-supplied value must match it.

`prediction_digest` is computed after rendering from a detached, canonical
prediction envelope containing:

```text
intensity tensor shape, dtype, byte order, and contiguous bytes
unsupported mask shape and contiguous bytes
supported through-plane profile mass
plane hash
renderer version
renderer output-schema version
```

Canonicalization transfers a detached copy to CPU, records original dtype and
shape, serializes finite values in contiguous little-endian order, maps every
NaN payload to one fixed quiet-NaN bit pattern, and serializes boolean masks as
contiguous `0/1` bytes. Device layout, non-contiguous strides, or platform byte
order therefore cannot silently change the scientific digest.

The digest operation must not replace or detach the prediction tensor retained
by the trainer for reconstruction loss.

Exact legal sequence:

```text
open assigned context only
→ freeze state_version
→ expose assigned target metadata
→ commit_target(target_id, state_version)
→ render_committed_target(commit, frozen state)
→ renderer adapter computes digest from a detached audit copy
→ register_prediction_receipt(..., render_evidence)
→ reveal_target(target_id, receipt_capability)
→ compute loss using the original live prediction tensor
```

Registration rejects:

- a missing or invalid commit capability;
- missing, forged, consumed, or caller-constructed render evidence;
- a commit from another ledger;
- a target, episode, assignment, or state version different from the commit;
- a plane hash different from bound target metadata;
- any renderer version, output-schema version, or prediction digest different
  from the renderer-minted evidence;
- a second receipt for the same commit.

Reveal rejects:

- commit without a receipt;
- a receipt from another target, episode, assignment, state, or ledger;
- a receipt with a mismatched target-plane hash;
- a forged, revoked, or previously consumed receipt;
- a second reveal of the same target.

Successful reveal atomically consumes both commit and receipt capabilities.
The event stream has deterministic sequence numbers and records
`OPEN_CONTEXT`, `COMMIT_TARGET`, `REGISTER_PREDICTION`, and `REVEAL_TARGET`
before the payload-open audit row. Failed attempts do not create a successful
open row.

## 5. Episode role versus deployment acquisition cost

`EpisodeLedger` has no budget, cost counter, or charging path. Context/target
assignment, context opens, and legal sparse target reveals cost exactly zero in
offline training.

Deployment uses a separate interface:

```python
@dataclass(frozen=True)
class AcquisitionCostEntry:
    cost_key: str
    canonical_amount: str

@dataclass(frozen=True)
class AcquisitionCostSchedule:
    schedule_id: str
    entries: tuple[AcquisitionCostEntry, ...]
    schedule_hash: str = field(init=False)

    @classmethod
    def create(
        cls,
        *,
        schedule_id: str,
        amounts: Mapping[str, str],
    ) -> "AcquisitionCostSchedule": ...

    def amount(self, cost_key: str) -> Decimal: ...

class DeploymentAcquisitionLedger:
    def __init__(
        self,
        *,
        budget: Decimal,
        schedule: AcquisitionCostSchedule,
    ) -> None: ...

    def commit_bootstrap(self, observation_id: str) -> AcquisitionCapability: ...
    def commit_observation(self, observation_id: str) -> AcquisitionCapability: ...
```

`create()` rejects mutable storage in the resulting object, blank or duplicate
keys, floats, non-finite values, negative values, and non-canonical decimal
strings. It parses input strings directly as `Decimal`, normalizes them to one
canonical string representation, sorts entries by key, freezes the tuple, and
computes `schedule_hash` internally from `schedule_id` plus the canonical
entries. The caller cannot supply a hash, and later mutation of the source
mapping cannot change the schedule.

A successful irreversible deployment commit charges:

- every bootstrap observation;
- every later committed observation;
- modality, orientation, thickness, or protocol-specific cost selected by its
  immutable cost key.

The cost event records budget before/after, canonical amount, observation ID,
modality/plane cost key, schedule hash, and ledger hash. Rejected commitments
do not charge. A successful commitment remains charged if rendering or later
processing fails.

Legacy float-facing `ObservationMeta.cost` and `target_budget` are compatibility
only and are not legal inputs to new deployment accounting.

## 6. Amplitude gauge and coverage policy

T0.5 selects policy A:

> **`MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE`**

The transformation has exactly one ownership boundary:

> **Gaussian-state construction owns gauge fixing; the renderer never applies
> or reapplies an amplitude gauge.**

T0.5 introduces a pure, differentiable
`fix_log_amplitude_gauge(raw_log_amplitude, patient_state_index, policy)`
utility and a typed `GaugeFixedLogAmplitude` result. T0.5 tests this utility with
the existing `GaussianBatch`/renderer path before the T1 safe constructor
exists. In T1, `construct_fixed_gaussians()` becomes the only main-path caller
of that utility and passes the resulting gauge-fixed log amplitudes to
`GaussianBatch.log_support_amplitude`. The renderer accepts already fixed log
amplitudes plus their policy/config provenance, performs its existing single
log-to-positive-amplitude conversion, rejects an unfixed Phase-1 state, and
performs no centering or other gauge normalization itself.

For one patient state with \(N\) primitive raw log amplitudes
\(\ell_i^{raw}\):

\[
\tilde\ell_i =
\ell_i^{raw}
- \frac{1}{N}\sum_j \ell_j^{raw}.
\]

The safe constructor may apply a symmetric dtype-safe saturation to the
centered values:

\[
\ell_i =
L_{max}\tanh(\tilde\ell_i/L_{max}),
\]

where `L_max` is named in config and identical across matched experiments.
Because the centering happens first, adding one constant to every raw log
amplitude produces exactly the same \(\ell_i\), intensity, raw rendered support
diagnostic, and supported/unsupported classification.

The transform is computed jointly over the complete Gaussian batch for one
patient state. Patient states may not be mixed in a single gauge group. If
future batching concatenates patients, a required `patient_state_index`
per primitive defines separate segment means.

`LEGACY_RAW` may exist only for T0 compatibility tests and cannot be selected
by Phase-1/T1 configs. The state builder records the policy and config hash in
the `GaussianBatch` provenance, state version, and artifact record. A blocking
test asserts that the renderer neither changes a `GaugeFixedLogAmplitude` nor
invokes the gauge utility a second time.

Even after gauge fixing:

- `support_mass` is a support diagnostic, not calibrated uncertainty;
- it is not a routing score;
- unsupported masking remains explicit;
- supported-only metrics cannot be headline metrics;
- coverage/failure is reported with every reconstruction metric.

Required regression: render one Gaussian state twice with raw log amplitudes
\(\ell\) and \(\ell+c\), for positive and negative finite `c`, in float32 and
float64. Gauge-fixed amplitudes, intensity, support diagnostics, through-plane
supported mass, and `unsupported_mask` must match within the declared numeric
tolerance. A legacy-raw control must demonstrate that the test would detect the
old gauge dependence.

## 7. T1 feature contracts

### 7.1 Analytic differential bank

Recommended module: `src/smagm/features/analytic.py`.

```python
@dataclass(frozen=True)
class AnalyticFeatureConfig:
    padding: Literal["replicate"]
    local_contrast_radii_mm: tuple[float, float]
    derivative_units: Literal["intensity_per_mm"]
    epsilon: float
    include_artifact_cue: bool

@dataclass(frozen=True)
class AnalyticFeatureOutput:
    channels: Tensor          # [B,C_phi,H,W]
    channel_names: tuple[str, ...]
    valid_mask: Tensor        # [B,1,H,W], bool
    config_hash: str
    runtime_seconds: float
```

Initial channel order is fixed:

```text
normalized_intensity
d_u
d_v
gradient_magnitude
laplacian
local_contrast_r1
local_contrast_r2
valid_mask
optional analytic reliability/artifact cue
```

`d_u` and `d_v` are derivatives along the plane `axis_u_ras` and
`axis_v_ras`, divided by `spacing_uv_mm`, so their declared units are normalized
intensity/mm. The Laplacian accounts for both in-plane spacings. Fixed
convolutions use explicit replicate padding; no framework default is accepted.
Local contrast is normalized intensity minus a spacing-aware local mean at two
declared physical radii.

Normalization statistics are fitted from legal training context only and are
recorded. Target or audit pixels cannot contribute. Constant and low-signal
images must produce finite channels. Runtime and analytic operation accounting
are included in every encoder variant, including E0.

Blocking independent references cover:

- constant, ramp, impulse, and low-signal images;
- translation/equivariance on the valid interior;
- explicit boundary values under replicate padding;
- local contrast at both scales;
- float32/float64 forward and finite gradient behavior.

### 7.2 Feature-grid-to-plane transform

Recommended module: `src/smagm/features/contracts.py`.

```python
@dataclass(frozen=True)
class FeatureGridToPlaneTransform:
    input_plane: PhysicalPlane
    output_stride: Literal[1, 2, 4]
    feature_shape_hw: tuple[int, int]
    sampling_convention: Literal["half_pixel_align_corners_false"]
    valid_feature_shape_hw: tuple[int, int]
    valid_feature_mask_hash: str

    def input_vu_from_feature_vu(self, vf: Tensor, uf: Tensor) -> tuple[Tensor, Tensor]: ...
    def feature_vu_from_input_vu(self, v: Tensor, u: Tensor) -> tuple[Tensor, Tensor]: ...
    def ras_mm_from_feature_vu(self, vf: Tensor, uf: Tensor) -> Tensor: ...
    def grid_sample_coordinates(self, ras_mm: Tensor) -> Tensor: ...
```

For stride \(s\in\{1,2,4\}\), feature centre \((v_f,u_f)\) maps to input
continuous pixel coordinates:

\[
u=(u_f+\tfrac12)s-\tfrac12,\qquad
v=(v_f+\tfrac12)s-\tfrac12.
\]

The inverse is:

\[
u_f=(u+\tfrac12)/s-\tfrac12,\qquad
v_f=(v+\tfrac12)/s-\tfrac12.
\]

Therefore the feature-plane pixel-centre origin is:

\[
\mathbf o_f=\mathbf o+
\frac{s-1}{2}\Delta_u\mathbf r+
\frac{s-1}{2}\Delta_v\mathbf c,
\]

with feature spacing \((s\Delta_u,s\Delta_v)\). Stride 1 is exactly the input
pixel grid; stride 2 begins at input coordinate `(0.5,0.5)`; stride 4 begins at
`(1.5,1.5)`.

Sampling uses `torch.grid_sample(..., align_corners=False)` with:

\[
x_{norm}=2(u_f+\tfrac12)/W_f-1,\qquad
y_{norm}=2(v_f+\tfrac12)/H_f-1.
\]

Downsampling must implement this same half-pixel convention, for example an
explicit alignment-tested factor-2 low-pass/downsample transition. A
stride-2 convolution whose receptive-field centre follows another convention
is not permitted without returning the corresponding different transform.
Odd input sizes require explicit right/bottom padding and a propagated
`valid_feature_mask: Tensor[B,1,H_f,W_f]`. The transform records both its valid
shape and mask hash; the actual mask is returned with encoder output and passed
to feature sampling/support validation. Padded support cannot become a legal
deterministic support point.

Blocking tests use independently calculated RAS landmarks at corners and
interior points for strides 1/2/4, rigid RAS frame changes, odd shapes, and
half-pixel positions. A transform derived from feature tensor shape alone is
invalid.

### 7.3 Micro-CNN output

Recommended modules:

- `src/smagm/features/encoder.py`
- `src/smagm/features/contracts.py`

```python
@dataclass(frozen=True)
class EvidenceEncoderOutput:
    z_str: Tensor                         # [B,C_str,H_f,W_f]
    z_app: Tensor                         # [B,C_app,H_f,W_f]
    reliability: Tensor | None            # [B,1,H_f,W_f]
    valid_feature_mask: Tensor            # [B,1,H_f,W_f], bool
    feature_to_plane: tuple[FeatureGridToPlaneTransform, ...]
    encoder_version: str
    compute_record: EncoderComputeRecord
```

Reference envelope:

```text
analytic or raw channels + small modality condition
→ 3x3 high-resolution stem
→ two depthwise residual blocks
→ zero, one, or two consecutive alignment-tested factor-2 transitions
→ two depthwise residual blocks
→ projections: Z_str≈16, Z_app≈8, reliability≈1 optional
```

Stride 1 uses zero transitions, stride 2 uses one, and stride 4 uses two
consecutive copies of the same declared half-pixel factor-2 transition before
the final two residual blocks. Each transition propagates the valid mask and
composes its transform, so stride 4 is not inferred from tensor shape or an
undeclared receptive-field convention.

The trunk is shared across modalities. A small embedding, FiLM, or
normalization condition is allowed. Output stride is only 1, 2, or 4. No
U-Net, Transformer, teacher, large decoder, direct Gaussian topology output, or
patient-global reasoning is permitted.

Only `Z_str` participates in spatial, intensity, and registered
cross-modality alignment. `Z_app` is not forced to match. Registration
confidence and valid physical overlap are explicit tensors; missing or
low-confidence registration disables the pair rather than creating a negative
pair.

## 8. Cache lifetimes

Recommended module: `src/smagm/features/cache.py`.

### Training cache

`EpisodeForwardCache`:

- exists only within one episode forward;
- retains autograd-connected tensors;
- is never reused after an optimizer update or a new encoder forward;
- is keyed internally by assignment hash, observation ID, encoder version,
  analytic-config hash, normalization hash, and a unique forward token;
- contains context observations only before reveal;
- cannot serialize target pixels or target-derived features.

### Inference cache

`DetachedInferenceFeatureCache`:

- is permitted only when global encoder weights are frozen;
- stores detached tensors;
- is keyed by patient ID, observation ID, encoder-version hash, opened-content
  digest, plane hash, normalization hash, analytic-config hash,
  feature-transform hash, dtype, and device representation;
- rejects any key mismatch instead of falling back to another version;
- has a separate namespace from training and audit caches;
- encodes each committed deployment observation once.

Content digest is an internal key obtained only after a legal context or
committed deployment open. It is never exposed in target metadata or used to
choose an unrevealed target.

## 9. Deterministic fixed support

Recommended module: `src/smagm/baselines/fixed_support.py`.

Before running E0/E1/E2, construct and hash one `FixedSupportManifest` from
episode context plane metadata and the common valid mask:

```python
@dataclass(frozen=True)
class FixedSupportPoint:
    support_id: str
    source_observation_id: str
    source_plane_hash: str
    source_vu: tuple[float, float]
    position_ras_mm: tuple[float, float, float]

@dataclass(frozen=True)
class FixedSupportManifest:
    assignment_hash: str
    points: tuple[FixedSupportPoint, ...]
    selection_config_hash: str
    support_manifest_hash: str
```

The reference selector is a deterministic physical lattice on each context
plane, clipped to valid-content support, with declared physical spacing and
lexicographic tie-breaking. A deterministic Poisson-disc alternative is
allowed only with a fixed seed derived from assignment/config hashes and exact
parity across encoder variants.

Required invariants:

- support selection never reads encoder values, target pixels, labels, audit
  data, reconstruction residuals, or learned reliability;
- E0/E1/E2 use the identical serialized support manifest;
- support count and per-plane allocation are identical across variants;
- every point has signed distance zero to its source plane within the
  dtype/geometry tolerance;
- RAS position is independently recomputed from `source_vu` and the bound
  source plane;
- each sampled feature coordinate is obtained only through the explicit
  `FeatureGridToPlaneTransform`;
- invalid or padded feature support is rejected, not silently zero-filled;
- source observation ID and plane hash survive into Gaussian provenance.

Stride-1/2/4 and half-pixel tests compare `grid_sample` values against an
independent analytic ramp evaluated at the same physical RAS points.

## 10. Safe fixed Gaussian constructor

Recommended module: `src/smagm/baselines/fixed_gaussian.py`.

### 10.1 Shared feature-to-parameter bridge

Every E0/E1/E2 variant must first satisfy one common sampled-evidence contract:

```python
@dataclass(frozen=True)
class SampledSupportFeatures:
    z_str: Tensor               # [N,16]
    z_app: Tensor               # [N,8]
    reliability: Tensor         # [N,1]; deterministic zero if disabled
    support_manifest_hash: str
    valid_sample_mask: Tensor   # [N], bool
```

E0 uses a named fixed, non-learned selection/normalization/zero-padding map from
the analytic bank into the `16 + 8 + 1` contract. E1 and E2 emit those same
dimensions directly. Invalid or padded samples are rejected before the bridge;
they are not represented by zeros in a valid row.

One shared downstream architecture maps these features to raw Gaussian
parameters:

```python
class FixedGaussianParameterHead(nn.Module):
    def forward(
        self,
        features: SampledSupportFeatures,
    ) -> RawFixedGaussianParameters: ...
```

The reference head is a pointwise two-layer MLP or equivalent `1x1` projection
with fixed hidden width and four named projections:

```text
concat(Z_str, Z_app, reliability)
→ shared pointwise hidden representation
├── raw local centre offsets
├── raw lower-triangular factor entries
├── raw log amplitude
└── raw modality appearance
```

It has no support-to-support communication and cannot change topology. E0, E1,
and E2 use identical head architecture, hidden width, output dimensions,
initialization seed, optimizer, steps, and update opportunity. Runs train
separate head instances from the same initialization; weights are not copied
from a privileged variant. E1 and E2 encoder widths are selected under the
predeclared matched parameter/FLOP envelope. E0's intentionally absent learned
encoder is reported transparently rather than hidden with unused parameters.
Head and encoder parameter/FLOP/runtime counts are reported separately and in
total.

The required differentiable sequence is therefore:

```text
encoder feature grids
→ FeatureGridToPlaneTransform + valid_feature_mask
→ SampledSupportFeatures
→ FixedGaussianParameterHead
→ RawFixedGaussianParameters
→ safe constructor
→ GaussianBatch
```

Blocking tests perturb each sampled feature family and prove that at least one
declared raw parameter changes, then prove non-null renderer-loss gradients for
the head and, for E1/E2, the encoder. A constructor that ignores sampled
features cannot pass T1-A.

### 10.2 Validity-preserving constructor

```python
@dataclass(frozen=True)
class RawFixedGaussianParameters:
    local_center_offset: Tensor   # [N,3]
    lower_triangle_raw: Tensor    # [N,3,3]
    log_amplitude_raw: Tensor     # [N,1]
    appearance_raw: Tensor        # [N,M]
    appearance_valid: Tensor      # [N,M], bool

def construct_fixed_gaussians(
    supports: FixedSupportManifest,
    raw: RawFixedGaussianParameters,
    *,
    max_center_offset_mm: tuple[float, float, float],
    factor_diagonal_epsilon: float,
    covariance_epsilon: float,
    amplitude_gauge_policy: Literal[
        "MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE"
    ],
) -> GaussianBatch: ...
```

Construction order is fixed:

```text
raw tensors
→ project covariance input with torch.tril
→ replace diagonal with softplus(raw diagonal) + epsilon
→ bound local centre offsets with tanh and declared mm limits
→ map offsets through source-plane RAS basis
→ call the canonical gauge utility exactly once
→ attach appearance validity and fixed-support provenance
→ construct and revalidate GaussianBatch
```

The support-plane basis is
`[axis_u_ras, axis_v_ras, signed_normal_ras]`. Bounds are in millimetres and
are explicit per local basis direction. The constructor never optimizes or
mutates a previously validated Cholesky factor directly.

Every raw tensor operation remains differentiable. Metadata and support IDs are
discrete. T1 has no Gaussian split, merge, prune, learned birth, propagation,
adaptive topology, or SDF field.

Tests assert:

- the factor is lower triangular with strictly positive diagonal;
- covariance is SPD for adversarial finite raw inputs;
- centres remain within declared local bounds;
- a common raw log-amplitude shift produces the same gauge-fixed state;
- support and primitive provenance are unchanged;
- all continuous raw inputs receive finite non-null gradients when relevant to
  the selected loss.

## 11. Renderer-to-encoder autograd path

The required live tensor path is:

```text
masked sparse target reconstruction loss
→ original RenderResult.intensity tensor
→ through-plane profile-aware Gaussian reference renderer
→ Gaussian appearance, centre, factor, and gauge-fixed amplitude
→ safe fixed Gaussian constructor
→ FixedGaussianParameterHead
→ differentiable feature sampling with grid_sample
→ Z_str / Z_app
→ shared micro-CNN
→ fixed analytic tensor bank
→ legal context tensor
```

The target payload, physical metadata, ledger, assignment, support selection,
receipt digest, event log, unsupported mask decision, and file I/O are discrete
contracts. The prediction used for hashing is a detached audit copy; the
prediction used for loss is the original tensor.

The end-to-end gate must show:

- non-null finite encoder parameter gradients;
- non-null finite gradients through sampled features and raw Gaussian
  constructor parameters;
- float64 gradcheck on a reduced synthetic case;
- chunked/un-chunked forward and backward parity;
- no target byte sentinel in any pre-reveal tensor, state, or cache.

Unsupported intensity remains `NaN`. The objective forms:

```text
loss_mask = target_valid_mask & ~render.unsupported_mask
```

and logs supported numerator/denominator, unsupported fraction, and total
coverage. It never replaces unsupported intensity with a confident zero.
Headline evaluation reports coverage/failure alongside full declared metric
treatment; a supported-only metric cannot establish Gate T1-R.

## 12. Structural losses and objective boundary

Recommended module: `src/smagm/losses/structural.py`.

Each term is independently switchable and returns both scalar loss and named
diagnostics:

- spatial equivariance of `Z_str`;
- intensity invariance of `Z_str`;
- registered cross-modality `Z_str` consistency with explicit registration
  confidence and valid-overlap mask;
- per-channel variance floor;
- off-diagonal channel covariance penalty;
- local differential preservation;
- masked sparse target-plane reconstruction.

`Z_app` is never passed to the cross-modality alignment loss. Per-channel
standard deviation, covariance, finite fraction, and active registration-pair
count are logged. Auxiliary feature diagnostics cannot pass T1 without
improved sparse target reconstruction.

## 13. Module ownership and file boundary

| Owner | Stage | Files / responsibility |
|---|---|---|
| Architect | T0.5/T1 design | Contracts in this document; geometry, hashes, gauge, cache, and constructor semantics |
| Medical Data Steward | T0.5/T1 | Manifest creation, patient split, registration confidence, missing modality, audit isolation, acquisition-cost schedule, leakage veto |
| Dev | T0.5 | `src/smagm/contracts/episode.py`, observation/manifest migration, renderer-minted evidence, canonical gauge utility/provenance, exports |
| QA | T0.5 | Assignment, receipt, wrong-capability, gauge, Decimal, leakage-positive, and compatibility tests |
| Reproducibility Auditor | T0.5 | Exact clean commit, CI, manifest/assignment/config/event/artifact hashes |
| Reviewer | T0.5 | Independent legality, physical geometry, receipt ordering, gauge, and overclaim review |
| Dev | T1-A | Analytic bank, feature transform, fixed support, safe constructor |
| QA | T1-A | Analytic references, alignment, provenance, SPD, gauge, and end-to-end autograd |
| Dev | T1 learned | Micro-CNN, cache, switchable losses, episodic trainer and objective |
| Experiment Lead | T1 | Immutable matched episode/support manifests; E0/E1/E2 configs, seeds, FLOPs, steps, hardware accounting |
| Reviewer | T1 | Reconstruction validity, lesion/ROI fidelity, sparse legality, geometry, and comparison fairness |

Every new implementation module updates `CHANGELOG.md` in the same tranche.
No owner may add T2+ scaffolding “for later”.

## 14. Blocking tests and gates

### Gate T0.5-L

Blocking suites must prove:

1. one availability manifest produces multiple legal immutable assignments;
2. overlapping, duplicate, unknown, cross-patient, or wrong-manifest IDs fail;
3. role reassignment does not mutate manifest/hash or consume cost;
4. commit alone cannot reveal;
5. wrong target, episode, assignment, state, ledger, plane, arbitrary digest,
   forged/missing renderer evidence, and reused receipt capabilities fail;
6. event order is commit → prediction registration → reveal → payload open;
7. offline training has no cost ledger;
8. deployment bootstrap and later commitments use an immutable, internally
   hashed schedule and exact `Decimal` cost;
9. common log-amplitude shifts cannot change supported classification, and the
   gauge utility is applied exactly once before `GaussianBatch` construction;
10. non-manifest/audit/symlink/mutated payload and target-sentinel
    leakage-positive controls fail closed;
11. exact clean-commit CI passes pytest, compileall, and diff checking.

T1 code must not start until this gate passes, an independent reviewer accepts
it, and Human Gate 2 explicitly approves Stage 3.

### Gate T1-F

Blocking suites must prove:

- analytic constant/ramp/impulse/boundary/translation and float32/float64
  behavior;
- pixel-centre, stride-1/2/4, half-pixel, odd-shape, and RAS-rigid-transform
  alignment;
- propagated valid masks exclude every padded feature/support sample;
- finite per-channel anti-collapse statistics;
- registered matches are better than mismatches under declared confidence;
- local differential cues remain recoverable;
- `Z_app` is not accidentally aligned.

### Gate T1-R

E2b must improve E0 and E1 with the same patients, manifest, assignment,
supports, primitive count, profile, gauge, optimizer opportunity, steps, and
hardware accounting. The downstream feature-to-parameter head is identical
across variants and initialized under the same seed/opportunity. Analytic
preprocessing is included in compute. Proxy feature improvement alone fails.

### Gate T1-M

On the patient-disjoint **T1 lesion-validation audit cohort**, any global gain
must not violate a predeclared paired non-inferiority margin for the primary
lesion/ROI and boundary-fidelity estimands. The cohort is evaluated once after
the checkpoint, config, sparse input manifest, masks, estimands, margins,
confidence interval, and multiplicity policy are frozen. Its dense pixels and
labels remain evaluator-only and never enter training, normalization,
registration fitting, support selection, early stopping, or checkpoint
selection. Coverage and failures are reported.

This T1 gate cohort is not the sealed final-audit cohort reserved for T5. Each
has disjoint patients, storage roots, credentials, loader/cache namespaces, and
hashes. T1 reconstruction receives only its predeclared sparse input manifest;
the isolated evaluator alone opens dense targets and lesion/ROI labels.

No T2 work begins until the human accepts all applicable gates.

## 15. Stop rules

- `E0 ≈ E2b`: remove the learned encoder from the novelty path.
- `E1 ≈ E2b`: remove the analytic-scaffold claim.
- Auxiliary losses improve diagnostics but not reconstruction: keep them only
  as diagnostics.
- Fixed Gaussian reconstruction does not beat interpolation: stop before
  propagation and repair data, state, geometry, coverage, or renderer
  assumptions.
- No competitive static representation baseline: do not implement routing.
- A scalar field without sign convention, Eikonal test, gradient-norm
  statistics, and distance calibration is `StructuralField`, never SDF.
- Any non-manifest or audit pixel in training invalidates the run.

Every `≈` stop decision uses a preregistered paired equivalence interval and
margin, not overlapping point estimates. T1-M likewise preregisters its
patient-level estimands, non-inferiority margins, confidence intervals, and
multiplicity handling before the evaluator is opened.

## 16. Execution contracts for later gated work

These contracts describe authorized handoffs after human approval. They do not
authorize implementation from this Stage-1 document.

### Stage 2 execution contract — T0.5 only

```yaml
stage: 2
precondition:
  - Human Gate 1 approved
inputs:
  - authoritative ISBI strategy
  - permanently sparse protocol
  - this design delta
write_scope:
  - src/smagm/contracts/episode.py
  - src/smagm/contracts/observation.py
  - src/smagm/data/manifest.py
  - src/smagm/gaussians.py
  - src/smagm/renderer.py
  - src/smagm/__init__.py
  - tests/contracts/test_episode_assignment.py
  - tests/integration/test_prediction_receipt_barrier.py
  - tests/render/test_support_gauge.py
  - .github/workflows/ci.yml
  - CHANGELOG.md
required_outputs:
  - immutable availability and assignment contracts
  - receipt-gated episode ledger
  - separate exact-Decimal deployment accounting
  - named gauge utility applied once at Gaussian-state construction
  - renderer-minted prediction evidence
  - leakage-positive blocking tests
verification:
  - python -m pytest -q
  - python -m compileall -q src tests
  - git diff --check
exit:
  - independent T0.5 review
  - exact clean commit and CI evidence
  - stop at Human Gate 2
prohibited:
  - any T1 implementation
  - any T2+ file or placeholder
```

### Stage 3 execution contract — T1-A synthetic contracts

```yaml
stage: 3
precondition:
  - Gate T0.5-L passed
  - Human Gate 2 approved
inputs:
  - immutable matched synthetic EpisodeAssignments
  - T0.5 receipt ledger
  - fixed renderer profile and gauge config
write_scope:
  - src/smagm/features/analytic.py
  - src/smagm/features/contracts.py
  - src/smagm/baselines/fixed_support.py
  - src/smagm/baselines/fixed_gaussian.py
  - corresponding synthetic tests
  - CHANGELOG.md
required_outputs:
  - analytic differential bank
  - explicit feature-grid transform
  - deterministic physical support manifest
  - shared fixed feature-to-parameter head
  - safe fixed Gaussian constructor
  - renderer-to-feature synthetic autograd test
verification:
  - full T0.5 suite remains green
  - analytic independent-reference tests
  - stride 1/2/4 and half-pixel tests
  - SPD, gauge, provenance, and float64 autograd tests
  - python -m compileall -q src tests
  - git diff --check
exit:
  - independent T1-A review
  - stop at Human Gate 3
prohibited:
  - micro-CNN training, learned losses, or experiment claims
  - any T2+ file or placeholder
```

### Stage 4 execution contract — T1 learned components and smoke

```yaml
stage: 4
precondition:
  - Stage 3 synthetic contracts passed
  - Human Gate 3 approved
inputs:
  - frozen matched assignment and support manifests
  - verified feature transform and safe constructor
write_scope:
  - src/smagm/features/encoder.py
  - src/smagm/features/cache.py
  - src/smagm/losses/structural.py
  - src/smagm/training/episode.py
  - src/smagm/training/objective.py
  - scripts/train_t1.py
  - configs/t1/common.yaml
  - configs/t1/e0_analytic.yaml
  - configs/t1/e1_raw_cnn.yaml
  - configs/t1/e2_teacher_free.yaml
  - corresponding tests
  - CHANGELOG.md
required_outputs:
  - E0, E1, E2a, and E2b executable paths
  - shared micro-CNN with modality condition
  - switchable structural losses and anti-collapse logs
  - episode/inference cache semantics
  - receipt-enforced episodic trainer
  - bounded synthetic or manifest-legal smoke run
verification:
  - all T0.5 and T1-A tests remain green
  - no-collapse and registered-match diagnostics
  - target-sentinel leakage control
  - encoder autograd and cache-version tests
  - matched support/primitive/profile/step assertions
  - full pytest, compileall, and diff checking
exit:
  - smoke report with coverage and failures
  - independent review
  - stop at Human Gate 4 before matched experiments
prohibited:
  - E3/E4 described as the main method
  - any T2+ file or placeholder
```

### Stage 5 execution contract — matched T1 decision

```yaml
stage: 5
precondition:
  - Stage 4 smoke accepted
  - Human Gate 4 approved
inputs:
  - frozen patient, manifest, assignment, support, renderer, and optimizer records
  - predeclared T1 lesion-validation sparse input manifest
  - sealed final-audit cohort remains unopened
ownership:
  experiment_lead:
    - configs/t1/**
    - experiments/**
    - docs/experiments/**
  qa:
    - tests/**
required_outputs:
  - paired E0/E1/E2a/E2b results under matched downstream conditions
  - E3 frozen-pretrained upper bound, separately labeled
  - E4 teacher-distilled privileged upper bound, separately labeled
  - parameter, FLOP, runtime, VRAM, coverage, failure, and patient-level statistics
  - T1 lesion-validation evaluation under frozen margins and multiplicity policy
  - reviewer PASS/PARTIAL/FAIL decision
exit:
  - stop after the T1 decision
  - no T2 authorization without a new human decision
prohibited:
  - E3/E4 main-method wording
  - tuning from the T1 lesion-validation evaluator
  - opening the sealed T5 final-audit cohort
  - any T2+ implementation
```

## 17. Stage-1 architecture recommendation

**PARTIAL — design-ready, implementation not authorized.**

The interfaces above close the known permanent-role, immediate-reveal,
amplitude-gauge, feature-coordinate, validated-factor, and autograd ambiguities.
They do not establish reconstruction benefit or medical fidelity. Proceed only
to Stage 2 after Human Gate 1, and stop again at every declared gate.
