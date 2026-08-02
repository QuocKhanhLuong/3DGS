# Structure-Constrained Interior Gaussian Propagation

Status: canonical method specification for the BraTS21 product-readiness
tranche, 2026-08-02. Software evidence and synthetic tests do not constitute a
scientific or clinical pass. T4 routing and adaptive acquisition are absent and
blocked.

## 1. Problem definition

The method reconstructs registered multi-sequence MRI in physical RAS-mm
coordinates from a declared sparse set of observed planes. BraTS21 supplies
dense registered source volumes only to construct a **simulated
sparse-acquisition reconstruction task**. The maintained initial episode has
one held-out target plane and a configurable target modality.

The method is not ordinary camera-view 3D Gaussian Splatting. Its renderer is a
through-plane-profile-aware physical-volume query operator that returns an
intensity, support mask, unsupported mask, and support-derived uncertainty.

## 2. Simulated sparse acquisition

The source affine, shape, spacing, and declared axial orientation define legal
plane geometry. Dense source values are not used to choose planes. The initial
protocol uses five physical axial positions shared by T1, T1CE, T2, and FLAIR,
with positions selected by quantiles 0.15 through 0.85. Training may use
seeded jitter up to 0.15 of local inter-plane spacing; validation and test use
zero jitter. The first ablation set is sparse-3/5/7/9, aligned versus
staggered, axial-only. In the optional staggered protocol, target-gap selection
is still made between two context planes of the target modality, and a midpoint
that coincides with any context plane from another modality is rejected.

## 3. Geometry-only context sampling

Sampling consumes only validated header geometry and an episode identity. It
does not inspect target intensity, segmentation, reconstruction error, edge
density, tumor location, or hidden full-volume values. The default target
policy selects an interval between adjacent context positions deterministically
from the episode seed and sets target geometry to the exact physical midpoint of
that gap. For aligned sampling this is the shared modality gap; for staggered
sampling the legal gap is selected from the target modality's context positions
and is also checked against the union of all context planes. The target plane
therefore carries a fractional source-slice index
when the midpoint falls between source samples; the nearest integer index is
retained only as a provenance reference. Hidden intensity extraction uses
bounded linear interpolation after receipt, while evaluator-only segmentation
uses nearest-neighbor extraction. The midpoint, fractional index, and resolved
plane geometry are hashed in the sampling protocol record.

## 4. Legal target isolation

Context payloads are decoded through the ledger and are the only inputs to
preprocessing, encoding, candidate scoring, anchors, fields, Gaussian seeds,
and propagation. Target geometry may be exposed before rendering. Target
intensity and evaluator segmentation bytes are not materialized by product
preparation; source/geometry references are bound. The deferred target reader
is reachable only after prediction receipt registration, and the deferred
segmentation reader is opened later for evaluator packaging after prediction
serialization.
Segmentation is
evaluator-only and is never used for sampling, normalization, state
construction, propagation, or training loss.
The completed dataset inventory may perform a separate full finite-value and
source-hash pass for provenance before episodes are scheduled; those values are
not an input to sampling or patient-state construction. Product preparation
consumes those recorded inventory hashes and hashes only the materialized
context payloads; it must not re-hash source NIfTI files during preparation or
prepared-bundle reuse because a full-file pass would open hidden target or
evaluator bytes before the receipt barrier. Deferred target and segmentation
readers re-check their bound source hashes only after their respective legal
access barriers.

## 5. Multi-sequence 2D evidence

Each context observation retains its modality, image payload, valid mask,
source affine, plane basis, pixel spacing, observation hash, and normalization
record. The teacher-free E2 path combines analytic structural evidence with the
shared shallow micro-CNN. Evidence includes appearance, boundary/orientation,
reliability, and observability channels. A feature pixel is a candidate, not
automatically an anchor.

## 6. Physical lifting

Candidate feature coordinates are mapped through the source affine and plane
basis to RAS-mm positions. Anisotropic spacing, non-identity affine, source
orientation, and supported oblique plane bases remain explicit. Physical NMS and
cross-plane consolidation operate in millimetres, never in a global voxel or
global-z proxy.

## 7. Candidate cloud versus anchor set

The candidate cloud is a scored, possibly redundant set of legal evidence
locations. The anchor set is the deterministic, budgeted, physically
non-maximum-suppressed representation used by the downstream state. Each
anchor binds provenance observations, compact multi-sequence evidence,
confidence/observability, support scales, modality-valid appearance, and a
patient/config/state identity hash. The anchor digest covers the physical
centre, complete orthonormal local frame and validity flags, support scales,
confidence/disagreement, contributing observations/planes, and all evidence
and observability tensors; changing any of those fields invalidates the
immutable state binding.

## 8. Anchor local frames

Every anchor stores two normalized tangent directions `t1`, `t2`, and a
normalized local normal `n` in RAS space. The frame is derived from the source
plane/structural evidence and is not assumed to be the global z axis. A local
child proposal is

```text
anchor_position + delta_t1*t1 + delta_t2*t2 + delta_n*n
```

The observed anchor remains provenance-stable. Bounded support refinement may
be used by an authorized contract; long-distance propagation creates children
instead of dragging the observed anchor away from its evidence.

## 9. Shared anchor-local StructuralField

One shared tiny StructuralField is evaluated on normalized anchor-local
coordinates and aggregated legal evidence. The field supplies a scalar
structural support/value and an explicit supported mask. It is a shared
structure prior, not an oracle segmentation field and not a hidden global
coordinate field. Field support and gradients are recorded as diagnostics.

## 10. Structural and volumetric Gaussian banks

The state contains two explicit banks. Structural Gaussians are thin,
anisotropic, and aligned with the local frame/StructuralField to preserve
boundaries and continuity. Volumetric Gaussians have broader support and carry
modality-specific appearance for interior tissue. They are not interchangeable
dense appearance primitives. In the maintained R4 path, their bounded local
offsets, covariance, amplitude, and modality appearance are produced by the
shared Gaussian head through the explicit `anchor_evidence_prefix` adapter
(currently the configured 25-channel prefix of context-only anchor evidence).
The local StructuralField also contributes a bounded local-normal placement
offset to head-produced volumetric seed centres. This keeps the field on the
live differentiable reconstruction path when a thin structural bank is
unsupported at a sparse interior target, while the Gaussian head remains the
owner of volumetric appearance, covariance, and modality validity. The field
offset uses context-derived evidence only and is not a target-derived
appearance shortcut.

Every primitive records patient identity, source provenance, parent anchor or
primitive, modality validity, positive-definite covariance, uncertainty,
support validity, propagation round, and immutable state/version binding.

## 11. Bounded propagation

Propagation is a finite transaction:

```text
frontier
→ bounded local child proposals
→ structural support and legal evidence checks
→ optional cross-modality agreement
→ uncertainty/redundancy penalties
→ physical validity, covariance, NMS, and budget checks
→ immutable accepted child set
→ declared stop condition
```

P0 disables propagation exactly. P1 uses fixed local-frame offsets and finite
rounds. Per-anchor child, per-round, per-bank, and per-patient budgets are
enforced. Proposals are rejected for duplicate collision, insufficient field
support/evidence gain, high uncertainty, invalid covariance/non-finite values,
outside-volume geometry, unsupported modality appearance, or budget limits.
P2/P3 adaptive topology are optional and are not required by the first product
run.

Volume rejection is affine-aware. The RAS axis-aligned bounds are only a fast
coarse rejection; when source geometry is available, every proposal is mapped
through the inverse source affine and must lie inside the continuous-index
voxel-center extent. This prevents oblique source boxes from admitting points
that lie inside their RAS AABB but outside the oriented physical volume.

The dual-bank memory digest likewise covers structural/volumetric kind,
primitive and parent identities, provenance, covariance policy and factor,
gauge provenance, appearance validity, and every observability component. This
is an integrity binding, not a claim that the propagated support is calibrated
uncertainty.

## 12. Support, unsupportedness, and uncertainty

Renderer support is explicit. Unsupported values remain NaN in serialized
prediction volumes and are never silently zero-filled for metrics. Support
fraction, unsupported fraction, support-conditioned metrics, propagation
counts, and support-derived uncertainty are reported per patient. Uncertainty
calibration claims are skipped unless a semantically valid uncertainty model is
declared and independently evaluated.

## 13. Renderer and query semantics

The query domain is a physical target plane or full physical grid. Gaussian
profiles account for through-plane thickness/profile and preserve the target
affine/grid. Camera pose, perspective projection, view-dependent color, and
global-z propagation are not method assumptions. The maintained initial
product episode queries the held-out target plane. An explicit full-source-grid
mode derives the `[d,h,w]` grid from source geometry and declared in-plane
stride, streams depth chunks, and serializes an affine-preserving prediction
package; its latency remains unmeasured until that mode is actually run.

## 14. Training legality

All preprocessing statistics are fitted per modality from exactly the declared
context set. Robust percentile clipping is context-only and has an explicit
output range. Target bytes are receipt-gated. Audit volumes and labels are
isolated from training. Patient pseudonyms, source hashes, split/assignment,
sampling protocol, preprocessing, model, checkpoint, and prediction package
hashes are bound in run provenance. W&B receives pseudonyms and safe derived
images only, never raw IDs, full NIfTI volumes, or segmentations.

The streamed cohort trainer keeps global encoder, Gaussian-head,
anchor-local-StructuralField, and optimizer state separate from each
patient-specific Gaussian volume. A successful training patient atomically
promotes that shared state into a target-free global checkpoint. Validation
episodes may consume the last training checkpoint but do not backward, update,
or promote it; patient state and evaluator payloads remain outside the global
checkpoint. Validation therefore cannot alter the state used by a later
training patient.

## 15. Evaluation protocol

The evaluator consumes serialized prediction packages and immutable target
packages, not mutable trainer state. It reports MAE, RMSE, PSNR with declared
data range, global-SSIM with declared window policy, NCC, gradient MAE/RMSE,
edge F1, local contrast error, support fractions, and explicit unsupported
handling. Distance-to-context, gap, observability, ROI, uncertainty, and
resource analyses are evaluator-only and are reported as unavailable when the
serialized package lacks the required semantics.

The streamed product controller preserves one row per patient and variant in a
target-free CSV, then writes an aggregate JSON containing unweighted patient
macro-statistics, deterministic patient-bootstrap percentile intervals, and
pooled support counts. Non-finite or unavailable metric values are omitted from
the corresponding aggregate with an explicit count; they are never converted
to zero. Patient rows retain both support-conditioned and complete-plane metric
fields with explicit status and data-range provenance. Unsupported voxels are
excluded by the evaluator and are never represented as zero-valued errors.

Product full execution may include a patient-disjoint validation sweep.
That sweep checks streamed execution and serialized evaluation only; it is not
checkpoint selection and does not establish held-out reconstruction quality.

## 16. Required ablations

The first matched matrix is E0/E1/E2, R0/R3/R4/R5 where implemented, P0/P1,
sparse-3/5/7/9, aligned/staggered, and axial-only. Any privileged dense,
teacher, or segmentation-informed upper bound must be isolated and labelled;
it cannot select the main method. T4 routing and learned acquisition remain
blocked.

## 17. Failure modes

The pipeline fails closed on malformed or incomplete NIfTI input, affine/shape
mismatch, non-finite payloads, invalid physical geometry, target/context
overlap, empty legal masks, unsupported coverage, non-finite loss, invalid
covariance, budget overflow, target-before-receipt access, corrupt artifacts,
or unavailable CUDA for a GPU product run. A failed engineering run is
preserved as evidence and is not converted into a scientific result.

## 18. Permitted and forbidden claims

Permitted claims are limited to implementation status, reproducibility,
software-contract tests, data inventory, execution diagnostics, and explicitly
scoped evaluator outputs. The repository must not claim reconstruction
superiority, clinical validity, safety, scanner acceleration, recovered unseen
pathology, calibrated uncertainty, or novelty from smoke, pilot, unit-test, or
Human-Gate-independent evidence. Scientific and clinical PASS remain pending.
