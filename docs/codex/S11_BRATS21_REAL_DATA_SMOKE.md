# S11 BraTS21 real-data smoke

> Historical handoff. This document records the former one-patient smoke and
> pilot staging protocol. It is retained for provenance, but those launches
> are retired; the active product entry point is
> `scripts/train_brats21_full.sh` with the full product config.

This is a software and execution smoke for one retrospective sparse derivative
of one BraTS21 patient. It is not a permanently sparse acquisition study and
does not establish reconstruction quality, superiority, clinical validity, or
any Human Gate decision.

## Data boundary — product protocol (2026-08-02)

`scripts/data/inspect_brats21.py` validates patient names, exact `t1`, `t1ce`,
`t2`, `flair`, and optional `seg` suffixes, three-dimensional finite data,
resolved qform/sform affine geometry, and BraTS labels. It fails closed on a
malformed patient. `scripts/data/prepare_brats21.py` creates a metadata-only
cohort manifest, while the product controller reads only selected source
planes, writes small rank-2 NumPy payloads, and records source and derivative
hashes. Product preparation consumes the completed inventory's source hashes
and hashes only its materialized context payloads; it does not re-hash source
NIfTI files during preparation or bundle reuse, because that would open hidden
target/evaluator bytes before the receipt barrier. The dense NIfTI root is never
copied into a prepared bundle.
Segmentation is evaluator-only.

The maintained product episode contains five aligned physical axial positions
per modality (20 context observations) and one disjoint held-out target. The
positions come from source-affine physical quantiles 0.15–0.85; training jitter
is seeded and bounded, while validation/test jitter is zero. Target selection is
deterministic from an interior context gap and uses its exact physical midpoint.
The target reference stores the corresponding fractional source index; target
intensity is linearly interpolated only after receipt, while evaluator-only
segmentation uses nearest-neighbor extraction. Preparation writes context planes
and target source/geometry references, not target intensity or segmentation
payloads; the target reader materializes intensity only after prediction receipt
registration, while the evaluator segmentation reader is opened later after
prediction serialization.

## Execution (historical staging; active launch is full-only)

The full product controller executes the declared R0 baseline and E2 + R4 + P1
path for the cohort. It streams patients, writes atomic state, keeps
serialized prediction packages, resumes at patient boundaries, and quarantines
invalid terminal markers without discarding resumable R0/R4 progress, and
quarantines incomplete prepared/R0/R4 substages with preserved evidence before retry. The
evaluator uses a declared data range and global SSIM policy.

The full experiment config explicitly declares the validation cadence as
`post_training_patient_disjoint_sweep`, with fixed geometry-only selection and
no checkpoint selection. This sweep is diagnostic pipeline evidence only; it
does not select checkpoints or create a scientific validation claim.

For the full training split, the encoder, Gaussian head, shared
StructuralField, and optimizer are carried forward through an atomic
`global_model_checkpoint.pt` at each successful training-patient boundary.
Patient Gaussian state remains separate and immutable. The patient-disjoint
validation sweep loads the final global checkpoint, builds one context-only
patient state, and performs no backward pass, optimizer step, or global-state
promotion; its target is used only after the receipt barrier by the isolated
evaluator. The global checkpoint is explicitly target-payload-free.

Promotion is a two-phase patient-boundary transaction: the run state first
records a journal containing the expected next index, predecessor hash, source
patient pseudonym, manifest/assignment hashes, and model binding, then atomically
promotes the checkpoint and clears the journal. Resume accepts only that exact
one-step successor and reconciles its completion record; an unjournaled or
arbitrarily advanced checkpoint is rejected. P1 also checks proposed centers by
inverse source affine against the oriented physical volume, not only an RAS
axis-aligned bounding box.

The maintained default query is the held-out target plane. The reconstruction
runner also accepts the explicit
`full_source_grid` mode. That mode constructs a `[d,h,w]` TargetGrid from the
prepared source shape, source affine, and declared in-plane stride only; it
streams the configured depth chunks and writes a separate hashed prediction
package with intensity, support mass, unsupported mask, uncertainty, and
affine metadata. It never opens target intensity to define the query grid.
NIfTI export, when enabled, serializes the volume tensor in source `[x,y,z]`
order under the preserved grid affine. Full-grid latency is therefore absent
from the default readiness evidence until that mode is explicitly requested.

The product config requests native single-GPU CUDA float32. If the host has no
usable driver, execution fails closed; there is no CPU fallback. Dry-run config
validation remains available without CUDA. The run reports device state, peak
CUDA memory, explicit support masks, gradient norms, and artifact paths. In
R4, the context-derived StructuralField supplies a bounded local-normal offset
to both structural seeds and head-produced volumetric seed centres, so the
field remains on the differentiable render path even when thin structural
support is absent at an interior target. Each
patient also prints and records the resolved experiment name, total/trainable
parameter counts, and one legal training-step FLOPs estimate from the native
PyTorch profiler; profiler-unavailable is recorded explicitly rather than
blocking training.
Repository provenance hashes tracked changes and new source/config text; it
records generated trees and binary/volumetric paths by relative path and size
without rereading their payloads. Those artifacts are bound separately by
their own manifest or report hashes.

After a successful stage, the controller writes a target-free
`patient_metrics.csv` and `aggregate_metrics.json`. The aggregate contains
unweighted patient macro-statistics, deterministic patient-bootstrap
percentile intervals, and pooled supported/unsupported voxel counts.
Non-finite metrics are omitted with an explicit count and are never converted
to zeros. The patient rows retain both support-conditioned and complete-plane
metrics, plus explicit metric-scope/data-range/unsupported-status columns;
unsupported pixels are never converted to zero-valued errors.

The full experiment config explicitly binds the run-directory policy, log
filename, atomic state filename, completion marker, global checkpoint filename,
and refusal to overwrite a successful run. It declares a unique UTC-suffixed
run directory; `scripts/train_brats21_full.sh` supplies the corresponding
output directory, CUDA selection, resume policy, W&B mode, and log path. There
is no smoke/pilot prerequisite gate. The configured free-disk value is an
advisory warning for this launch; it is recorded in state but does not replace
the operating system's actual out-of-space behavior.
When W&B mode is `offline` or `online`, the runner requires an active W&B
client (online initialization may be recorded as an explicit offline fallback);
it does not silently continue with no logger. `disabled` is the only mode that
permits no W&B run.

No T4 routing or adaptive acquisition is present. No scientific PASS is
generated automatically; T2/T3/T5 Human Gates remain separate.
