# PFGR-Lite evidence integration contract

Principal clarification, 2026-09-07. This supplements the accepted plan and
interface companion; it changes no architecture, gain objective, sampling
law, calibration minimum or scientific acceptance criterion. Early W3a/W4
review found that independent strings for role membership cannot provide the
required joins. The following shared seam is frozen before their fixes.

## Shared role manifest

W4 has exclusive additional ownership of `pfgr_lite/types.py` for the new
`TrainingRoleManifest` declaration and the narrow bundle/route fields below.
W3a consumes this declaration and must not create a competing role type.
Tests for it belong to W4 `test_calibration.py`/`test_checkpoint.py`; existing
W1 contract tests must remain passing. No other W1 file is authorized here.

The immutable declaration contains exactly these semantic fields, with
explicit canonical serialization and digest:

```text
schema_version = pfgr-lite-training-roles-v1
baseline_split_hash: str
baseline_train_subject_ids: tuple[str, ...]
baseline_validation_subject_ids: tuple[str, ...]
baseline_test_subject_ids: tuple[str, ...]
producer_fit_subject_ids: tuple[str, ...]
calibration_fit_subject_ids: tuple[str, ...]
calibration_allowance_subject_ids: tuple[str, ...]
subject_group_ids: tuple[tuple[str, str], ...]  # subject -> related-scan group
assignment_seed: int = 20260907
engineering_only: bool = False
```

Identifiers are unique and nonempty. Baseline train/validation/test are
disjoint. The three training roles partition the unchanged baseline training
IDs exactly. Every subject has exactly one group entry; a related group may
not cross baseline splits or the training roles. MAIN uses the already
planned deterministic hash assignment of groups; the data/preflight service
must validate the original split through the existing loader before forming
this declaration. Artifact loaders compare the canonical role digest and
expected baseline split identity. A manifest's self-declared source hash is
not a substitute for this external split check.

Production preflight/adaptive release requires at least 32 independent
groups in each calibration role and at least one producer-fit group. Tiny
engineering manifests may have empty calibration roles, but cannot establish
real adaptive release. V fitting and scale calculation accept only
`producer_fit` rows. Explicit development-evaluation rows, if stored, must
join baseline validation and cannot contribute to training/scale. Calibration
and test rows never become V training examples. No generic `train` or
`calibration` alias silently substitutes for these roles.

## Evidence envelopes

The bank index is lowercase `index.json`. Its one canonical envelope contains
the authoritative `ValueBankManifest`, this role manifest, complete typed
producer dependencies, producer-stage/spectral provenance, fixed scale
provenance and row/shard identities. Missing row identities fail closed;
the writer may not invent them from an arbitrary producer string. MAIN
complete-support measurement uses exact versioned law identifiers or a
validated structured record, never substring matching such as `complete`.
Diagnostic/engineering provenance remains explicit in every derived artifact.

W4 may define a local calibration evidence envelope containing the canonical
`GainCalibration`, `TrainingRoleManifest`, completed forced-trace receipts,
unique winner/action records, measurement/confirmation metadata and all
dependency hashes required by plan section 8. It is not a replacement
calibration type. The envelope must bind actual completed target-free traces
and reject duplicate winners, role overlap, synthetic release, stale value or
scale, and screening labels used as confirmation. The fitted parameters and
pooled allowance remain exactly as already planned.

The public policy loader validates this envelope against the current
producer, exact `ValueFitIdentity`, fixed scale, role manifest and effective
policy dependencies. Bare `GainCalibration(capability='adaptive')` is
insufficient for deployment. Synthetic tests may exercise the same math and
policy in a explicitly engineering-only harness; that harness cannot save or
load a real adaptive-release bundle.

## Narrow shared bundle/trace fields

W4 may extend authoritative `InferenceBundle` with the strict frontend
configuration sidecar, `ValueFitIdentity`, fixed scale provenance, effective
policy, role manifest, stage/spectral provenance and calibration evidence
envelope. Existing structural fixtures may retain explicit diagnostic defaults;
production checkpoint APIs require the complete appropriate fields and strict
model hydration. Do not create a second inference or resume schema merely to
avoid extending the authoritative declaration.

W4 may extend `PFGRRouteResult` with its `CompletedBehaviorTrace` so W2/W3
consume the actual states, proposals and decisions from the completed route.
Trace retention must be explicit and bounded by K<=4; detached teacher/bank
snapshots and differentiable updater routes preserve their separate APIs.
The existing `ResumeState` remains the checkpoint authority. W3a's local fit
cursor/result is stage payload within it, not a competing public resume
checkpoint. Real optimizer integer keys and Python/NumPy/Torch RNG state must
round-trip and restore correctly.

All additions are independently reviewed and tested before acceptance.
W4 requests a serialized commit slot for its additional owned file. W2 and
W3a must not edit it concurrently. CLI, real-data execution, GPU numerical
evidence and trained improvement remain pending.
