# PFGR-Lite evidence integration contract

## Final service integration clarifications

The staged execution configuration is a strict wrapper around the existing
PFGR and frontend configurations, resolved normalization and explicit stage
options. Optimizer, loss, teacher and operational work settings are saved;
they are not undeclared callable defaults. W3b owns `stages.py`,
`objectives.py` and `data.py`, including the real data/checkpoint input factory.
The public stage result includes canonical `StageState`, actual artifact
paths, metrics and completion evidence. A boolean `prediction_ready=True`
cannot authorize a target read: deferred supervision requires the actual
completed prediction/trace and its `ObservationContext`.

Production target validation binds the observation context's actual mask,
affines, normalization and producer identity. An unbound standalone fixture
must explicitly declare engineering-only use. Immutable target validation is
reused for the same subject context; it is not repeated for every candidate.
Full audit and cheap mutation/version guards have distinct scopes.

W4 owns a local `ForcedCalibrationTrace` wrapper in `calibration.py`, rather
than adding more fields to shared route types. It binds the actual observation
context, sequential route and collection `EffectivePolicy`. Collection uses
forced greedy budget four with no calibration, the exact fitted V and scale,
and matching route/decision policy hashes. Complete no-legal short routes are
valid only when the stored terminal proposals have no legal nonzero action.
The collection-policy identity is distinct from the intended adaptive-policy
identity; their value, scale, candidate, writer, query, tie and revisit
dependencies must agree. Winner raw scores must equal the executed decision's
raw gain score. Exact confirmation has Q=0, no sampling seed and zero Monte
Carlo uncertainty.

`ObservationContext.context_id` is a content digest, not a subject identifier.
The real W3b load/encode boundary creates a versioned subject-to-context
receipt before supervision, containing subject, source observation record,
context, geometry and normalization identities. W4 validates this receipt
against the actual context and external training-role manifest; winner rows
cannot independently assign subject identities to arbitrary traces.

W5 is split into disjoint implementation tasks after interfaces are available.
The scientific-services worker owns `experiments.py`, `oracle.py`, `metrics.py`
and `benchmark.py` plus their tests. It exposes `run_evaluation`,
`run_oracle_evaluation` and `run_teacher_benchmark` using W3b `StageInputs` and
strict local scenario options. S6 delegates the same evaluation service.
The later CLI/runbook worker owns `smagm/cli/pfgr_lite.py`, PFGR configuration
files, root runbook, README/docs links and command integration tests. It calls
these services and the existing staged APIs; it does not implement another
policy or data path. Both workers must remain `gpt-5.6-luna` / `max`.

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

## Parallel diagnostic trace clarification

W4 review found that rebinding an initial proposal batch to successive state
digests fabricates freshly scored actions. The parallel control must retain
the original initial-state proposal/action identities throughout compound
execution. It is an explicit diagnostic exception to sequential freshness,
not permission to weaken `apply_scored_action` stale-action rejection.

W4 additionally owns a narrow `ParallelBehaviorTrace` declaration in
`types.py`, plus optional `PFGRRouteResult.parallel_trace`. It contains the
original initial state and proposal batch, ordered selected original action
digests, actual intermediate/final states, and effective policy identity.
Validate initial state/proposal bindings, unique selected rows, unchanged
producer/context, state order and exact selected delta provenance. The
compound writer may apply those stored deltas without re-querying or calling
U. It must retain the original action IDs; it may not mint sequential
proposal digests for intermediate states.

Parallel results carry this trace and no sequential `completed_trace`.
Sequential teacher/calibration/bank consumers reject a parallel trace.
Diagnostic interaction evaluation may explicitly compare compound final
gain with independently measured initial-state single-action gains. This
clarification changes neither the writer nor the correction family.

## Producer-stage receipt

W3a validates and stores one `stage_provenance` mapping; W4 checkpoints
preserve the same mapping and W3b emits it from actual stage measurements.
Its exact fields are `schema_version=pfgr-lite-producer-stage-v1`,
`stage=updater`, `spectral_arm` (`u_plus_spectral` or `verified_prior`),
`completed`, `producer_compatibility_hash`, `projector_before_hash`,
`projector_after_hash`, `projector_gradient_evidence`,
`projector_update_evidence`, `initialization_id`, `checkpoint_id`, `source_id`,
`split_role_hash`, `role_manifest_digest`, `verified_prior_receipt`, and
`verified_prior_receipt_hash`. The checkpoint ID identifies the input/parent
snapshot; it is not a circular checksum of the output artifact.

Gradient evidence contains finite nonnegative `l2_norm_max`, integer
`nonzero_steps` and `measured_steps`. Update evidence contains integer
`changed_parameter_count` and `optimizer_steps`. MAIN U-plus-spectral
evidence requires completion, positive measured gradient/update evidence,
different before/after projector hashes, and an after hash matching the
current producer's spectral projector dependency. These are software checks
of a trained producer, not evidence of useful reconstruction capacity.

For a verified prior, the original U-plus-spectral receipt and its canonical
hash are required and validated, including the same final projector hash.
A magic `not_applicable` string is insufficient. The current outer receipt
binds current U and other producers; the original receipt records ancestry.
Both prior fields are null for a direct U-plus-spectral stage. Engineering
fixtures may omit this receipt and remain engineering-only. Static-only
checkpoints carry explicit static-stage provenance and cannot supply a MAIN
ValueBank until useful spectral producer training is documented as above.

## Completed calibration rollouts and terminal assessments

The calibration rollout has a forced compute budget of four. Completion
does not require four executed writes when no legal action remains. Such
short completed traces retain their measured winners and actual stop reason;
the independent-group and unique-winner minimums remain unchanged. An
arbitrarily truncated trace is not complete evidence.

W4 may add optional typed `PFGRRouteResult.terminal_proposals` containing the
actual final assessed `ActionProposalBatch` when stopping early. It is bound
to the final state and terminal decision, without inventing a next state or
sequential write. A local calibration collection receipt may wrap that route
result to validate forced policy, budget, all executed actions and a genuine
no-legal-action terminal assessment. This is preferable to mutable undeclared
attributes. Sequential transition traces retain their existing meaning.

The already-authorized synthetic harness uses the same selector/STOP code
with explicit engineering-only policy and diagnostic affine parameters.
Every derived receipt retains that status, and adaptive deployment artifact
save/load/promotion remains forbidden. Synthetic tests must not fabricate
production training provenance to exercise this harness.
