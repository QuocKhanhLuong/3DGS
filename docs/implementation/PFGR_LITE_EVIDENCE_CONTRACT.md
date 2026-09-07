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

## Diagnostic oracle-state teacher seam

Integration review exposed that a target-aware oracle history cannot honestly
be represented by `CompletedBehaviorTrace`, whose contract is target-free
completed behavior. The accepted clarification is a separate
`measure_diagnostic_actions(state, actions, target_context, decoder,
teacher_config, *, lattice, chunk_size=1024, candidate_chunk_size=1,
seed=None, observation_context=None, counters=None, scope="oracle_state")`
API in `teacher.py`. It validates the actual immutable state and action
identities, preserves their original versions, and shares the existing
exact/fixed-Q numerical evaluator. It must not fabricate transitions, no-op
successors, target-free trace objects, or rebased action identities.

Its return is a tuple of frozen `DiagnosticGainResult` wrappers with schema
`pfgr-lite-diagnostic-gain-v1`, actual action/context/state/proposal identities,
`label: GainLabel`, target-context digest, constant `scope="oracle_state"`
and `privileged=True`. It does not delegate ordinary GainLabel attributes.
MAIN ValueBank adapters and deployment APIs reject the wrapper; oracle
exports explicitly retain privileged provenance. The ordinary
`measure_actions` API remains strict and unchanged externally. Diagnostic
target contexts bind the immutable subject/context/mask/normalization and
have `trace_route_hash=None`; their target join occurs only after the actual
initial prediction and first proposal bank. Route-bound targets are rejected
by this diagnostic API rather than having their binding discarded.

W5 scientific services owns only the necessary additional `teacher.py` and
`test_teacher.py` edits. Acceptance requires the existing teacher suites plus
actual Oracle-1/greedy-2 final-write, state-identity and telescoping checks.
This clarification repairs a software interface gap; it is not an observed
research result or a new teacher approximation.

## Actual staged optimization resume state

S0/S1 receipts alone cannot resume optimization. `StageResult.runtime_state`
and `StageInputs.resume` use strict schema `pfgr-lite-stage-runtime-v1` with
fields `stage_state`, `optimizer_state`, `rng_state`, `cursor`,
`parameter_names`, `execution_config`, `execution_config_hash`,
`training_config_hash`, `producer_compatibility_hash`, and `split_role_hash`.
The cursor contains `epoch`, `batch_index`, `update`, `microstep`,
`sample_order`, and `route_rng_state`; it identifies the next unprocessed
batch. Parameter names preserve exact optimizer-group/order ownership.
Snapshots own detached CPU state, including actual Adam moments and supported
Python/NumPy/Torch/CUDA RNG streams. Accumulation is initially one; unsupported
values fail closed rather than being ignored.

The full resolved execution mapping is retained and hashed.
`training_config_hash` excludes only `stage_options.max_updates`, allowing an
explicitly recorded continuation budget to change after a bounded interruption.
Old/new full hashes and that override are recorded. Model, data, optimizer,
epoch, seed and precision semantics must match. Restore occurs after strict
model hydration and identity checks, before route randomness or processing
the next batch. An interrupted epoch is not marked complete.

CLI persists the actual optimizer/RNG through existing W4 `save_resume` and
stores the remaining metadata in `bank_state["stage_runtime"]`. Resume
hydrates model and optimizer and continues the same stage; loading metadata
and printing a summary is not a resume experiment. Empty optimization state
after completed updates is not accepted as resumable. W3b owns runtime capture
and execution; W5 CLI owns serialization wiring and may narrowly harden
`checkpoint.py`/`test_checkpoint.py` for incomplete optimization artifacts.
Acceptance compares uninterrupted and interrupted/resumed actual S0/S1
weights, optimizer state and action schedule under the same numerical setup.

## Parallel measurement and resume serialization clarifications

The completed frozen-proposal parallel route has a `ParallelBehaviorTrace`,
not a sequential `CompletedBehaviorTrace`. Its post-route control uses
`measure_parallel_actions(parallel_trace, selected_actions, target_context,
decoder, teacher_config, *, lattice, observation_context, chunk_size,
candidate_chunk_size, seed, counters)`. This validates the actual typed parallel
trace, initial state, selected action membership/digests and compound completion,
then shares the sparse numerical engine on that initial state. It returns
diagnostic wrappers with `scope="parallel_initial_state"`; these remain excluded
from MAIN banks. The oracle public API retains its separate `oracle_state`
scope. No fake sequential transition or action rebasing is permitted. The sum
of independent initial-state gains is an interaction control, not a telescoping
claim for the compound parallel prediction. W5 scientific services owns this
narrow teacher extension and its actual-model integration regression.

Resume's exact execution configuration legitimately contains the schema field
`bank_state.stage_runtime.execution_config.pfgr_config.teacher`. W5 CLI may allow
this precise path after strict execution/config validation. The value contains
only the versioned teacher configuration, never target tensors, labels or teacher
contexts. Unknown nested fields and forbidden target/oracle state elsewhere
remain rejected. Redacting or stringifying the configuration to evade validation
is not an accepted serialization strategy.

The runtime schema additionally requires `input_manifest_hash`, derived from
the ordered `(subject_id, observation_record_id)` pairs. Observation record
identity includes observation/mask content, geometry, normalization and source
paths. Equal subject names alone do not establish equal resume inputs. The
hash is validated before optimizer/RNG mutation. Synthetic observation generation
uses an independent deterministic generator so model hydration cannot change
fixture data under unchanged names and seeds. `continuation` is the explicitly
permitted optional runtime field recording the allowed max-update override.

`ExperimentOptions.local_footprint_audit` is an explicit false-by-default
diagnostic. Its selected stored actions are measured after completed inference
using both physical-sphere local signed Charbonnier gain and MAIN complete
footprint/global-denominator gain. `local_footprint_audit.jsonl` records both
denominators, voxel counts and sign disagreements; local control labels never
enter MAIN banks. W5 scientific services owns computation; the CLI owner
forwards the flag and documents its R4 command. Paired comparison must consume
the actual service artifacts, retain required producer/split/loss identities,
and permit absent learned results for pre-ValueNet correction/selection
headroom decisions. Whole-route oracle-minus-learned gain is a route gap,
not top-1 regret at a single measured candidate state.

## Concrete calibration runner

W5 CLI owns `calibration_runner.py` and its dedicated tests. W3b delegates
`run_calibration(inputs, CalibrationRunOptions, output_dir)` directly. Options
use `pfgr-lite-calibration-run-v1`: `confirmation_mode` (`exact`, default, or
`iid_fixed_q`), `confirmation_q_draws` (zero for exact, at least two for sampled),
`confirmation_seed`, `collection_seed`, `value_input_variant` (366 default),
`max_subjects` (positive integer or null), and `engineering_only`. Roles come
from the immutable manifest, not configurable role-name aliases. The factory
has explicit `subject_role="calibration"` for the union of training-only fit
and allowance subjects; the CLI must not inherit a validation role for S5.

Inputs carry actual model, samples, manifest, deferred subject-ID target
provider, and the existing value-model/fit-identity/scale metadata. Concrete
defaults construct W4 forced learned K4 routes and real subject/context
bindings, then measure their stored winners through W2. Callback-only
production defaults and fabricated public synthetic winners are prohibited.
Collection completes before teacher measurement. Producers remain frozen and
run without autograd; cohort GPU contexts/graphs are not retained. CPU staging
or exact bounded replay must preserve identity and account for memory and any
additional encodings. A required new compact validation seam needs an explicit
planner amendment to W4.

The result carries `schema_version`, `calibration_evidence`, `fit_winners`,
`allowance_winners`, `completed_traces`, `collection_policy`, `calibration`,
`artifacts`, and `metrics`. Bounded engineering collection uses actual subjects
and labels; insufficient data yields INCONCLUSIVE, `calibration=None`, and no
adaptive artifact. It never fabricates subjects to meet minimum data counts.
Actual action-derived confirmation seeds remain distinct from the parent
confirmation seed; exact labels use Q=0, seed=null and zero MC uncertainty.

The legacy frontend import test permits the authorized staged data boundary
only for `load_point_guided_subject` and `load_point_guided_split` imported from
`smagm.data.brats21_point_guided` in the exact PFGR `data.py` and `stages.py`
paths. Inference/model/routing modules and all other forbidden imports retain
the original restriction. W3b owns this narrowly scoped test amendment and
negative sibling-path/symbol coverage.

## CLI continuation and selected-state audit clarification

CLI S0/S1 publication preserves the actual W3 runtime snapshot; it must reject
missing/conflicting fields instead of filling defaults or rewriting identities.
W3 validates all dependencies before optimizer/RNG restore. Cached V continuation
uses the existing W4 resume envelope with the producer InferenceBundle and the
actual W3a fit state under bank_state.cached_value_fit; it does not invent a
second checkpoint family or load MRI during continuation.

W3b additionally owns pfgr_lite/bank_audit.py and test_bank_audit.py.
write_state_snapshot(bank_root, state, context, *, subject_binding, route_hash,
selected_actions, split_role_hash) writes one detached CPU snapshot per selected
state under bank/replay/<sha256>.pt. It returns the safe relative reference for
existing ValueBankRow.selected_replay_ref. Schema is
pfgr-lite-selected-state-snapshot-v1. The payload includes the three state
planes once, actual context/state/producer/split/normalization/subject binding,
output and feature geometry metadata, route identity and selected action
identities. It contains no target, segmentation or image prediction. Snapshot
file digest is bound by its filename and the immutable indexed row reference;
existing differing files are never overwritten.

audit_bank_replay(reader, replay_count, *, producer, role_manifest) validates a
bounded selected row set, safe reference paths, file digests, weights-only
schema, plane state digest and subject/context/state/action/producer/split joins.
Its receipt explicitly reports audit_kind=state_snapshot_and_row_identity,
actual rows/snapshots/bytes checked and zero decoder/teacher calls. It does not
claim reconstruction replay. Positive requested audits with missing snapshots
fail closed. One shared snapshot may serve multiple action rows; raw bank and
snapshot tensors remain excluded from the default evidence package. W3 owns
snapshot production/helper tests; W5 CLI only calls the helper and documents
the real audit scope. No change to MAIN row or checkpoint schema is required.

## S2 measurement budget sidecar clarification (2026-09-07)

`StageOptions.teacher_q_draws: int | None = None` is the explicit S2 measurement budget. `None` inherits the frozen config teacher Q; fixed-Q overrides require at least two draws; exact enumeration resolves Q=0. CLI `--query-count` binds this sidecar, while `query_mode` selects exact versus sampled measurement. These execution options are serialized and labelled, and do not mutate the saved producer PFGR configuration or its loss/mask definition. Unsupported state/candidate counts fail explicitly rather than being clamped. W3 owns the stage field/resolution; W5 owns CLI forwarding.
