# Permanently Sparse Multi-Sequence MRI Training Protocol

Date: 2026-07-29
Status: blocking data-legality contract for T0.5 and T1
Owner: Medical Data Steward
Venue framing: ISBI 2027 / medical imaging

## 1. Scope

This protocol defines which observations may exist, which bytes may be opened,
and when they may be opened during permanently sparse multi-sequence MRI
training. It applies to dataset preparation, patient splitting, episode
construction, target reveal, matched T1 experiments, and isolated audit
evaluation.

The scientific unit is a patient with a fixed set of sparse observations that
were actually acquired or legally exported before training:

\[
\Omega_i^{sparse}
=
\{(a_{i,j}, I_{i,j})\}_{j=1}^{K_i}.
\]

`SparseAvailabilityManifest` records \(\Omega_i^{sparse}\).
`EpisodeAssignment` assigns temporary context and target roles without changing
that manifest. No T0.5 or T1 code may infer permanent `CONTEXT` or `TARGET`
roles from availability metadata.

This is a scientific-validity and reproducibility boundary, not a security
sandbox. The Medical Data Steward must veto a loader, cache, preprocessing job,
or experiment that can access non-manifest or audit pixels.

## 2. Cohort classes

Every patient belongs to exactly one declared cohort class before any experiment
is launched:

1. **Main permanently sparse cohort.** Only the acquired sparse observations in
   the availability manifest are present under the training data root. This is
   the only cohort that supports the main permanently sparse training claim.
2. **T1 lesion-validation cohort.** Patient IDs are disjoint from main
   train/validation patients. Reconstruction receives only a predeclared,
   hashed sparse input manifest. Full volumes and lesion/ROI labels remain
   evaluator-only. Aggregate reconstruction and lesion/ROI results may guide
   development decisions for architectures, configs, thresholds, and
   checkpoint-selection rules, but neither sparse inputs nor evaluator-held
   targets or labels may enter gradient training. This cohort decides T1-M and
   is not the final paper audit.
3. **Sealed T5 final-audit cohort.** Full volumes and lesion/ROI labels exist
   under a separate evaluation root and credentials for patient IDs absent from
   every main and T1 lesion-validation manifest. Its sparse inputs, full
   volumes, labels, and derived summaries remain unopened until architectures,
   configs, thresholds, checkpoint-selection rules, and analysis plans are all
   frozen and final T5 evaluation is authorized in a future tranche.
4. **Privileged or simulated cohort.** A sparse set derived retrospectively from
   a fully sampled parent scan must be labeled simulated or privileged. The
   dense parent remains outside the main loader. Results from this cohort may be
   reported as a simulator, oracle study, E3/E4 upper bound, or engineering
   check, but not as evidence that the main method trained on permanently sparse
   acquisitions.

Dataset licenses, institutional approvals, data-use agreements, and permitted
derivatives must be recorded before manifest creation. A file that is technically
readable but outside those permissions is not a legal observation.

## 3. Manifest creation

### 3.1 Source-of-truth inventory

For each main-cohort patient, create an inventory from acquisition/export
records before model training. Include only observations whose pixel payloads
were actually acquired or legally exported as part of the sparse study set.
Do not scan a hidden dense volume to choose visually useful slices.

Each observation record must contain at least:

- globally unique `observation_id`;
- pseudonymous `patient_id`;
- patient-level split;
- manifest-relative payload path;
- modality and protocol identifiers;
- immutable physical plane in canonical RAS millimetres;
- source-affine convention and transform provenance for non-synthetic data;
- shape, pixel-center origin, axes, spacing, signed normal, and thickness;
- acquisition/export provenance and a synthetic/privileged flag;
- optional registration transform identifier and confidence;
- a canonical decimal acquisition-cost key or reference to a versioned cost
  schedule.

Observation IDs and paths must be unique. Paths must be manifest-relative and
must not escape the bound data root through `..`, absolute paths, or symlinks.
All observations in a patient manifest must have the same patient ID and
patient-level split.

### 3.2 Immutable hashes

Manifest preparation produces two distinct bindings:

1. **Canonical manifest hash.** SHA-256 of canonical JSON containing only
   immutable public metadata, with entries sorted by `observation_id`, stable
   field names, normalized RAS-mm geometry, and canonical decimal strings.
   This hash identifies the legal availability set.
2. **Private content binding.** SHA-256 of each payload plus a sealed canonical
   hash of the sorted `(observation_id, content_sha256)` map. Payload digests
   are available only to the bound file provider and audit tooling. They must
   not appear in target metadata, model inputs, episode sampling, or the public
   pre-reveal manifest representation.

Both bindings are immutable once an experiment family is registered. Any pixel,
geometry, provenance, modality, path, or cost-schedule change creates a new
manifest/version and new hashes; it is never an in-place correction. A read
whose payload digest differs from the sealed binding fails before emitting an
opened-file audit record.

### 3.3 Legal loader boundary

The training process receives:

- one declared data root;
- one `SparseAvailabilityManifest`;
- a private provider bound to that exact manifest hash;
- one `EpisodeAssignment`.

It does not receive a raw filesystem provider, glob, directory scan, audit root,
full-volume index, or fallback path. Unknown IDs and non-manifest paths fail
before file I/O. All legal opens append deterministic ledger events and
content-addressed audit rows.

## 4. Patient-level split

Split patients before generating episodes, augmentations, registrations,
supports, or caches. Every observation, repeated scan, derived image, label, and
registration product associated with one patient inherits the same split.
Slice-level or modality-level splitting is prohibited.

The split registry must:

- map each patient ID to exactly one of `train`, `validation`,
  `t1_lesion_validation`, or `t5_final_audit`;
- be canonicalized and hashed;
- be validated jointly across every manifest used by an experiment;
- keep both evaluation cohorts mutually disjoint and disjoint from train and
  validation patients;
- group linked/repeated visits under one patient or subject-family identifier
  when linkage could leak anatomy;
- record site, scanner, protocol, and pathology strata used for balancing,
  without moving slices independently between splits.

All learned preprocessing, normalization statistics, registration confidence
calibration, model weights, and gradient-derived state are fit on the training
split only. Validation may select configurations. Aggregate results from the
patient-disjoint T1 lesion-validation evaluator may guide architecture, config,
threshold, and checkpoint-selection-rule decisions, including iterative
development. Its sparse inputs remain constrained to their predeclared manifest,
and its full volumes, target pixels, and lesion/ROI labels remain evaluator-only
and never become training tensors. The distinct T5 final-audit cohort is opened
only by the future final-audit evaluator after the complete evaluation protocol
is frozen.

## 5. Episode assignment is not availability

An immutable `EpisodeAssignment` contains:

```text
episode_id
manifest_hash
patient_id
context_ids
target_ids
assignment_hash
```

The canonical assignment hash binds the sorted, immutable context and target
ID tuples together with the episode ID, patient ID, and manifest hash.

Required invariants:

- context and target IDs are disjoint;
- every assigned ID exists in the same availability manifest;
- all assigned observations belong to the declared patient;
- at least one context and one target exist when reconstruction loss is used;
- assignment mutation is impossible after construction;
- multiple assignments may reuse the same manifest with different legal roles;
- constructing or changing an assignment does not alter the manifest hash;
- role assignment consumes no acquisition budget.

The legacy `AccessLevel.CONTEXT/TARGET` field may exist only as a migration
adapter for T0 tests. New Phase-1/T1 loaders, trainers, and caches must use the
episode assignment as the sole source of training role.

Context payloads may enter analytic features, encoder maps, deterministic
supports, Gaussian construction, and patient state. Target metadata needed for
rendering may be exposed, but target pixels, target-derived statistics, content
digests, labels, and cached features may not enter state construction.

## 6. Prediction-before-reveal

For every target, the legal order is:

```text
open context payloads
→ build immutable state version
→ expose target plane metadata
→ commit_target(target_id, state_version)
→ render from that pre-reveal state
→ register_prediction_receipt(
      target_id,
      state_version,
      plane_hash,
      renderer_version,
      prediction_digest
  )
→ reveal_target(receipt_capability)
→ compute target loss
```

The canonical target `plane_hash` is derived from immutable physical-plane
metadata in the availability manifest. `prediction_digest` binds the rendered
tensor bytes plus shape, dtype, supported mask, and renderer output convention.
A receipt must also be internally bound to its ledger, episode, assignment
hash, and commit record.

Reveal must reject:

- commit without a prediction receipt;
- a receipt for another target, episode, assignment, ledger, or state version;
- a receipt whose plane hash differs from the target manifest plane;
- a receipt whose renderer version or prediction digest is missing or invalid;
- a forged, revoked, or already consumed capability;
- a second reveal of the same target.

Registering a receipt asserts that rendering finished before target pixels
became available. It does not prove operating-system isolation or clinical
scanner behavior. Events for commit, receipt registration, reveal, and payload
open are ordered, canonical, and included in the ledger audit hash.

## 7. Registration assumptions and confidence

Cross-modality structural consistency is legal only for observations from the
same patient with declared physical transforms. Every registration product must
record:

- source observation IDs and modalities;
- reference frame and direction of the transform;
- registration method, software/version, parameters, and interpolation policy;
- transform and configuration hashes;
- whether registration used only manifest observations;
- a confidence value and its interpretation;
- valid-overlap mask and known failure regions.

Registration for the main cohort must not read a hidden dense volume, audit
image, audit label, or target pixel before its legal reveal. If registration
requires such data, it is a privileged preprocessing path and cannot support the
main claim.

Cross-modality `Z_str` consistency is evaluated only within valid physical
overlap and is explicitly weighted by registration confidence. Low-confidence
or missing registration disables that pairwise loss; it is not treated as a
negative pair and does not justify forcing alignment. `Z_app` is never required
to match across modalities. Registration confidence, overlap fraction, and
excluded pairs are logged per episode.

## 8. Missing-modality policy

A modality absent from a patient's availability manifest is genuinely missing:

- do not synthesize, impute, or borrow it to create context or targets;
- do not open a full-volume parent to manufacture missing slices;
- do not require every patient to contribute every modality;
- expose an explicit modality-availability mask to legal model components;
- sample targets only from observations that exist in the manifest;
- normalize losses and experiment summaries without silently dropping patients
  with missing modalities.

Matched E0/E1/E2 comparisons must use the same patients, availability masks,
episode assignments, and target modalities. Per-modality sample counts and
missingness patterns are reported. Performance on an unavailable modality is a
separate conditional reconstruction task and cannot be scored against hidden
ground truth in main training.

## 9. Evaluation isolation

The T1 lesion-validation and T5 final-audit cohorts are each physically and
logically isolated from training and from one another:

- separate patient IDs, patient-level split, storage root, credentials/mount,
  loader class, evaluator process, and cache namespace;
- a distinct, predeclared canonical sparse input manifest for the T1 gate
  cohort, containing the only pixels available to its reconstruction process;
- every T1 sparse-input-manifest change creates a new version and hash and is
  recorded before the affected evaluation; it is never an implicit expansion;
- no evaluation path or evaluator provider in a training process or training
  config;
- no gradient training, feature-statistic fitting, weight updates, patient-state
  reuse, or support construction for training from evaluation pixels or labels;
- no encoder features, registrations, crops, masks, summaries, or pretrained
  patient state derived from either evaluation cohort in training artifacts;
- T1 lesion-validation may return aggregate metrics and lesion/ROI diagnostics
  to guide architecture, config, threshold, and checkpoint-selection-rule
  decisions, but it returns no dense target, ROI label, per-pixel target tensor,
  or patient state to the training process;
- each T1 evaluation records its code, checkpoint, config, input-manifest,
  metric, margin, interval, and multiplicity-policy hashes so development use is
  auditable rather than hidden;
- T5 final audit remains sealed throughout T0.5/T1 and until architectures,
  configs, thresholds, checkpoint-selection rules, and analysis plans are all
  frozen;
- lesion/ROI labels are opened only by the corresponding isolated evaluator.

T1 lesion-validation feedback may accept, reject, or guide revision of the
developing representation. It must not be used for gradient updates, feature
normalization, dense supervision, target-derived inputs, or patient-specific
state reuse. Every development decision informed by this cohort must cite the
corresponding evaluation and artifact hashes. Before opening T5, the final
architecture, configs, thresholds, checkpoint-selection rules, and analysis
plans are frozen and hashed. T5 feedback cannot authorize model revision,
checkpoint reselection, threshold tuning, or T2 selection.

Oracle routing, E3, E4, or other privileged studies must use separate configs,
artifact namespaces, and explicit `PRIVILEGED_UPPER_BOUND` labels. Their
checkpoints cannot initialize or select the main method.

Every training run must terminate with a machine-checkable assertion that:

```text
opened observation IDs ⊆ assignment context IDs ∪ legally revealed target IDs
opened observation IDs ⊆ availability manifest IDs
opened T1-lesion-validation patient IDs during training = ∅
opened T5-final-audit patient IDs = ∅
non-manifest pixel opens = 0
evaluation pixel opens during training = 0
```

## 10. Leakage-positive controls

Leakage tests are blocking and must deliberately attempt forbidden behavior.
A test passes only when the bad access fails before returning bytes and the
ledger records no legal open for it.

Required positive controls include:

1. open a target as context before commit;
2. reveal a committed target before registering a prediction receipt;
3. use a receipt from another target, episode, state version, assignment, or
   ledger;
4. reuse a consumed receipt;
5. assign an unknown or cross-patient observation ID;
6. open a T1 lesion-validation or T5 final-audit ID through the training ledger;
7. access a path outside the bound root or through a symlink escape;
8. mutate a payload after manifest sealing;
9. place one patient in more than one split;
10. inject target bytes or target-derived features into analytic preprocessing,
    encoder input, support construction, or Gaussian state.

Controls must inspect event order and audit hashes, not only exception types.
At least one synthetic target payload should contain a sentinel byte pattern;
the pattern must be absent from all pre-reveal state, cache, and prediction
inputs. Removing or disabling a leakage-positive control blocks the gate.

## 11. Training roles and deployment acquisition cost

Context/target is an offline optimization role. Acquisition cost is a deployment
quantity. They use separate ledgers and must never share counters.

### 11.1 Offline training

- Sampling `context_ids` and `target_ids` costs exactly zero.
- Reassigning an observation across episodes costs exactly zero.
- Opening a legal context or revealing a legal sparse target does not simulate
  a deployment purchase.
- Historical acquisition cost may remain immutable metadata for later matched
  analyses, but it does not affect the training episode budget.

### 11.2 Deployment

Deployment starts with an explicit `Decimal` budget and a versioned cost
schedule. Cost values are parsed from canonical decimal strings and are never
round-tripped through binary float.

Charge the declared modality/plane-specific cost for:

- every bootstrap observation committed for the initial patient state;
- every subsequent committed observation;
- repeated acquisitions when the protocol declares them distinct billable
  observations;
- modality, orientation, thickness, or protocol-specific multipliers defined
  before evaluation.

The charge occurs at irreversible deployment commitment, before pixels are
opened, and remains charged if downstream rendering or processing fails. A
rejected commitment does not consume cost. Every cost event binds the
observation ID, modality, plane/protocol cost key, canonical decimal amount,
budget before/after, cost-schedule hash, and deployment-ledger hash.

Bootstrap observations are not free context. A target role in offline training
does not imply an acquisition charge. Conversely, a deployment observation is
charged regardless of whether it later serves as state evidence, quality
control, or evaluation.

## 12. Unsupported pixels and coverage reporting

Unsupported predictions are failures of current evidence coverage, not valid
zero-valued MRI intensity:

- renderer output for unsupported intensity remains `NaN`, never silent zero;
- reconstruction loss uses explicit valid-content and supported masks;
- the numerator and denominator of every masked loss are logged;
- unsupported fraction and physical coverage are reported per plane, modality,
  patient, and experiment;
- supported-only MAE/NMSE/PSNR/SSIM may be diagnostic but cannot be the sole or
  headline result;
- headline reporting includes coverage/failure rate and a declared penalty or
  failure treatment for unsupported regions;
- E0/E1/E2 use the same coverage policy and downstream support locations.

Raw Gaussian `support_mass` is not calibrated uncertainty and must not determine
coverage under an arbitrary common amplitude gauge. T1 uses the separately
named, gauge-invariant coverage policy approved by T0.5.

## 13. Prohibited hidden-target use

The following are prohibited in the main path:

- non-manifest or audit pixels as training inputs or targets;
- dense parent volumes used for slice selection, registration, normalization,
  cropping, artifacts, feature targets, support placement, or pseudo-labels;
- target pixels or target-derived values before receipt-gated reveal;
- hidden target content used for neighbor search, primitive selection,
  coverage, early stopping, or topology decisions;
- T5 final-audit metrics used to tune training, select checkpoints, revise
  thresholds, or change analysis plans;
- teacher or pretrained dense features presented as the main method;
- silent filling of missing modalities or unsupported pixels.

Target plane geometry may be known before reveal because it is immutable
manifest metadata. Target intensity, label, content digest, and any derivative
of them may not be known to the model or trainer before reveal.

## 14. Required run record and gate

Each T0.5/T1 run records:

- dataset license/approval reference and cohort class;
- patient-split registry hash;
- T1 lesion-validation sparse input-manifest hash when that evaluator runs;
- sealed T5 final-audit registry hash without opening its payloads;
- availability manifest ID, canonical hash, and sealed content-binding version;
- episode ID, assignment hash, context IDs, and target IDs;
- registration/configuration hashes and confidence summaries;
- modality availability;
- ordered ledger events, opened-file audit hash, and prediction receipts;
- training and deployment ledgers separately;
- cost-schedule hash and exact decimal cost events for deployment studies;
- unsupported fraction and coverage policy;
- random seeds, code commit, config hash, checkpoint hash, and artifact hashes;
- explicit zero counts for non-manifest and evaluation-cohort pixel access
  during training.

T0.5-L is blocked unless role/availability separation, receipt-gated reveal,
patient-level splitting, content binding, exact deployment accounting, and all
leakage-positive controls pass. T1 results are invalid if any opened training
pixel is outside the fixed availability manifest or belongs to either
evaluation cohort.

This protocol authorizes no T2 support-anchor field, T3 propagation, T4 routing,
or T5 full-volume export implementation.
