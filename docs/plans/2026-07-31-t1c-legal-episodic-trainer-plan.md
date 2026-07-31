# T1-C Plan — Legal Episodic Trainer and Matched T1 Attribution

Date: 2026-07-31  
Status: **IMPLEMENTED SOFTWARE PLAN — HUMAN GATE PENDING**
Depends on: T1-B implemented software tranche and committed Human Gate `PASS`  
Stable owners: `src/smagm/data/`, `src/smagm/losses/`, `src/smagm/training/`, `configs/`, `experiments/`

## 1. Purpose

T1-C converts the executable T1-B evidence path into a legal, reproducible,
context-to-target training system for permanently sparse multi-sequence MRI.
It answers one bounded question:

> Can E0, E1, and E2 be trained and compared through the same legal sparse
> episode, fixed physical supports, fixed Gaussian head, renderer, optimization
> opportunity, and medical-fidelity protocol without target leakage?

T1-C is still part of the fixed-topology T1 attribution bridge. It does not
implement the support-anchor field, anchor-aligned Gaussian birth, propagation,
adaptive topology, active routing, or final full-volume export.

The governing references are:

- [`../reconstruction/phases/01_DIRECT_SPARSE_TRAINING.md`](../reconstruction/phases/01_DIRECT_SPARSE_TRAINING.md)
- [`../../CODEBASE.md`](../../CODEBASE.md)
- [`../codex/T1B_TEACHER_FREE_ENCODER.md`](../codex/T1B_TEACHER_FREE_ENCODER.md)
- [`../protocols/permanently_sparse_training.md`](../protocols/permanently_sparse_training.md)
- [`../../quality/checklists.json`](../../quality/checklists.json)

## 2. Authorization boundary

This document remains the implementation plan, not a Human Gate decision.

The repository owner explicitly authorized the T1-C implementation tranche on
2026-07-31. The existing T1-B Human Gate and that implementation authorization
do not establish that the representation improves reconstruction.

The completed implementation remains bounded as follows:

- `src/smagm/training/` contains only the fixed-topology T1-C trainer;
- T1-C config and synthetic diagnostics make software claims only;
- no T2 package or placeholder is created;
- `docs/codex/README.md` reports T1-C as implemented with Human Gate pending.

## 3. Scientific and software scope

T1-C may contain:

- deterministic legal episode sampling from a role-free sparse manifest;
- manifest-bound payload decoding and context-only preprocessing;
- explicit registration and normalization records;
- context-only feature encoding and exact cache insertion;
- deterministic fixed-support sampling shared by E0/E1/E2;
- the existing fixed Gaussian head and canonical-RAS Gaussian conversion;
- frozen patient-state creation before target commitment;
- render, prediction receipt registration, target reveal, and live-loss handoff;
- supported-mask-aware reconstruction objectives;
- structural auxiliary objective composition;
- optimizer, precision, gradient, checkpoint, and bounded logging policy;
- independent E0/E1/E2 training runs under a matched protocol;
- immutable experiment and artifact provenance;
- CPU synthetic integration and bounded real-data smoke execution.

T1-C must not contain:

- learned anchor candidates or anchor consolidation;
- an anchor-local structural field;
- structural/volumetric dual-bank memory beyond the existing fixed Gaussian
  attribution state;
- Gaussian propagation, birth, split, merge, or prune;
- active candidate routing or acquisition utility;
- full-grid reconstruction, NIfTI export, or sealed T5 audit code;
- teacher or foundation-model supervision in the main path.

## 4. Locked legal state machine

The trainer must reuse the existing T0.5 contracts. It must not duplicate or
weaken `EpisodeAssignment`, `EpisodeLedger`, `FrozenPatientState`,
`EpisodeController`, or `PredictionRegistrar`.

```text
SparseAvailabilityManifest
→ deterministic EpisodeAssignment
→ EpisodeLedger
→ open every assigned context payload
→ decode and normalize context only
→ E0/E1/E2 encoder
→ exact feature cache
→ deterministic fixed supports
→ common fixed Gaussian head
→ canonical-RAS GaussianBatch
→ FrozenPatientState
→ expose target geometry only
→ commit target against frozen state version
→ EpisodeController.render_and_register
→ receive live RenderResult + receipt
→ reveal target payload
→ decode target after reveal
→ compute legal supported-mask loss on live prediction
→ backward
→ optimizer step
→ immutable episode/provenance report
```

Hard ordering rules:

1. `EpisodeAssignment` is finalized before any payload is opened.
2. Every assigned context observation is opened before state freeze.
3. No target bytes, target normalization statistic, target feature, or
   target-derived support may exist before receipt-gated reveal.
4. Target geometry may be exposed before rendering; target pixels may not.
5. The prediction used for the loss remains attached to autograd.
6. Receipt hashing uses only a detached audit copy.
7. A target receipt is single-use and bound to the exact ledger, assignment,
   state version, target plane, renderer version, Gaussian state, and output.
8. The renderer remains pure and owns no episode state.

## 5. Stable software ownership

### 5.1 Data

`src/smagm/data/io.py`

- Decode only bytes returned by a legal provider or ledger.
- Produce tensors plus immutable source/provenance metadata.
- Reject arbitrary-path reads.
- Never inspect target payload before reveal.

`src/smagm/data/normalization.py`

- Fit or resolve normalization from legal context only.
- Return an immutable, hashable `PreprocessingRecord`.
- Apply the frozen context-derived transform to a revealed target only after
  receipt-gated reveal.
- Record invertibility and modality policy where applicable.

`src/smagm/data/registration.py`

- Validate existing registration metadata and confidence.
- Declare transform direction and canonical RAS-mm semantics.
- Do not hide a learned registration model inside the loader.

`src/smagm/data/episodes.py`

- Deterministically construct `EpisodeAssignment` from a fixed manifest.
- Return IDs and metadata, never target pixels.
- Support identical assignment schedules across E0/E1/E2.

### 5.2 Losses

`src/smagm/losses/reconstruction.py`

- Implement typed supported-mask-aware intensity loss.
- Optionally add declared gradient and frequency components.
- Return component values, legal pixel count, supported fraction, and explicit
  skipped/failure status.

`src/smagm/losses/compose.py`

- Compose reconstruction and existing structural losses without opening data.
- Keep every component switchable and separately logged.
- Do not silently renormalize away an empty legal target mask.

### 5.3 Training

`src/smagm/training/episode.py`

- Own the legal orchestration from context opening through target reveal.
- Reuse T0.5 controller and receipt contracts.
- Return a typed live prediction/loss handoff and immutable audit summary.

`src/smagm/training/objective.py`

- Resolve reconstruction and structural objectives from typed results.
- Apply only explicit masks and declared weights.

`src/smagm/training/sampling.py`

- Produce deterministic patient, assignment, and target schedules.
- Bind all schedules to manifest, split, and seed hashes.

`src/smagm/training/schedule.py`

- Describe structural warm-up, joint sparse reconstruction, and
  reconstruction-dominant refinement.
- Stage names must remain training policy, not model architecture.

`src/smagm/training/trainer.py`

- Own optimizer, AMP/precision, gradient clipping, accumulation, checkpointing,
  validation cadence, early stopping policy, and bounded logging.
- Fail closed on non-finite loss or gradient.
- Never bypass the legal episode orchestrator.

`src/smagm/training/metrics.py`

- Record unsupported fraction, structural collapse, gradient health, parameter
  count, cache bytes, runtime, and legal episode statistics.

`src/smagm/training/provenance.py`

- Bind commit, dirty state, environment, hardware, config, manifest, split,
  assignment schedule, preprocessing, encoder state, checkpoint, and artifacts.

### 5.4 Entrypoints and experiment files

- `src/smagm/cli/train.py`: thin resolved-config entrypoint.
- `configs/data/`: manifest, split, registration, preprocessing.
- `configs/model/`: E0/E1/E2 and common downstream contract.
- `configs/training/`: optimizer, schedule, precision, objective weights.
- `configs/experiments/`: matched T1 compositions.
- `experiments/manifests/`: immutable run manifests.
- `experiments/reports/`: patient-level development reports.

Do not place model logic in configs, CLI code, or experiment scripts.

## 6. Minimal typed contracts

The implementation may refine names while preserving these responsibilities.

```python
@dataclass(frozen=True)
class DecodedObservation:
    observation_id: str
    patient_id: str
    modality_id: str
    image: torch.Tensor              # [1, H, W]
    valid_mask: torch.Tensor         # [1, H, W], bool
    plane: PhysicalPlane
    payload_sha256: str
    decoder_config_hash: str


@dataclass(frozen=True)
class PreprocessingRecord:
    policy_id: str
    modality_id: str
    fitted_from_context_ids: tuple[str, ...]
    parameters_hash: str
    config_hash: str


@dataclass(frozen=True)
class LegalEpisodeStep:
    assignment_hash: str
    state_version: str
    target_id: str
    prediction: RenderResult         # live tensors
    target: torch.Tensor             # available only after reveal
    target_valid_mask: torch.Tensor
    receipt_record_hash: str
    audit_hash: str


@dataclass(frozen=True)
class ReconstructionLossResult:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]
    legal_pixel_count: int
    supported_fraction: float
    status: str                      # OK or typed skip/failure reason
```

No contract may store pre-reveal target pixels or target-derived features.

## 7. Reconstruction-loss semantics

The minimum legal mask is:

```python
legal_loss_mask = target_valid_mask & ~render_result.unsupported_mask
```

Requirements:

- unsupported pixels are excluded from intensity comparison but are reported;
- the trainer may not present a supported-only metric without coverage;
- an empty legal mask returns a typed skipped/failure result;
- a skipped episode cannot silently contribute zero loss;
- non-finite prediction, target, or component fails closed;
- target normalization uses only the frozen context-derived preprocessing
  record;
- intensity, gradient, and frequency terms remain separately logged;
- structural auxiliary losses never receive hidden target pixels.

A recommended initial composition is:

\[
\mathcal L_{T1C}
=
\mathcal L_{intensity}
+
\lambda_{grad}\mathcal L_{gradient}
+
\lambda_{freq}\mathcal L_{frequency}
+
\lambda_{str}\mathcal L_{structural}.
\]

The exact weights are experiment configuration, not hard-coded scientific
claims.

## 8. E0/E1/E2 fairness contract

Every compared variant must use the same:

- patient-level split;
- sparse availability manifests;
- deterministic episode assignments and target order;
- context-derived preprocessing policy;
- physical support positions and count;
- fixed Gaussian-head architecture and parameter opportunity;
- renderer profile and support policy;
- optimizer family, number of update opportunities, and stopping rule;
- checkpoint-selection rule;
- evaluation planes and legal masks;
- hardware class, precision, seed schedule, and accounting method.

Each run must have independently initialized and trained weights. Feature caches
must bind encoder state and cannot be shared across incompatible runs.

Report separately:

- encoder and adapter parameters;
- Gaussian-head parameters;
- analytic preprocessing operations;
- FLOPs or declared operation count;
- wall-clock training and inference time;
- peak memory;
- cache bytes;
- support count and coverage.

## 9. Training schedule

### Stage A — structural diagnostic warm-up

- Use legal observed slices only.
- Optimize the authorized T1-B structural objectives.
- Record per-channel variance and collapse diagnostics.
- Do not infer representation value from proxy loss alone.

### Stage B — legal joint sparse reconstruction

- Run the exact context-to-target state machine.
- Optimize reconstruction plus declared structural objectives.
- Compare E0/E1/E2 with matched opportunity.

### Stage C — reconstruction-dominant refinement

- Reduce auxiliary weights according to a resolved schedule.
- Keep legality, support topology, and downstream head fixed.
- Select checkpoints only by the predeclared development rule.

## 10. Required automated tests

### Data and legality

- arbitrary-path payload reads are rejected;
- normalization statistics use context only;
- target bytes cannot be decoded before receipt-gated reveal;
- assignments are deterministic and patient-consistent;
- sealed T5 audit cohorts cannot enter the trainer;
- leakage-positive controls fail closed.

### Episode state machine

- all assigned context is opened before state freeze;
- target geometry can be exposed without target payload;
- prediction precedes reveal;
- wrong target, state, plane, assignment, ledger, renderer, or reused receipt is
  rejected;
- live prediction retains gradient after receipt registration;
- target never enters state construction.

### Loss and numerics

- legal mask equals target-valid intersect renderer-supported;
- empty legal comparison is typed and visible;
- constant and low-signal cases remain finite;
- unsupported fraction is reported;
- gradients reach authorized E1/E2 encoder and common Gaussian head;
- E0 remains a valid analytic baseline;
- amplitude gauge is applied exactly once.

### Fairness and reproducibility

- E0/E1/E2 consume identical assignment schedules;
- independent run state and cache hashes differ when expected;
- config and experiment manifests are deterministic;
- checkpoint provenance binds exact commit and environment;
- dirty-tree runs are rejected for gate evidence.

### Integration

A CPU synthetic test must execute:

```text
manifest
→ assignment
→ legal context open
→ encode/cache
→ fixed supports/Gaussians
→ freeze/commit/render/register/reveal
→ supported-mask loss
→ backward
```

## 11. Diagnostic CLI and smoke run

Required entrypoint:

```bash
python -m smagm.cli.train --help
```

Provide a CPU synthetic configuration that runs one legal optimizer step for
E0, E1, and E2. Report:

- variant;
- commit and config hash;
- assignment hash;
- state version;
- receipt record hash;
- legal target pixel count;
- supported fraction;
- loss components;
- encoder and head gradient norms;
- parameter count;
- runtime and cache bytes;
- opened context IDs and revealed target ID.

A later bounded real-data smoke run may validate I/O and resource assumptions.
It is not reconstruction evidence by itself.

## 12. Quality checklist activation

The implemented tranche converts the T1-C `planned` checks in
`quality/checklists.json` into exact `pytest`, `command`, or `file` evidence
covering:

- legal episode state machine;
- prediction-before-reveal;
- no target tensor in patient state;
- supported-mask loss and typed empty-mask result;
- gradient reachability;
- matched E0/E1/E2 schedules;
- independent weights and cache state;
- immutable run provenance;
- absence of T2 packages and placeholders.

The automated verdict may become `PASS`, but the phase remains pending until a
committed Human Gate decision exists.

## 13. Scientific gates after implementation

T1-C software completion is necessary but not sufficient to advance to T2.
The following evidence must be separately accepted:

### T1-F — feature validity

- aligned features under declared transforms;
- no structural collapse;
- registered matches outperform mismatches;
- local differential information remains recoverable.

### T1-R — reconstruction attribution

- E2 improves both E0 and E1 under the matched protocol;
- the fixed-topology Gaussian path beats the declared interpolation floor;
- improvement is not explained by more parameters, primitives, steps, or
  hidden compute.

### T1-M — medical fidelity

- patient-disjoint lesion/ROI development evaluation;
- predeclared estimands, margins, intervals, multiplicity, coverage, and
  failure rules;
- no validation or audit pixels enter training or caches;
- no meaningful lesion/ROI or boundary regression.

No T2 implementation begins before an explicit Human decision accepts the
applicable T1-F, T1-R, and T1-M evidence.

## 14. Stop and demotion rules

- If E0 and E2 are equivalent under the predeclared criterion, remove the
  learned encoder from the novelty path.
- If E1 and E2 are equivalent, remove the analytic-scaffold claim.
- If structural objectives improve proxies but not target reconstruction,
  retain them only as diagnostics.
- If the fixed-topology Gaussian baseline does not beat interpolation, stop
  before T2 and repair geometry, data, coverage, renderer, or optimization.
- Any target, non-manifest, lesion-validation, or sealed-audit leakage
  invalidates the run.
- Global metric improvement with meaningful lesion/ROI regression fails T1-M.

## 15. Suggested implementation commits

1. `feat(data): add legal decoding normalization and episode sampling`
2. `feat(losses): add supported-mask reconstruction objectives`
3. `feat(training): add legal episode orchestration`
4. `feat(training): add trainer schedule metrics and provenance`
5. `feat(cli): add T1 legal training entrypoint and synthetic config`
6. `test(t1c): add legality fairness autograd and reproducibility gates`
7. `docs(codex): add T1-C executable handoff`

## 16. Luna High implementation prompt boundary

When T1-C is explicitly authorized, the coding prompt must begin with:

```text
Implement only the approved T1-C legal episodic trainer described by the active
plan, CODEBASE.md, Phase-1 theory, and T1-C quality checklist.

Reuse EpisodeAssignment, EpisodeLedger, FrozenPatientState, EpisodeController,
PredictionRegistrar, the T1-B encoder/cache, deterministic supports, fixed
Gaussian head, GaussianBatch, and render_plane. Do not duplicate or weaken
those contracts.

Do not implement anchors, fields, Gaussian propagation, adaptive topology,
routing, full-volume reconstruction, or T2+ placeholders. Stop when a required
scientific choice is not frozen instead of inventing it.
```

This boundary governed the implemented tranche and remains the scope limit for
T1-C rework while its Human Gate is pending.
