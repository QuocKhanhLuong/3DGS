# Sol-High Main-Branch Audit Plan

## Frozen audit target

- Branch: `main`
- HEAD: `0efeb94af72ffa067769e19afcd19ad358feefd2`
- Upstream: `origin/main` at the same commit
- Dirty state present before the audit:
  - modified: `.DS_Store`, `src/.DS_Store`, `tests/.DS_Store`
  - untracked: `configs/.DS_Store`, `docs/.DS_Store`,
    `docs/architecture/point_guided_reward_cost_trajectory.html`,
    `scripts/.DS_Store`
- Freeze rule: every report must identify this HEAD. Workers must not switch,
  reset, stash, clean, stage, commit, or overwrite the pre-existing dirty files.
- Audit mutation rule: before the Human Gate, workers may write only their
  assigned report under `reports/audit/`; production code, tests, configs,
  plans, and existing documentation are read-only.

## Current system map

This repository is a Python/PyTorch research system rather than a web product.
There is no frontend, HTTP API, user account, role, or interactive viewer in
the active point-guided path. The human-facing boundaries are Python APIs, two
CLI modules, server shell scripts, JSON/JSONL/CSV/checkpoint artifacts, and
saved NIfTI predictions.

| Component | Current owner | Responsibility and boundary |
| --- | --- | --- |
| Canonical geometry | `src/smagm/contracts/coordinates.py` | RAS-mm XYZ and tensor DHW affine semantics. |
| Locked frontend | `src/smagm/features/point_guided/{model,medicalnet_resnet10,semantic_prior,points,refinement,pou,triplane_projection,swt_haar,spectral_anchor,spectral_query,cross_plane_consistency}.py` | One shared MedicalNet traversal from T1/T2/FLAIR to coarse semantics, deterministic/refined points, sparse PoU, static B/A planes, reliability, and 168-d point spectral evidence. |
| Gate C | `state_init.py`, `reward.py`, `trajectory_cost.py`, `trajectory_solver.py`, `updater.py`, `writeback.py`, `trajectory.py` | Bounded target-free reward-cost route over fixed point evidence, producing final dynamic tri-planes and diagnostics. |
| Gate D | `decoder.py`, explicit methods in `model.py` | Geometry-aware chunked final-Z-only implicit decoding. Generic `forward()` remains fail-closed. |
| Gate E | `losses.py`, `reward_supervision.py`, `training_objective.py`, explicit methods in `model.py` | T1ce enters only after the target-free context and prediction exist; computes reconstruction, counterfactual reward, trajectory, and update objectives. |
| Full-volume BraTS adapter | `src/smagm/data/brats21_point_guided.py` | Discovers NIfTI subjects, validates registered geometry, converts XYZ arrays to DHW tensors, derives input-only masks/normalization, carries target/segmentation separately, batches only common geometry, and builds deterministic subject splits. |
| Training runtime | `src/smagm/training/point_guided.py`, `src/smagm/cli/point_guided_train.py` | Config composition, model/optimizer construction, single-process or DDP execution, AMP, training and validation, semantic auxiliary supervision, logging, checkpoints, resume, and preflight. |
| Gate-G inference/evaluation | `baseline_inference.py`, `baseline_metrics.py`, `baseline_checkpoint.py`, `src/smagm/cli/point_guided_eval.py` | Validated checkpoint load, deterministic exact-no-revisit inference, post-inference metrics, per-subject prediction output, and aggregate report. |
| Runtime packaging | `configs/training/**`, `configs/evaluation/**`, `scripts/point_guided_*.sh`, `docs/POINT_GUIDED_SERVER_RUN.md` | Server presets and launch instructions. No container/orchestrated deployment exists for this research path. |
| Verification/governance | `tests/features/point_guided/**`, `tests/data/test_brats21_point_guided.py`, `tests/test_codegraph.py`, `quality/**`, `.github/workflows/ci.yml` | Synthetic contracts, data tests, codegraph scope checks, and general CI. Passing software checks are not experimental evidence. |
| Legacy research system | `src/smagm/{anchors,fields,memory,routing,reconstruction,training,evaluation,cli,data}` outside the explicit additive imports | Retained code that must not be reused by the locked frontend except the separately authorized server owner. It is a migration/coexistence risk, not an automatically active dependency. |

## Documented intended architecture

The active repository authority (`AGENTS.md`, `PLAN.md`,
`PLAN_GATE_C_D_E.md`, `PLAN_GATE_F_G.md`, `CODEBASE.md`, and the task-scoped
`CODEGRAPH.json`) intends this flow:

```text
T1/T2/FLAIR [B,3,D,H,W]
  -> one frozen-by-default MedicalNet ResNet10 traversal
  -> coarse normal/edema/core semantics
  -> deterministic initial points -> observation-only bounded refinement
  -> fixed 4-mm sparse semantic-aware PoU
  -> static B planes -> fixed SWT-Haar A planes
  -> geometry-aware point queries and deterministic 168-d f_spec
  -> bounded Gate-C reward-cost trajectory and final dynamic Z
  -> final-Z-only Gate-D implicit decoder
  -> absolute T1ce prediction
  -> only then optional Gate-E target supervision
  -> Gate-F optimizer/training/checkpoint runtime
  -> target-free Gate-G checkpoint inference and held-out metrics/artifacts
```

Scientific authority is fail-closed: target data may not affect the frontend,
route, stopping, or inference; point movement stays within 2 mm; writeback uses
4-mm physical support; no second encoder, target lookup, attention/transformer,
FFT, legacy reconstruction bypass, or generic `forward()` is permitted.

The authoritative status is: Phases 1-7 and Gates A-E implemented, Gate F
F1/F2 complete, F3/F4 software ready but not executed on the server, Gate G
G1-G4 software complete but trained-checkpoint/held-out evidence pending, and
Gate H denied. Older README, frontend architecture, and quality passages still
describe Gate F/G as inactive; workers must distinguish obsolete prose from a
runtime defect.

## Critical flows

### 1. Operator and runtime entry

The operator chooses a checked-in JSON preset and invokes
`point_guided_train.py`, `point_guided_eval.py`, or the corresponding shell
script. There is no authentication or role system. The relevant boundary is
local/server filesystem authority over the dataset, output root, checkpoint,
and process environment.

### 2. Dataset discovery, eligibility, and split persistence

The training runtime inventories BraTS subject directories, excludes invalid
subjects explicitly, creates a deterministic subject-level train/val/test
split, and persists `split.json` plus provenance. Evaluation must reuse that
exact split or an explicitly provided split file, never silently make a new
held-out cohort.

### 3. Training sample and target boundary

The adapter loads T1/T2/FLAIR observations and derives the legal brain mask and
normalization only from those inputs. T1ce and segmentation are loaded into
separate typed fields. The trainer first asks the model for a target-free
training context and prediction, then passes T1ce to the Gate-E objective and
segmentation only to the training-only semantic auxiliary objective.

### 4. Model execution

`PointGuidedMRIModel._forward_frontend_with_gate_b_context()` performs the one
shared frontend traversal. The explicit training, reconstruction, trajectory,
and baseline-inference methods compose Gates C/D/E/G. Generic `forward()`
raises. Gate-G forces eval plus no-grad and restores the previous training
mode.

### 5. Optimizer, distributed execution, and state transitions

The trainer builds the locked trainable set, executes forward/objective/
backward/step under optional AMP and DDP, reduces validation statistics, logs
epoch records, and updates last/best/resume checkpoints. DDP must not duplicate
or omit validation subjects and must keep rank-specific effects coordinated.

### 6. Checkpoint and resume persistence

Resume checkpoints include model, optimizer, scheduler/scaler/RNG and run
metadata; clean inference checkpoints contain the validated baseline state and
architecture metadata. Writes are intended to be atomic. A run's split and
normalization provenance must remain coupled to any checkpoint used later.

### 7. Held-out evaluation and artifacts

The evaluation CLI resolves the original split, loads the checkpoint strictly,
loads test subjects without allowing targets into inference, runs deterministic
Gate-G inference, computes metrics afterward, writes subject NIfTI predictions
with affine metadata, and emits aggregate JSON. This is the repository's
closest analogue to artifact generation and viewer consumption; no viewer is
implemented.

### 8. Deployment and operations

Shell scripts map GPUs and config paths into Python module invocations. The
server environment must supply PyTorch, nibabel, data, optional MedicalNet
weights, writable artifact paths, and correct distributed variables. CI is a
software gate only and does not reproduce real server GPU/data behavior.

## High-risk changes

- Relative to merge `6f754ed`, the point-guided work changed 77 files with
  about 19,912 insertions. This is a large additive system with many cross-file
  contracts landing faster than independent runtime evidence.
- `9a33058` alone changed 29 files and about 2,455 inserted lines, including
  data behavior, training, reward, model, inference, and tests. Migration
  completeness and target-boundary consistency are high risk.
- `11ba203` then hardened training/evaluation, split handling, normalization,
  DDP validation, and metrics across 13 files. Older callers/configs/tests may
  still encode the pre-hardening contract.
- `39a39d3` reverted an attempted paired reward change and restored the 126-d
  descriptor. Old 124-d/paired assumptions, checkpoint incompatibility, or
  partially reverted tests/configuration are explicit regression targets.
- Active authority and older public docs disagree about Gate F/G status. This
  can mislead operators even if code is correct.
- Real BraTS, GPU, DDP, trained-checkpoint, memory, throughput, and held-out
  evidence are intentionally absent. Tests must not be reported as substitutes.
- The repository retains legacy training/data/reconstruction packages beside
  the additive point-guided system. Any accidental import or entrypoint mix can
  violate the locked architecture.
- Pre-existing untracked HTML architecture output is outside audit ownership
  and must not be treated as authoritative or modified.

## Worker decomposition

All workers audit frozen HEAD `0efeb94af72ffa067769e19afcd19ad358feefd2`.
Each worker starts with the named `scripts/codegraph.py` task, remains
read-only except for its assigned report, and follows the required finding
schema.

### AGY-A — Architecture, wiring, and documentation authority

- Objective: prove which point-guided and legacy modules are actually wired,
  identify dead/duplicate paths and partial migrations, and reconcile the
  status authority hierarchy.
- Scope: `AGENTS.md`, `README.md`, `CODEBASE.md`, `PLAN*.md`,
  `CODEGRAPH.json`, `docs/architecture/POINT_GUIDED_FRONTEND.md`, public
  `__init__.py` exports, and imports/call paths rooted in `model.py` and the two
  point-guided CLI modules.
- Flow: operator entrypoint -> imports -> model/data/training/evaluation owner.
- Questions: Is legacy code reachable? Are all completed Gate APIs exported
  consistently? Do docs and task scopes describe current code? Did the reward
  revert leave stale callers or types?
- Commands: `python scripts/codegraph.py --task frontend`, `--task trajectory`,
  `--task server_pipeline`; targeted import tracing; `git diff 6f754ed..HEAD`;
  focused static reads.
- Expected output: `reports/audit/workers/agy-a-architecture.md` with call-path
  evidence, doc mismatches separated from code defects, and no implementation.
- Done when every active point-guided entrypoint is mapped to its owner and all
  suspected legacy/revert seams are classified with evidence.

### AGY-B — Frontend, trajectory, decoder, and target-boundary correctness

- Objective: audit the scientific/tensor contracts from observation through
  final-Z decoding and Gate-E target entry.
- Scope: `src/smagm/features/point_guided/**` limited by the `frontend`,
  `trajectory`, `decoder`, `supervision`, and `baseline_inference` routers;
  corresponding focused tests.
- Flow: observation -> MedicalNet -> semantics/points/PoU -> B/A/f_spec ->
  route -> decoder -> supervision/inference.
- Questions: one traversal? exact dimensions/order? affine correctness?
  target-free route? deterministic Gate-G? mode restoration? descriptor revert
  complete? illegal bypass or stale API?
- Commands: all five task routers; focused point-guided tests; read-only
  synthetic reproductions when necessary; `git diff 9a33058^..39a39d3` on
  reward-related files.
- Expected output: `reports/audit/workers/agy-b-model-flow.md` with concrete
  tensor/call traces and reproducible findings.
- Done when every public model method and target-entry seam is checked against
  the locked authority and P0-P2 claims have direct code or execution evidence.

### AGY-C — Data, split provenance, checkpoint, and artifact persistence

- Objective: verify subject discovery, NIfTI/affine conversion, normalization,
  split integrity, batching, checkpoint atomicity/compatibility, and artifact
  provenance.
- Scope: `src/smagm/data/brats21_point_guided.py`,
  `baseline_checkpoint.py`, relevant data/checkpoint tests, config fields, and
  persistence calls in training/evaluation.
- Flow: filesystem cohort -> structural inventory -> split -> sample/batch ->
  checkpoint/split -> evaluation prediction/report.
- Questions: target leakage? mixed-affine behavior? missing/duplicate IDs?
  exact split reuse? resume drift? collision/overwrite risk? atomic writes?
  target normalization consistent with metric data range?
- Commands: `python scripts/codegraph.py --task server_pipeline`; focused data
  and checkpoint/server-pipeline tests; temporary-directory read-only
  reproductions that write only under `/tmp`.
- Expected output: `reports/audit/workers/agy-c-data-persistence.md`.
- Done when all persisted identities and provenance links are traced and each
  failure claim includes a minimal reproduction or precise invariant proof.

### AGY-D — Training, DDP, multi-process concurrency, and resource lifecycle

- Objective: answer what happens under simultaneous processes/ranks and audit
  optimizer ownership, gradient flow, DDP, AMP, validation, resume, and cleanup.
- Scope: `src/smagm/training/point_guided.py`,
  `baseline_training.py`, train CLI/configs/scripts, related tests.
- Flow: CLI config -> distributed init -> split/loaders -> training context ->
  loss/backward/step -> reductions -> logs/checkpoints -> teardown.
- Questions: rank races on directories/files? sampler duplication? per-rank
  RNG correctness? cleanup on errors? shared output collision? scheduler/resume
  consistency? all and only authorized trainables optimized?
- Commands: `python scripts/codegraph.py --task baseline_training` and
  `--task server_pipeline`; focused CPU trainer tests; safe multi-process tests
  if supported; static exception-path analysis.
- Expected output: `reports/audit/workers/agy-d-training-concurrency.md`.
- Done when single-process and DDP state machines are documented, concurrency
  hazards are evidence-backed, and real-GPU-only gaps are explicitly labeled.

### AGY-E — Evaluation, runtime artifacts, and operator flow

- Objective: trace checkpoint-to-held-out-results end to end and verify errors,
  empty states, reproducibility, output correctness, and operator recovery.
- Scope: `baseline_inference.py`, `baseline_metrics.py`,
  `point_guided_eval.py`, evaluation config/script/server guide, and evaluation
  tests.
- Flow: checkpoint + split -> test dataset -> target-free inference -> metrics
  -> NIfTI/JSON output.
- Questions: strict checkpoint contract? held-out isolation? no-grad/eval?
  metric masks/ranges? affine preservation? deterministic outputs? partial-run
  cleanup/resume? missing-subject/error behavior? deep filesystem path safety?
- Commands: `python scripts/codegraph.py --task baseline_inference` and
  `--task server_pipeline`; CLI help; focused inference/metric/eval tests;
  temporary synthetic end-to-end smoke where feasible.
- Expected output: `reports/audit/workers/agy-e-evaluation.md`.
- Done when the actual operator-visible evaluation lifecycle is explained and
  every artifact field is traced to source or marked unavailable.

### AGY-F — Tests, CI, deployment, and operational readiness

- Objective: run the relevant software gates, audit whether they can detect
  critical failures, and compare CI/local/server assumptions.
- Scope: `.github/workflows/ci.yml`, `pyproject.toml`, `quality/**`, all
  point-guided/data tests, configs, server scripts and documentation.
- Flow: checkout -> dependency environment -> CI/tests -> preflight -> server
  training/evaluation command.
- Questions: what really runs in CI? skipped optional tests? mock-heavy gaps?
  missing target/concurrency/persistence/negative tests? shell/env/CUDA/DDP
  assumptions? secrets or unsafe paths? logs/observability? build/install drift?
- Commands: `python scripts/codegraph.py --task quality`, `--task tests`, and
  `--task server_pipeline`; smallest suites then broader point-guided suite;
  `compileall`; shell syntax checks; CLI help; `git diff --check`.
- Expected output: `reports/audit/workers/agy-f-verification-ops.md` with exact
  pass/fail/skip counts and a separate unverified-gates section.
- Done when executed checks are fully enumerated and no software check is
  misrepresented as real data, GPU, DDP, performance, or scientific evidence.

### AGY-G — Recent-regression hunter

- Objective: independently audit changes from `6f754ed` through frozen HEAD,
  with special attention to the handoff, runtime-hardening, and reward revert.
- Scope: commits `bc4f6bf`, `fe9fc03`, `11ba203`, `9a33058`, `39a39d3` and
  their changed point-guided/config/test/doc files.
- Flow: old contract -> changed contract -> current callers -> tests -> runtime
  compatibility.
- Questions: incomplete migrations? stale config/checkpoint schema? removed
  behavior still referenced? tests updated around bugs rather than contracts?
  runtime hardening inconsistently applied?
- Commands: task routers matching each changed file; `git show`, `git diff`,
  caller tracing, and focused tests/reproductions.
- Expected output: `reports/audit/workers/agy-g-regressions.md`, including a
  per-important-change old/new/callers/tests table.
- Done when every high-risk recent change is classified as clean, defective,
  risky, or documentation-only with current-HEAD evidence.

## Phase sequencing and review gates

1. Complete all seven read-only first passes at the frozen HEAD.
2. Run an independent system-wide reviewer only after all worker reports exist.
3. Sol-High validates and deduplicates every important finding; P0/P1/P2 items
   require concrete evidence or are downgraded to `NEEDS_REPRO`.
4. Build the definitive current flow and docs-vs-code comparison.
5. Run a final evidence-bounded editorial critique.
6. Write `reports/main-audit-0efeb94.md` and stop at the Human Gate.
7. No production/test/config/doc remediation occurs until the human explicitly
   accepts finding IDs.
