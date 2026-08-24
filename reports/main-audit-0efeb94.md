# Main Branch Audit

## 1. Snapshot

- Branch: `main`
- HEAD: `0efeb94af72ffa067769e19afcd19ad358feefd2`
- Upstream: `origin/main` at the same commit
- Audit reference: `run_84c69db82b28`
- Audit date: 2026-08-23 (Asia/Ho_Chi_Minh)
- Frozen pre-audit dirty state:
  - modified: `.DS_Store`, `src/.DS_Store`, `tests/.DS_Store`
  - untracked: `configs/.DS_Store`, `docs/.DS_Store`,
    `docs/architecture/point_guided_reward_cost_trajectory.html`,
    `scripts/.DS_Store`
- Audit-only additions: `reports/audit/**` and this report
- Production code, tests, configs, plans, and existing documentation modified:
  **none**
- Audit agents:
  - Sol-High controller, architecture reviewer, and technical classifier
  - seven Agy first-pass lanes
  - GPT-5.6-Luna-Max fallbacks for Agy-B, Agy-D, Agy-F, and Agy-G after
    individual quota exhaustion; the first Agy-F fallback startup was also
    unavailable and was restarted with the same preserved scope
  - independent system reviewer: GPT-OSS 120B substitute; DeepSeek was not
    installed/configured and was not impersonated
  - Claude Sonnet 4.6 final evidence/editorial critic

Executed checks included:

- snapshot commands: `git status`, `git branch --show-current`,
  `git rev-parse HEAD`, and `git log --oneline --decorate -20`
- repository task routers for frontend, trajectory, decoder, supervision,
  baseline inference/training, server pipeline, quality, and tests
- full scoped CPU suite: **299 passed, 37 skipped** in 27m37s; skips include
  optional NIfTI-dependent coverage
- focused model-flow suite: **15 passed, 2 skipped**
- focused training/server suite: **18 passed**
- server-pipeline test file: **14 passed**
- targeted fixed-temp-file concurrency reproduction: 8 threads, 40 writes,
  **10 `FileNotFoundError` observations**
- targeted checkpoint-mutation, cross-model-context, mixed-training-mode,
  split-hash, malformed-inventory, dtype, and parameter-count probes
- shell syntax, CLI help, targeted `compileall`, and `git diff --check`

No real BraTS21 cohort, CUDA, NCCL/Gloo multi-process failure injection,
trained checkpoint, held-out evaluation, W&B run, or clinical/reconstruction
quality gate was executed.

## 2. Executive verdict

**RED — do not run or claim the locked Gate-F/G server protocol yet.**

The target-free frontend, bounded Gate-C trajectory, final-Z-only decoder, and
target-after-inference supervision are coherently wired in the audited source.
The 126-d reward revert is clean, legacy reconstruction is not reachable, and
no target-derived value was found in inference.

However, the checked-in server profiles train a 15,107-parameter offset
predictor where the active authority locks 1,419 parameters, and split digests
are accepted without recomputation or immutable checkpoint binding. Those are
research-protocol blockers. Interpreter inconsistency, artifact collisions,
resume gaps, and probable DDP failure handling add operational risk.

The verdict is about software/protocol readiness only. It is not evidence for
or against reconstruction quality, novelty, or clinical validity.

## 3. Current system flow

There is no web frontend, HTTP API, auth system, user role model, or interactive
viewer in this repository path. The product boundary is a CLI/filesystem
PyTorch research pipeline.

1. An operator selects a JSON profile and invokes a Python CLI or server shell
   wrapper.
2. The BraTS adapter inventories subject directories, validates registered
   NIfTI geometry, creates or loads a deterministic train/val/test split, and
   converts `[X,Y,Z]` arrays to `[D,H,W]` tensors.
3. T1/T2/FLAIR become `[B,3,D,H,W]`; the input-only union mask and robust
   normalization are derived without T1ce. T1ce and segmentation remain
   separate typed fields.
4. One shared MedicalNet traversal yields shallow/deep features and three
   coarse semantic classes.
5. Deterministic points are refined from observation features within 2 mm;
   fixed 4-mm semantic PoU support produces static base planes B.
6. Fixed two-level SWT-Haar produces spectral anchor A; affine-aware plane
   queries produce the 168-d point spectral evidence `f_spec`.
7. Gate C initializes Z from B, scores reward/cost with the restored 126-d
   descriptor, enforces exact no-revisit selection, and performs bounded
   update/write-back to final Z.
8. Gate D queries only final Z and performs one chunked `96 -> 64 -> 32 -> 1`
   absolute reconstruction. Generic `forward()` remains fail-closed.
9. Training creates the target-free context/prediction first; only afterward
   Gate E receives T1ce and the auxiliary semantic objective receives labels.
10. The trainer optimizes the authorized module set, logs metrics, and writes
    resume and clean-inference checkpoints. Current server profile widths do
    not match the locked parameter count.
11. Evaluation loads a checkpoint and split, runs eval/no-grad deterministic
    Gate-G inference, then computes metrics and writes NIfTI/JSON artifacts.
    Current split verification and artifact isolation are incomplete.

## 4. Docs vs current implementation

| Area | Docs | Current code | Status | Consequence |
| --- | --- | --- | --- | --- |
| System type | Point-guided research pipeline | CLI/filesystem PyTorch pipeline; no FE/API/auth/viewer | Aligned | Web-product lanes are not applicable. |
| Frontend and target boundary | One T1/T2/FLAIR traversal; no T1ce before objective | Typed target-free path; T1ce enters after prediction | Aligned in software | No target leakage found; real-data behavior unverified. |
| Gate C/D | Bounded route; final-Z-only decode | 126-d reward route and explicit final-Z decoder | Aligned | Temporary 222-d design was fully reverted. |
| Gate F trainable set | Offset predictor is locked at 1,419 parameters | Server profiles select 15,107 parameters | Contradicts docs | A server run would not be the locked baseline. |
| Split provenance | Exact deterministic split reuse | Consuming paths accept any 64-character hash label; checkpoint omits split binding | Broken/incomplete | Cohort changes can evade detection. |
| Inventory | Source cohort should be fully accounted for | Malformed directory names are filtered before exclusion logging | Broken/incomplete | Structural provenance can omit source anomalies. |
| Server execution | Profiles/wrappers form an executable handoff | Wrappers mix `POINT_GUIDED_PYTHON`, bare `python`, and bare `torchrun` | Architecture drift | Documented commands can select incompatible environments. |
| Gate F/G status | Active authority says software ready; experiments pending | F/G software exists; no trained/held-out evidence | Code ahead of older docs | README/quality prose can mislead operators. |
| Evidence | GPU/real-data/trained claims require server execution | CPU synthetic suite passes; server gates absent | Aligned when stated conservatively | No experiment claim is authorized. |

## 5. Confirmed findings

### MAIN-001 — P1 — Server profiles violate the locked offset-predictor contract

- Status: **CONFIRMED**
- Subsystem: model configuration / Gate-F optimizer contract
- Evidence: `AGENTS.md:200-207`, `PLAN.md:1476-1483`, and
  `baseline_training.py:118` lock 1,419 parameters with hidden width 12.
  `configs/training/point_guided_brats21_4070.json:16`,
  `point_guided_brats21_2xa4000.json:16`, and
  `point_guided_brats21_overfit.json:16` set width 128. Direct construction
  measured 15,107 parameters.
- Reproduction: construct the model from any server profile and count
  `point_refiner.offset_predictor.parameters()`.
- Root cause: module ownership is checked, but the locked width/count is not.
- Impact: an F3/F4 run can be mislabeled as the locked baseline while training
  a materially different refinement head.
- Docs relationship: **contradicts active authority**.
- Minimal fix: restore width 12 in accepted profiles or explicitly reopen the
  design decision.
- Regression test needed: build every checked-in profile and assert the locked
  module set and exact counts.

### MAIN-002 — P1 — Split digests are not recomputed and checkpoints are not split-bound

- Status: **CONFIRMED**
- Subsystem: cohort provenance / checkpoint / evaluation
- Evidence: training/evaluation loaders validate only type and length of
  `split_hash`; the worker passed `"a" * 64`. Clean checkpoint metadata omits
  the expected digest.
- Reproduction: provide a complete three-way partition with an arbitrary
  64-character hash; `_load_split()` accepts it.
- Root cause: canonical hashing is used when creating a split, not when
  consuming it, and inference metadata omits cohort identity.
- Impact: modified membership or cross-cohort evaluation can appear bound to
  the original cohort.
- Docs relationship: **contradicts exact-split-reuse claims**.
- Minimal fix: centralize canonical recomputation, reject mismatches, and bind
  the digest into checkpoint metadata.
- Regression test needed: tampered membership, non-hex/mismatched digest, and
  checkpoint/split mismatch must fail closed.

### MAIN-003 — P1 — Server wrappers do not share one interpreter/Torch contract

- Status: **CONFIRMED**
- Subsystem: deployment / launch scripts
- Evidence: the 4070 wrapper honors `POINT_GUIDED_PYTHON`; other wrappers use
  bare `python`, and 2xA4000 uses bare `torchrun`. On the audited host these
  resolve to different Python installs and the bare-Python Torch probe fails.
- Reproduction: compare `command -v python`, the configured venv Python, and
  the `torchrun` shebang, then run each wrapper probe.
- Root cause: per-script executable resolution.
- Impact: preflight/training/evaluation can fail before model execution or run
  DDP under the wrong package/CUDA environment.
- Docs relationship: **partially matches server handoff docs**.
- Minimal fix: one explicit interpreter contract and
  `python -m torch.distributed.run`.
- Regression test needed: shell test with intentionally conflicting PATH
  executables.

### MAIN-004 — P2 — Training/evaluation artifact namespaces are not exclusive

- Status: **CONFIRMED**
- Subsystem: persistence / concurrency
- Evidence: reusable or one-second names use `mkdir(exist_ok=True)`; JSON
  helpers share fixed `.<name>.tmp` files; predictions write directly.
- Reproduction: 8 threads performing 40 writes through `_atomic_json()`
  produced 10 `FileNotFoundError`s in a temporary directory.
- Root cause: no exclusive run reservation and non-unique temporary names.
- Impact: concurrent/repeated jobs can mix or overwrite configs, logs,
  checkpoints, summaries, and predictions.
- Docs relationship: **docs imply isolated reproducible runs; code is incomplete**.
- Minimal fix: exclusive run reservation, unique sibling temp files, atomic
  prediction rename, and explicit reuse policy.
- Regression test needed: concurrent writers and reused output directory.

### MAIN-005 — P2 — Strict checkpoint failure can partially mutate the live model

- Status: **CONFIRMED**
- Subsystem: checkpoint integrity / public inference API
- Evidence: after metadata validation, the loader calls in-place
  `load_state_dict(strict=True)` without tensor-shape/dtype preflight.
- Reproduction: a checkpoint with an early valid changed tensor and a later
  malformed tensor raises while leaving the early tensor changed.
- Root cause: strict PyTorch loading is not transactional.
- Impact: a caller that catches the error can reuse a hybrid model. CLI exit
  behavior limits severity.
- Docs relationship: **contradicts fail-closed checkpoint intent**.
- Minimal fix: preflight all tensors or load into a fresh compatible instance.
- Regression test needed: failed load leaves every model tensor unchanged.

### MAIN-006 — P2 — Gate-E accepts a context from another model instance

- Status: **CONFIRMED**
- Subsystem: supervision / optimizer ownership
- Evidence: `compute_training_objective()` forwards the supplied context; the
  objective uses context-owned trajectory/decoder modules without identity
  comparison to the receiver.
- Reproduction: create context with model A and call model B's objective API;
  gradients follow model A's stored modules.
- Root cause: typed context lacks producing-model identity.
- Impact: a valid-looking loss can update the wrong model.
- Docs relationship: **partially matches Gate-E; violates ownership boundary**.
- Minimal fix: require context module identity to match `self`.
- Regression test needed: cross-instance context is rejected before target use.

### MAIN-007 — P2 — Gate-G does not restore mixed child training modes

- Status: **CONFIRMED**
- Subsystem: inference state management
- Evidence: only `self.training` is saved; recursive `eval()` is followed by
  recursive `train(was_training)`.
- Reproduction: `(parent=True, trajectory=False, decoder=False)` becomes
  `(True, True, True)` after a forced exception; the same restoration runs on
  success.
- Root cause: aggregate mode snapshot instead of module-local state.
- Impact: later route selection can silently use training/straight-through
  semantics rather than hard evaluation semantics.
- Docs relationship: **target-free behavior matches; state restoration is broken**.
- Minimal fix: snapshot/restore every relevant submodule flag.
- Regression test needed: mixed modes preserved on success and exception.

### MAIN-008 — P2 — Resume does not preserve or validate the full training protocol

- Status: **CONFIRMED**
- Subsystem: training resume / provenance
- Evidence: saved settings are returned but not compared; current config is
  written into the existing run; `patience_count` is reset and not checkpointed.
- Reproduction: resume with changed compatible settings and a nearly exhausted
  patience window; the current runtime accepts the drift and restarts patience.
- Root cause: checkpoint schema covers tensors/split label but not immutable
  protocol/progress.
- Impact: a resume can silently change optimization semantics and run longer
  than the original early-stopping contract.
- Docs relationship: **docs ahead of implementation**.
- Minimal fix: define compatible fields, fail closed on drift, and persist
  patience/progress.
- Regression test needed: incompatible config and restored-patience cases.

### MAIN-009 — P2 — Structural inventory omits malformed source directories

- Status: **CONFIRMED**
- Subsystem: data inventory / cohort provenance
- Evidence: directory names are regex-filtered before discovered/excluded sets;
  the active inventory already has an `OTHER_INVALID` classification.
- Reproduction: one valid and one malformed directory yields no record for the
  malformed directory.
- Root cause: validation is performed as a pre-filter rather than a classified
  inventory outcome.
- Impact: cohort accounting can hide typoed or partial source directories.
- Docs relationship: **partially matches; ledger completeness is broken**.
- Minimal fix: enumerate first, then classify malformed names as exclusions.
- Regression test needed: every immediate source directory appears exactly once
  in eligible or excluded output.

### MAIN-010 — P3 — Evaluation metadata always persists `git_head: null`

- Status: **CONFIRMED**
- Subsystem: evaluation provenance
- Evidence: `point_guided_eval.py:230` writes literal `None`; training already
  has a best-effort Git helper.
- Reproduction: inspect any completed `evaluation_metadata.json`.
- Root cause: placeholder field was never wired.
- Impact: artifact-to-code provenance requires console logs or external notes.
- Docs relationship: **partially matches provenance intent**.
- Minimal fix: shared best-effort Git helper.
- Regression test needed: repository and unavailable-Git cases.

### MAIN-011 — P3 — Baseline-training task router declares nonexistent paths

- Status: **CONFIRMED**
- Subsystem: repository navigation / governance
- Evidence: `CODEGRAPH.json` references four superseded files that do not
  exist.
- Reproduction: run the baseline-training task path checks.
- Root cause: task manifest was not updated after server-pipeline owners landed.
- Impact: maintainers are directed to phantom files; runtime is unaffected.
- Docs relationship: **documentation/configuration is outdated**.
- Minimal fix: align paths with current owners.
- Regression test needed: every declared task path exists.

### MAIN-012 — P3 — Decoder point dtype mismatch fails with a raw matmul error

- Status: **CONFIRMED**
- Subsystem: explicit decoder API / tensor contract
- Evidence: float64 physical points with float32 state pass the query, create
  float64 features, then fail in the float32 MLP.
- Reproduction: call explicit point decode with float64 points and float32 Z.
- Root cause: no point/state dtype equality or explicit conversion policy.
- Impact: public callers receive an implementation error instead of a typed
  contract failure.
- Docs relationship: **docs are ambiguous about dtype policy**.
- Minimal fix: enforce a clear dtype rule at the query boundary.
- Regression test needed: mismatch fails early with stable error text.

### MAIN-013 — P3 — A duplicate metrics implementation is orphaned

- Status: **CONFIRMED architectural risk**
- Subsystem: evaluation maintenance
- Evidence: production uses `baseline_metrics.py`; `point_guided_metrics.py`
  has a different SSIM definition and only its own test imports it.
- Reproduction: import/caller trace from training and evaluation entrypoints.
- Root cause: superseded helper was retained.
- Impact: maintainers can choose the wrong metric contract; current production
  results use the intended module.
- Docs relationship: **not documented**.
- Minimal fix: deprecate/remove only if this cleanup is accepted.
- Regression test needed: one authoritative metric contract remains covered.

## 6. Probable / needs reproduction

### MAIN-PROB-001 — P1 — Rank-divergent failure can hang DDP teardown

- Status: **PROBABLE**
- Evidence: an unconditional teardown barrier runs from broad `finally` while
  peers may be in different collectives or rank-0 I/O.
- Missing proof: no two-process failure-injection execution.
- Needed reproduction: two-process Gloo test injecting one-rank failure before
  a collective and asserting prompt coordinated exit.
- Minimal fix direction: barrier only on coordinated success; failure-aware
  abort/teardown.

### MAIN-PROB-002 — P2 — DDP resume collapses rank-local RNG payloads

- Status: **PROBABLE reproducibility risk**
- Evidence: only rank 0 saves RNG; all ranks restore that payload. Current
  sampler/counterfactual code uses explicit generators, limiting proven impact.
- Needed reproduction: two-rank save/resume comparison for Python, NumPy, CPU,
  CUDA, and DataLoader-worker streams.
- Minimal fix direction: rank-specific payloads or a documented deterministic
  post-resume reseeding protocol.

### MAIN-PROB-003 — P2 — Cross-version resume can append under an incompatible CSV header

- Status: **PROBABLE**; artifact subcase of MAIN-008
- Evidence: current row fields determine output order, while any existing CSV
  suppresses header writing without comparison.
- Needed reproduction: resume from a pre-`11ba203` header fixture.
- Minimal fix direction: validate/version/migrate the header before append.

### MAIN-PROB-004 — P3 — Exceptional training exit leaves logger/in-process state incomplete

- Status: **PROBABLE operational risk**, worker severity demoted from P2
- Evidence: W&B finish is normal-path only; partial gradients and persistent
  workers are not explicitly cleared before re-raise.
- Missing proof: supported CLI normally exits; no W&B or long-lived retry test.
- Needed reproduction: injected failure with mock W&B and a retained process.
- Minimal fix direction: explicit abort cleanup in `finally` without masking
  the original exception.

## 7. Documentation mismatches

### MAIN-DOC-001 — P3 — Public Gate-F/G status prose is stale

`README.md`, `docs/architecture/POINT_GUIDED_FRONTEND.md`, and `quality/`
describe Gate F/G as inactive/default-deny. Active authority says F1/F2 and
G1-G4 software are complete while F3/F4 experiments, trained checkpoints, and
held-out evidence remain pending. This is a documentation defect, not an
implementation defect.

## 8. Rejected hypotheses

- No active point-guided import reaches legacy 3DGS reconstruction/training.
- The temporary 222-d reward/candidate-updater design left no stale production
  caller at HEAD; the 126-d revert is coherent.
- No target-derived value was found entering frontend computation, routing,
  stopping, final-Z decoding, or Gate-G inference.
- `DistributedSampler` padding on uneven cohorts is standard DDP behavior and
  a design decision, not an automatic defect. GPT-OSS classified it P2 and
  recommended `drop_last=True`; that recommendation is rejected because it
  discards subjects and lacks governance approval.
- Package-root exports and early placeholder interfaces are not documented
  public-runtime requirements.
- Dice 1.0 for two empty semantic masks is the implemented convention.
- Single-pass evaluation without subject-level resume is a future capability;
  unsafe shared outputs are already MAIN-004.
- Code inspection and passing synthetic tests do not prove mathematical
  soundness, reconstruction quality, novelty, clinical validity, or Gate-H
  readiness.

## 9. Test gaps

- CI omits real-data extras, so NIfTI-dependent tests skip.
- No shell-wrapper environment contract test exists.
- No real dataset preflight or payload load was executed.
- No CUDA AMP, NCCL/Gloo rank-failure, GPU memory, or throughput evidence.
- No concurrent training/evaluation output test exists.
- No transactional failed-checkpoint-load regression exists.
- No cross-model Gate-E context or mixed-mode Gate-G regression exists.
- No tampered split/checkpoint binding or malformed-directory completeness test.
- No DDP resume RNG/protocol, cross-version CSV, W&B abort, disk-full, or quota
  test exists.
- No trained checkpoint or held-out output/metric/NIfTI inspection exists.

## 10. Multi-user / concurrency assessment

There are no authenticated users or server requests. The relevant concurrency
actors are independent CLI jobs and DDP ranks.

- DDP normal-path validation sharding and metric reduction are covered by
  synthetic tests.
- Independent jobs can select the same run/output directory and race on fixed
  temporary names and final artifacts (MAIN-004).
- A rank-divergent exception can reach a different collective sequence and
  probably delay termination until the process-group timeout
  (MAIN-PROB-001).
- Resume state is rank-0-centric and does not prove rank-local stochastic
  reproducibility (MAIN-PROB-002).
- Filesystem permissions, locks, quotas, and concurrent server scheduling are
  otherwise unmodeled.

## 11. Deployment assessment

The repository has CI and server shell wrappers but no container, immutable
environment lock, scheduler manifest, or deployment platform configuration for
this research path.

- The documented wrappers are not portable/reliable across the audited host's
  Python environments (MAIN-003).
- CI pins a CPU stack while project/server instructions allow broad ranges and
  optional real-data/W&B extras; one reproducible server environment is not
  defined.
- Preflight does not substitute for real BraTS, weights, CUDA, DDP, disk, or
  W&B execution.
- Artifact directories are local-filesystem assumptions without exclusive
  reservation, retention policy, or observability/alerting.
- Gate F3/F4 and Gate G held-out evidence remain pending server execution even
  after software remediation.

## 12. Recommended remediation order

Dependencies take precedence over raw count:

1. **MAIN-001** — restore the locked model configuration before training.
2. **MAIN-002** and **MAIN-009** — make inventory, split digest, and checkpoint
   cohort binding trustworthy.
3. **MAIN-003** — make all documented commands use one verified runtime.
4. **Investigate MAIN-PROB-001** — prove/fix coordinated DDP failure handling
   before multi-GPU execution.
5. **MAIN-004** — reserve run namespaces and make persistence collision-safe.
6. **MAIN-005**, **MAIN-008**, **MAIN-PROB-002**, and **MAIN-PROB-003** — make
   checkpoint load/resume transactional and reproducible.
7. **MAIN-006** and **MAIN-007** — close model ownership/state seams.
8. **MAIN-010** through **MAIN-013**, **MAIN-DOC-001**, and the explicit test/
   environment gaps — finish provenance, navigation, hygiene, docs, and
   software evidence.

No production remediation is authorized before the Human Gate.
