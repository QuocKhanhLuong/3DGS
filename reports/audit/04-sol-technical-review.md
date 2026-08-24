# Sol-High Technical Review

## Review basis

- Frozen branch: `main`
- Frozen HEAD: `0efeb94af72ffa067769e19afcd19ad358feefd2`
- Inputs: `01-sol-plan.md`, seven worker reports, and
  `03-deepseek-system-review.md`.
- Independent-auditor disclosure: DeepSeek was unavailable and was not
  impersonated. The Phase-3 role was executed by the available GPT-OSS 120B
  model and is treated as a second opinion, not an authority.
- Mutation boundary: this report only. Production code, tests, configs, plans,
  and existing documentation remained read-only.

## Technical verdict

**RED for server execution under the locked Gate-F/G protocol.** The active
target-free model path is coherently wired and the available CPU suite is
strong software evidence, but the checked-in server profiles do not implement
the locked offset-predictor size, and split provenance is not cryptographically
recomputed or checkpoint-bound. Those two issues can invalidate a future
training/evaluation claim even when the model runs. Launcher, concurrency,
resume, and artifact-isolation defects add operational risk.

This is not a claim that reconstruction quality, novelty, or clinical validity
failed. None of those experimental gates has been executed locally.

## Definitive current flow

This repository is a CLI/filesystem PyTorch research pipeline, not a web
application. Authentication, roles, HTTP APIs, and an interactive viewer do
not exist in the audited path.

| Stage | Docs say | Code currently does | Difference |
| --- | --- | --- | --- |
| Operator entry | Checked-in profiles and server wrappers launch preflight, training, or evaluation. | Python CLIs own the runtime. Shell wrappers choose configs, devices, Python, and `torchrun`; several wrappers use bare executables while one uses `POINT_GUIDED_PYTHON`. | **Architecture drift / operationally broken on the audited host.** |
| Cohort inventory | Every source subject should be accounted for before a deterministic split. | Structural inventory checks matching BraTS directory names and required files, but filters malformed directory names before the exclusion ledger. | **Broken/incomplete provenance.** |
| Split persistence | Train/val/test membership is deterministic and exact split reuse is enforced. | Canonical split creation computes SHA-256, but training/evaluation loaders accept any 64-character `split_hash`; clean inference metadata does not bind the expected split hash. | **Broken integrity contract.** |
| Sample loading | T1/T2/FLAIR form `[B,3,D,H,W]`; geometry is canonical RAS XYZ mm; targets remain separate. | The adapter validates registered NIfTI geometry, converts XYZ arrays to DHW tensors, derives an input-only mask/normalization, and carries T1ce/segmentation in separate fields. | **Aligned in software; real NIfTI execution was unavailable.** |
| Frontend | One shared frozen-by-default MedicalNet traversal yields semantics, deterministic/refined points, sparse PoU, B, A, and 168-d `f_spec`. | `PointGuidedMRIModel._forward_frontend_with_gate_b_context()` implements that typed path with no target argument. | **Aligned.** |
| Gate C | Fixed evidence drives a bounded target-free reward-cost route and final dynamic Z. | The trajectory scores candidates with the restored 126-d RewardNet descriptor, masks utility for exact no-revisit, selects, updates, and writes back within the locked scope. | **Aligned; the temporary 222-d experiment was cleanly reverted.** |
| Gate D | Only final Z reaches the geometry-aware implicit decoder. | Explicit reconstruction/inference APIs query final `DynamicTriPlanes` and decode in chunks; generic `forward()` raises. | **Aligned.** |
| Gate E | T1ce enters only after target-free context/prediction; segmentation is training-only auxiliary supervision. | The trainer obtains a target-free context first, then calls the Gate-E objective and semantic auxiliary loss. A public API can accept a valid context from another model instance. | **Core boundary aligned; ownership guard incomplete.** |
| Gate F training | Only the locked eight-module trainable set is optimized, including the existing 1,419-parameter offset predictor. | Module ownership is enforced, but all server profiles set `offset_hidden_channels=128`, producing 15,107 offset-predictor parameters instead of 1,419. | **Contradicts locked protocol.** |
| Resume/DDP | Checkpoints, RNG, early stopping, DDP teardown, and provenance should resume coherently. | Rank 0 writes checkpoints; all ranks restore the one RNG payload; patience is not restored; current config can replace prior run metadata; teardown always barriers. | **Incomplete and partly probable until multi-process reproduction.** |
| Gate G evaluation | A validated checkpoint and exact held-out split drive target-free inference; metrics and predictions are computed afterward. | The evaluator loads the model, forces eval/no-grad, performs deterministic routing and one final-Z decode, then computes metrics and writes NIfTI/JSON artifacts. | **Core inference aligned; checkpoint transactionality, split binding, provenance, and output isolation are defective.** |
| Evidence status | F3/F4 and trained-checkpoint/held-out evidence require server execution. | CPU synthetic checks exist; real BraTS, GPU, DDP, trained checkpoint, and held-out evaluation were not run. | **Aligned when reported conservatively; older docs are stale.** |

## Consolidated finding classification

### Confirmed

#### MAIN-001 — P1 — Server profiles violate the locked offset-predictor contract

- Sources: `AGY-D-FIND-007`.
- Evidence: `AGENTS.md:200-207`, `PLAN.md:1476-1483`,
  `baseline_training.py:118`, and the focused training test lock 1,419
  parameters with hidden width 12; all three checked-in server profiles set
  `offset_hidden_channels` to 128 at
  `configs/training/point_guided_brats21_4070.json:16`,
  `configs/training/point_guided_brats21_2xa4000.json:16`, and
  `configs/training/point_guided_brats21_overfit.json:16`. Direct construction
  measured 15,107 parameters in the predictor.
- Root cause: configuration width remained adjustable after governance locked
  the concrete trainable count; ownership validation checks module membership,
  not the count.
- Impact: an F3/F4 run can be labeled as the locked baseline while training a
  materially different refinement head.
- Minimal fix: set the server profiles to the locked width or request an
  explicit design change. Add a profile-level parameter-count test.

#### MAIN-002 — P1 — Split digests are not recomputed and clean checkpoints are not split-bound

- Sources: `AGY-G-001`.
- Evidence: training and evaluation loaders check only that `split_hash` is a
  64-character string; the worker reproduced acceptance of `"a" * 64` for a
  complete partition. `baseline_checkpoint_metadata()` omits the expected
  split digest.
- Root cause: the canonical hashing implementation is used when creating a
  split but not when consuming it, and the inference checkpoint contract
  omits cohort identity.
- Impact: altered membership can retain a plausible hash label, and a
  checkpoint can be evaluated against a different cohort without detection.
- Minimal fix: centralize canonical hash recomputation, reject mismatches, and
  bind the digest into immutable checkpoint metadata with negative tests.

#### MAIN-003 — P1 — Server wrappers do not share one interpreter/Torch contract

- Sources: `AGY-F-FIND-002`.
- Evidence: the 4070 launcher honors `POINT_GUIDED_PYTHON`; preflight,
  overfit, evaluation, and 2xA4000 wrappers use bare `python`, and 2xA4000 uses
  bare `torchrun`. On the audited host these resolve to three different Python
  installations, and the bare-Python Torch probe fails.
- Root cause: each wrapper resolves tools independently.
- Impact: documented workflows can fail before data/model execution or launch
  DDP under a different environment.
- Minimal fix: one explicit interpreter variable, `python -m torch.distributed.run`,
  and a wrapper contract test.

#### MAIN-004 — P2 — Training and evaluation artifact namespaces are not exclusive

- Sources: `AGY-D-FIND-002`, `AGY-F-FIND-003` (collapsed root cause).
- Evidence: reusable or one-second run names use `mkdir(exist_ok=True)`;
  JSON helpers share fixed `.<name>.tmp` paths; an 8-thread/40-write temporary
  reproduction produced 10 `FileNotFoundError`s. Checkpoints and summaries are
  last-writer-wins, and NIfTI predictions are written directly.
- Impact: concurrent/repeated jobs can mix configs, metrics, checkpoints,
  summaries, and predictions.
- Minimal fix: exclusive run reservation, collision-safe unique temporary
  files, atomic prediction rename, and concurrent/reused-directory tests.

#### MAIN-005 — P2 — A failed strict checkpoint load can partially mutate the live model

- Source: `AGY-B-FIND-001` (worker severity demoted from P1).
- Evidence: metadata checks precede a direct in-place
  `model.load_state_dict(..., strict=True)`. The worker supplied one valid
  changed tensor followed by a malformed tensor; load raised while the valid
  tensor remained changed.
- Impact: a caller that catches the error can reuse a hybrid model. The CLI
  normally exits, which limits but does not remove the public-API risk.
- Minimal fix: preflight exact keys/shapes/dtypes before copying or load into a
  separate compatible instance; add a no-mutation-on-failure test.

#### MAIN-006 — P2 — Gate-E accepts a supervision context from another model instance

- Source: `AGY-B-FIND-002`.
- Evidence: the receiving model checks only that it owns a trajectory and
  decoder; the objective takes `context._trajectory` and `context._decoder`.
  The typed context has no producing-model identity.
- Impact: model B's public objective call can backpropagate through model A.
- Minimal fix: require identity equality for context-owned modules and add a
  two-model negative test.

#### MAIN-007 — P2 — Gate-G does not restore mixed child training modes

- Source: `AGY-B-FIND-003`.
- Evidence: the wrapper snapshots only `self.training`, recursively calls
  `eval()`, then recursively calls `self.train(was_training)`. A synthetic
  exception-path probe changed `(parent=True, trajectory=False,
  decoder=False)` to `(True, True, True)`.
- Impact: a later call can silently use training/straight-through route
  behavior where the caller had selected hard evaluation behavior.
- Minimal fix: snapshot and restore module-local flags; test success and
  exception paths.

#### MAIN-008 — P2 — Resume does not preserve or validate the full training protocol

- Sources: `AGY-D-FIND-004`; `AGY-G-003` is a probable artifact subcase (see
  `MAIN-PROB-003`).
- Evidence: the checkpoint returns saved settings but the runtime does not
  compare them; current `config.json` is written into the existing run;
  `patience_count` is reset and absent from checkpoint state.
- Impact: a resume can silently change optimization settings and extend early
  stopping.
- Minimal fix: define compatible/immutable fields, fail closed on drift, and
  checkpoint patience/progress. Validate CSV schema before appending.

#### MAIN-009 — P2 — Structural inventory omits malformed source directories

- Source: `AGY-G-002`.
- Evidence: directory names are regex-filtered before `discovered` and
  `excluded` are built; a temporary fixture containing one valid and one
  malformed directory omitted the malformed entry entirely. The later active
  inventory already has an `OTHER_INVALID` classification.
- Impact: cohort provenance can claim complete accounting while hiding a
  typoed/partial directory.
- Minimal fix: enumerate all immediate directories first and record malformed
  names as exclusions; add a ledger-completeness test.

#### MAIN-010 — P3 — Evaluation metadata always persists `git_head: null`

- Source: `AGY-E-FIND-001`.
- Evidence: `point_guided_eval.py:230` writes the literal `None`, while the
  training runtime already implements `_git_head()`.
- Impact: evaluation artifacts lose a direct code provenance link.
- Minimal fix: use a shared best-effort Git provenance helper and test both
  repository and unavailable-Git states.

#### MAIN-011 — P3 — The baseline-training task router declares nonexistent paths

- Source: `AGY-A-FIND-001` (worker severity demoted from P2).
- Evidence: `CODEGRAPH.json` lists four drafting paths that do not exist and
  were superseded by current data, CLI, config, and test owners.
- Impact: task-scoped navigation/check commands lead maintainers to phantom
  files; runtime behavior is unaffected.
- Minimal fix: align the router with current owners and add a task-path
  existence test.

#### MAIN-012 — P3 — Decoder point dtype mismatch fails late with a raw matmul error

- Source: `AGY-B-FIND-004`.
- Evidence: float64 physical points with float32 state are accepted by the
  query, yield float64 sampled features, and fail in the float32 decoder MLP.
- Impact: public explicit point queries fail with an implementation error
  instead of a typed contract message.
- Minimal fix: enforce one explicit dtype policy at the query boundary.

#### MAIN-013 — P3 — A duplicate metrics implementation is orphaned

- Sources: `AGY-A-FIND-002`, `AGY-E-FIND-002` (duplicate).
- Evidence: production training/evaluation imports `baseline_metrics.py`;
  `point_guided_metrics.py` has a different SSIM definition and is referenced
  only by its own test.
- Impact: ambiguous maintenance surface, not an active metric bug.
- Minimal fix: deprecate/remove the orphan under a separately accepted cleanup.

### Probable or needs reproduction

#### MAIN-PROB-001 — P1 — Rank-divergent failure can hang DDP teardown

- Source: `AGY-D-FIND-001`.
- Evidence: `destroy_distributed()` unconditionally enters a barrier from a
  broad `finally`; peer ranks can still be in a different all-reduce,
  broadcast, backward collective, or rank-0 artifact write.
- Status: **PROBABLE**, not confirmed, because no two-process failure-injection
  test was run. The consequence is bounded by the process-group timeout rather
  than proven indefinite deadlock.
- Needed reproduction: two-process Gloo test injecting one-rank failure before
  a collective and asserting prompt coordinated termination.

#### MAIN-PROB-002 — P2 — DDP resume collapses rank-local RNG payloads

- Source: `AGY-D-FIND-003`.
- Evidence: only rank 0 saves RNG and every rank restores it. Current sampler
  and counterfactual paths use explicit deterministic generators, so a
  user-visible divergence/correlation was not reproduced.
- Status: **PROBABLE reproducibility risk**.
- Needed reproduction: two-rank save/resume test comparing per-rank Python,
  NumPy, CPU, CUDA, and DataLoader-worker streams.

#### MAIN-PROB-003 — P2 — Cross-version resume can append rows under an incompatible CSV header

- Source: `AGY-G-003`.
- Evidence: field names come from the current row, while an existing file
  suppresses header writing without validation.
- Status: **PROBABLE**; no old-schema fixture was executed.
- Needed reproduction: resume from a fixture with the pre-`11ba203` header and
  assert rejection rather than column-shifted append.

#### MAIN-PROB-004 — P3 — Exceptional training exit leaves logger/in-process state incomplete

- Source: `AGY-D-FIND-006` (worker severity demoted from P2).
- Evidence: `wandb_run.finish()` is normal-path only, and partial gradients or
  persistent workers are not explicitly cleared before re-raise.
- Status: **PROBABLE operational risk**. The supported CLI process normally
  exits and the OS/SDK may clean resources; no W&B or long-lived retry test was
  run.

## Documentation mismatches

### MAIN-DOC-001 — P3 — Public Gate-F/G status prose is stale

`README.md`, `docs/architecture/POINT_GUIDED_FRONTEND.md`, and `quality/`
still describe Gate F/G as inactive/default-deny. Active authority states F1/F2
and G1-G4 software are complete while server experiments and held-out evidence
remain pending. This is documentation drift, not proof of a code defect.

## Operational and test gaps

- **MAIN-OPS-001 — P2:** CI does not install the real-data extras and therefore
  skips NIfTI-dependent data coverage; it does not execute shell launchers,
  real dataset preflight, CUDA AMP, or multi-process DDP.
- **MAIN-OPS-002 — P2:** CI pins and broad project/server dependency ranges do
  not define one reproducible server environment. A server lock or recorded
  environment manifest is absent.
- Real BraTS21, MedicalNet checkpoint loading, GPU memory/throughput, NCCL,
  W&B, trained-checkpoint inference, held-out metrics, and NIfTI artifact
  inspection remain unverified. These are missing evidence, not failed tests.

## Design decisions and rejected hypotheses

- **DESIGN_DECISION_REQUIRED:** Training `DistributedSampler` padding on an
  uneven cohort is standard DDP behavior that keeps collective counts equal.
  It changes sample weighting and should be documented/recorded, but
  `drop_last=True` would discard data and is not automatically a correct fix.
  GPT-OSS classified this as P2 (`SYS-DEF-011`) and recommended
  `drop_last=True`; that recommendation is rejected because `drop_last`
  discards subjects and is not a governance-approved change.
- **DESIGN_DECISION_REQUIRED:** Exporting every Gate A-G type from package
  `__init__.py` is not a documented public-API requirement. Deep submodule
  imports are currently intentional enough that the incomplete-export claim is
  rejected as a defect.
- **DESIGN_DECISION_REQUIRED:** Early placeholder interfaces and legacy-only
  package-root exports are dead/coexistence hygiene, not active wiring bugs.
- **FALSE_POSITIVE:** No active point-guided import reaches the legacy 3DGS
  reconstruction/training path.
- **FALSE_POSITIVE:** The temporary 222-d/candidate-updater reward contract did
  not leave a stale production caller at HEAD; the 126-d revert is coherent.
- **FALSE_POSITIVE:** No target-derived value was found entering the frontend,
  route, stopping, final-Z decoder, or Gate-G inference.
- **NOT A DEFECT:** Dice 1.0 for two empty semantic masks is the implemented
  documented evaluation convention.
- **NOT A DEFECT:** Single-pass evaluation without subject-level resume is a
  possible future capability, not a present contract failure. Its unsafe
  shared output directory is already covered by MAIN-004.
- **UNSUPPORTED CLAIM REJECTED:** Passing software inspection does not prove
  the architecture is mathematically sound, reconstructs T1ce, is clinically
  valid, or is ready for Gate-H.

## Recommended remediation order

1. **MAIN-001** — restore the locked model configuration before any training.
2. **MAIN-002** and **MAIN-009** — make cohort inventory, split digest, and
   checkpoint binding trustworthy before producing checkpoints.
3. **MAIN-003** — make every documented launcher use the same verified runtime.
4. **MAIN-PROB-001** — reproduce and repair coordinated DDP failure handling
   before multi-GPU execution.
5. **MAIN-004** — reserve run namespaces and make persistence collision-safe.
6. **MAIN-005**, **MAIN-008**, **MAIN-PROB-002**, and **MAIN-PROB-003** — make
   checkpoint load/resume transactional and reproducible.
7. **MAIN-006** and **MAIN-007** — close public model state/ownership seams.
8. **MAIN-010** through **MAIN-013**, **MAIN-DOC-001**, and operational test
   gaps — repair provenance, navigation, hygiene, docs, and evidence coverage.

No remediation is authorized before the Human Gate.
