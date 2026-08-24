# Main Remediation Plan

## 1. Safety snapshot

- Audited base: `main` at `0efeb94af72ffa067769e19afcd19ad358feefd2`.
- Remediation branch: `remediation/main-0efeb94`, created at the identical SHA.
- Push policy: no push.
- Pre-existing modified files preserved: `.DS_Store`, `src/.DS_Store`, `tests/.DS_Store`.
- Pre-existing untracked files preserved: `configs/.DS_Store`, `docs/.DS_Store`,
  `docs/architecture/point_guided_reward_cost_trajectory.html`, `scripts/.DS_Store`,
  and the audit reports under `reports/`.
- The point-guided target-free, final-Z-only, deterministic geometry, and Gate-E
  target-after-inference contracts remain locked.

## 2. Accepted scope

Correctness fixes: `MAIN-001` through `MAIN-012`, plus `MAIN-DOC-001`.
`MAIN-013` is separately deferred.

Investigate before fixing: `MAIN-PROB-001` through `MAIN-PROB-004`.

Approved low-risk performance fixes: `PERF-001`, `PERF-002`.

Approved experiments only: `PERF-003`, `PERF-004`, `PERF-005`, `PERF-006`,
`PERF-007`, `PERF-008`, `PERF-010`. Production adoption requires real-server
measurement and equivalence evidence.

Second-Human-Gate items, excluded from implementation: `PERF-011` through
`PERF-014`. `PERF-009` is not in the accepted decision set and is untouched.

## 3. Dependency DAG

```text
A1 MAIN-001 ─────┐
A2 MAIN-002 ─────┼─> Sol Gate A
A3 MAIN-009 ─────┘
                     |
                     v
B1 MAIN-003 ─────┐
B2 PROB-001 repro┘
                     |
                     v
C1 PERF-001 + PERF-002
                     |
                     v
D1 MAIN-004 ─────┐
D2 MAIN-005 ─────┼─> D3 MAIN-008
D4 PROB-002/003/004 investigations ┘
                     |
                     v
E1 MAIN-006 ─────┐
E2 MAIN-007 ─────┼─> Sol invariant review
E3 MAIN-012 ─────┘
                     |
                     v
F1 MAIN-010 ─────┐
F2 MAIN-011 ─────┼─> F4 MAIN-DOC-001 review
F3 MAIN-DOC-001 ─┘
                     |
                     v
Full correctness suite and combined-diff review
                     |
                     v
Real-server baseline -> isolated PERF experiments -> Sol decisions
                     |
                     v
System review -> Claude evidence pass -> final reports
```

Workers may run in parallel only when the file sets below do not overlap. Later
workers must preserve earlier diffs and re-read current files before editing.

## 4. Worker assignments

### A1 — locked research architecture (`MAIN-001`)

- Owns the three checked-in training JSON profiles and a dedicated profile
  regression test.
- Restores `offset_hidden_channels=12`; asserts the eight-module optimizer
  ownership and exactly 1,419 offset-predictor parameters for every profile.
- Must not change `OffsetPredictor` or the locked architecture.

### A2 — split/cohort binding (`MAIN-002`)

- Owns canonical split hashing, training/evaluation split consumers, clean
  checkpoint split metadata, and focused provenance tests.
- Recomputes a canonical digest from actual membership and split-defining
  fields, validates lowercase hexadecimal SHA-256, binds the digest into clean
  checkpoint metadata, and rejects checkpoint/split mismatch.
- Must not migrate invalid historical checkpoints silently.

### A3 — complete source ledger (`MAIN-009`)

- Runs only after A2 releases shared data-adapter ownership.
- Owns structural inventory classification and ledger-completeness tests.
- Every immediate source directory appears exactly once as eligible or
  excluded; malformed names become `OTHER_INVALID` without opening payloads.

### B1 — launcher interpreter contract (`MAIN-003`)

- Owns `scripts/point_guided_*.sh`, launcher documentation directly tied to
  invocation, and shell contract tests.
- All Python and DDP launch paths resolve through `POINT_GUIDED_PYTHON`, using
  `-m torch.distributed.run` rather than ambient `torchrun`.

### B2 — DDP failure investigation (`MAIN-PROB-001`)

- First creates a minimal two-process Gloo failure-injection test.
- `CONFIRMED`: implement the smallest failure-aware teardown and regression.
- `NOT_REPRODUCED`: retain current synchronization and record evidence.
- `ENVIRONMENT_BLOCKED`: record the missing capability exactly.

### C1 — safe data-path cleanup (`PERF-001`, `PERF-002`)

- Removes only the extra mask `.copy()` already preceded by
  `np.ascontiguousarray`.
- Removes/fuses only validations proven redundant across an immutable boundary.
- Documents `previous validation -> immutable transform -> redundant scan` and
  retains adversarial rejection coverage.

### D1 — collision-safe artifacts (`MAIN-004`)

- Owns run reservation, atomic write helpers, prediction output atomicity, and
  concurrency tests.
- Requires exclusive run creation, unique sibling temporaries, explicit reuse
  policy, and no cross-run mixing.

### D2 — transactional clean checkpoint loading (`MAIN-005`)

- Owns clean inference checkpoint preflight/load and focused tests.
- A failed strict load leaves every live model tensor unchanged.

### D3 — explicit resume protocol (`MAIN-008`)

- Runs after D1 because both touch training persistence.
- Defines immutable and compatible fields; validates saved vs requested
  protocol; persists/restores patience and required run state.

### D4 — probable resume/cleanup investigations (`MAIN-PROB-002/003/004`)

- Reproduces rank-local RNG resume, old-schema CSV append, and exceptional
  logger/process cleanup independently before any fix.
- A confirmed subcase receives a narrow follow-up implementation task.

### E1/E2/E3 — model/state contracts (`MAIN-006/007/012`)

- E1 binds Gate-E contexts to the producing model and rejects cross-instance
  use before target work.
- E2 snapshots/restores module-local train/eval flags on success and exception,
  including mixed parent/child states.
- E3 enforces an explicit ordinary/AMP point-state dtype contract with a
  stable typed error. Physical coordinates are never narrowed; low-precision
  latent state is promoted explicitly only at the sampling boundary.

### F1/F2/F3 — provenance and governance (`MAIN-010/011/DOC-001`)

- F1 shares best-effort Git HEAD resolution between training and evaluation.
- F2 repairs `CODEGRAPH.json` task paths and adds an existence validator.
- F3 updates public F/G status without claiming F3/F4 execution, trained
  checkpoints, held-out evidence, GPU evidence, or clinical/reconstruction
  validity.

## 5. Review gates

### Sol Gate A

Requires exact 1,419-parameter profile construction, canonical split digest
tamper failures, checkpoint/split mismatch failure, and complete source ledger.

### Correctness integration gate

Requires focused tests, all point-guided CPU tests available locally,
launcher syntax/contracts, concurrency, resume, provenance, targeted
`compileall`, and `git diff --check`. Skips and environment blocks are reported,
not converted into passes.

### Real-server performance gate

Begins only after the combined correctness diff passes. It requires actual
BraTS-style volumes, NiBabel, CUDA, target storage, and the target
Python/PyTorch environment. If those are unavailable, phases 9–10 are
`ENVIRONMENT_BLOCKED`; no CPU proxy is substituted.

### Performance adoption gate

An approved experiment becomes `ADOPT` only with a measured bottleneck,
measured improvement, correctness/numerical equivalence, and acceptable
resource use. `PERF-011`–`PERF-014` remain `HUMAN_GATE` regardless of results.

## 6. Fallback and mutation rules

- Each Agy worker owns only its declared files and tests and must not revert
  parallel changes.
- On Agy quota/unavailability, the same task continues with
  `gpt-5.6-luna-max`, carrying the exact diff, evidence, tests, and remaining
  work.
- No unrelated refactor, no economic-stop change, no target-boundary change,
  no architecture change, no push.

## 7. Orchestration record

- Orca run: `run_43f8ba01b84e`.
- `MAIN-001`: `task_2fec6dcb0d3d`; Agy completed, then Sol independently
  reran the exact profile regression.
- `MAIN-002`: `task_a50b8769509a`; Agy completed, then Sol independently
  reran focused provenance, evaluation, checkpoint, and data tests.
- `MAIN-009`: `task_d5250c034919`. Initial Agy dispatch
  `ctx_e03e265a63b3` remained unavailable at the login splash and made no
  repository changes. The same task continued as required with
  `gpt-5.6-luna` at `max` reasoning (`ctx_ceb1b67d2817`), equivalent to the
  requested Luna-Max fallback. Sol found an all-malformed-source boundary gap
  in that first result and continued the same objective in follow-up task
  `task_04fa9f7538f9` / dispatch `ctx_eb57fde398d1`. The follow-up preserved a
  complete all-excluded ledger and moved the no-eligible failure to the
  training/preflight consumer.
- Sol Gate A passed independently: `41 passed, 19 skipped`; targeted
  `compileall` and `git diff --check` also passed.
- `MAIN-003`: both fresh Agy terminals were unavailable (Antigravity sign-in
  and an interactive shell-update interception). The Orca Luna-Max supervised
  start also failed agent readiness, so the unchanged task continued through a
  direct `gpt-5.6-luna` / `max` worker. Sol independently passed the dedicated
  conflicting-`PATH` contract test (`1 passed`), `bash -n`, and
  `git diff --check`.
- `MAIN-PROB-001`: the first Luna-Max continuation could not establish the
  prerequisite clean two-rank local Gloo baseline and stopped producing useful
  output. Sol fenced it before any production teardown change and continued the
  same task with a fresh Luna-Max worker, preserving the environment evidence
  and partial test for correction/removal. A later rank-local RNG investigation
  established a healthy project-runtime `torch.distributed.run` baseline, so
  Sol reopened only the invalidated environment premise. The coordinated
  recheck classified `MAIN-PROB-001` as `NOT_REPRODUCED`: in a bounded manual
  run, the supported elastic launcher terminated the surviving rank in the
  current teardown barrier and exited failed in 3.398 seconds under a 10-second
  bound. No committed reproduction script or durable log was retained;
  production teardown remained unchanged.
- `PERF-001`: adopted after Sol independently passed content/dtype/shape/
  stride/contiguity/source-independence equivalence and the focused suite.
- `PERF-002`: no production change. Sol rejected removal of the batch label
  scan because exported `PointGuidedBatch` can be constructed directly and is
  therefore a trust boundary, not merely an immutable collate result.
- Persistence and model tasks: `task_4ca84cc46e97` (MAIN-004),
  `task_44c24df68a06` (MAIN-005), `task_4571c395e79a` (MAIN-008),
  `task_dff3f44eafe5` (confirmed MAIN-PROB-004), `task_210fbc6ea0f3`
  (MAIN-006/007), and `task_46678e60f741` (MAIN-012). Sol's independent system
  review found an AMP interaction in the first MAIN-012 result; the same scope
  continued with Luna-Max and passed independent re-review after correction.
- Governance tasks: `task_2a387ec3df8d` (MAIN-010),
  `task_7cb80ca5925f` (MAIN-011), and `task_bd6d964f9ab6`
  (MAIN-DOC-001), all completed before the final full suite.
- The system review used the Codex `reviewer` role (`gpt-5.6-sol`, high) because
  DeepSeek was unavailable. Claude proofreading ran as Orca task
  `task_0d5a863dd465` / dispatch `ctx_ea561553f97d` and returned PASS with
  evidence-hygiene corrections incorporated into the final reports.
- The actual-server/Ox performance phase was environment-blocked as documented
  in reports 03/04; no downstream performance experiment is described as
  completed merely because a task was registered.
