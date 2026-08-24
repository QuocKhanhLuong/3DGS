# Main Branch Remediation

Audit target: `0efeb94af72ffa067769e19afcd19ad358feefd2`  
Working branch: `remediation/main-0efeb94`  
Resulting HEAD: `0efeb94af72ffa067769e19afcd19ad358feefd2`  
Commits/push: none.

## Accepted findings

| ID | Result | Remediation |
| --- | --- | --- |
| MAIN-001 | FIXED | Restored locked offset hidden width 12 and exact 1,419 parameters across all profiles. |
| MAIN-002 | FIXED | Central canonical split digest, membership recomputation, checkpoint binding, and evaluation compatibility. |
| MAIN-003 | FIXED | One explicit `POINT_GUIDED_PYTHON` contract for all launchers and distributed execution. |
| MAIN-004 | FIXED | Exclusive run reservation, writer lock, unique temporaries, atomic persistence, explicit reuse policy. |
| MAIN-005 | FIXED | Complete key/shape/dtype preflight before strict live-model mutation. |
| MAIN-006 | FIXED | Gate-E context is bound to the producing model. |
| MAIN-007 | FIXED | Module-local train/eval flags restore on success and exception. |
| MAIN-008 | FIXED | Versioned resume protocol, immutable/compatible fields, run progress, rank RNG, historical config, CSV fail-closed behavior. |
| MAIN-009 | FIXED | Complete once-only immediate-source ledger including `OTHER_INVALID`. |
| MAIN-010 | FIXED | Shared best-effort Git HEAD provenance with explicit null when unavailable. |
| MAIN-011 | FIXED | Current task routing plus declared-path validation. |
| MAIN-012 | FIXED | Explicit ordinary/AMP point-state dtype contract without coordinate narrowing. |
| MAIN-DOC-001 | FIXED | Evidence-bound public Gate-F/G status. |

MAIN-013 was deferred and remains untouched.

## Investigated probable findings

| ID | Result | Action |
| --- | --- | --- |
| MAIN-PROB-001 | NOT_REPRODUCED | A bounded manual two-rank Gloo run exited through elastic termination in 3.398 s; no committed script/log was retained and no production change was made. |
| MAIN-PROB-002 | CONFIRMED / FIXED | Resume checkpoints now save rank-indexed RNG payloads; Python, NumPy, CPU Torch, and derived worker streams were verified. CUDA handling is implemented but unverified locally. |
| MAIN-PROB-003 | CONFIRMED / FIXED | Existing CSV headers must match exactly before any JSONL/CSV append. |
| MAIN-PROB-004 | CONFIRMED / FIXED | W&B finalizes exactly once and exceptional partial gradients are cleared without masking the original failure. |

## Performance decisions

| ID | Decision |
| --- | --- |
| PERF-001 | ADOPT |
| PERF-002 | REJECT |
| PERF-003 | NEEDS_MORE_EVIDENCE |
| PERF-004 | NEEDS_MORE_EVIDENCE |
| PERF-005 | NEEDS_MORE_EVIDENCE |
| PERF-006 | NEEDS_MORE_EVIDENCE |
| PERF-007 | NEEDS_MORE_EVIDENCE |
| PERF-008 | NEEDS_MORE_EVIDENCE |
| PERF-009 | NOT_AUTHORIZED |
| PERF-010 | NEEDS_MORE_EVIDENCE |
| PERF-011 | HUMAN_GATE |
| PERF-012 | HUMAN_GATE |
| PERF-013 | HUMAN_GATE |
| PERF-014 | HUMAN_GATE |

PERF-001 removed one provably redundant brain-mask copy with exact contract
tests. PERF-002 retained the remaining public trust-boundary validation.
No other performance production change was adopted.

## Worker ownership and fallback events

- Research identity: independent profile, split-provenance, and inventory
  workers, followed by Sol's tampering review.
- Execution/data: launcher/DDP and PERF-001/002 workers.
- Persistence: artifact, checkpoint, resume/RNG/CSV, and cleanup workers.
- Model invariants: context/mode and dtype workers.
- Governance: Git provenance, task routing, and documentation workers.
- Agy was unavailable because of provider quota/sign-in/runtime failures;
  assignments were continued with `gpt-5.6-luna` at max effort as the required
  Luna-Max fallback. One Orca readiness failure was retried, then continued in
  the same shared worktree without discarding evidence or partial work.
- DeepSeek was unavailable; the Codex `reviewer` role (`gpt-5.6-sol`, high)
  was substituted and disclosed. It found and then cleared the MAIN-012 AMP
  blocker.
- Claude completed the final report proofread and its ten evidence-hygiene
  corrections were incorporated.

## Tests and checks

- Final full suite: `678 passed, 42 skipped, 26 subtests passed` in 71.36 s.
- Focused areas covered split tampering, fake/malformed hashes, ledger
  completeness, conflicting PATH launchers, concurrent writers, transactional
  load failure, resume protocol/RNG/CSV, cleanup failures, cross-model context,
  mixed module modes, AMP dtype pairs, Git absence, and task paths.
- All point-guided shell launchers passed `bash -n`.
- All declared CodeGraph task paths validated.
- `compileall` and `git diff --check` passed after final edits.
- CUDA/server-dependent tests remained skipped; they are not claimed as proof.

## Server performance

No real-server measurements were possible. This host is an Apple M1 with no
CUDA/NVIDIA tooling, no NiBabel, no real NIfTI volumes, no remote server, and
no OpenCode Ox Alpha model. OpenCode exposed only Semble MCP. Consequently:

- samples/sec: **NOT MEASURED**
- DataLoader wait: **NOT MEASURED**
- GPU idle/utilization: **NOT MEASURED**
- Gate-E CUDA percentage: **NOT MEASURED**
- dominant server CPU stage: **NOT MEASURED**
- dominant GPU stage: **NOT MEASURED**

The earlier CPU-synthetic Ox report remains investigation evidence only.

## Files changed

Remediation touches the point-guided profiles, launchers, server documentation,
task routing, BraTS data/provenance, artifact/checkpoint/resume/training,
model/query boundaries, and their focused tests. New shared modules are
`artifacts.py` and `provenance.py`. Reports are under `reports/remediation/`.

Pre-existing `.DS_Store` changes and the untracked architecture HTML were
preserved and not overwritten. No commit or push was performed.

## Remaining gates

- Gate F3/F4 experiment execution.
- Real trained checkpoints and held-out Gate G evaluation evidence.
- Real CUDA/server performance baseline and isolated experiments.
- A second Human Gate for PERF-011/012/013/014.

## Final readiness verdict

**Correctness remediation is ready for human review. Experimental and server
readiness remains pending and is not claimed.**
