# Correctness Remediation Results

Audit base: `0efeb94af72ffa067769e19afcd19ad358feefd2`  
Working branch: `remediation/main-0efeb94`  
State: correctness remediation complete; no commit or push.

## Closed review gates

| ID | Result | Evidence |
| --- | --- | --- |
| MAIN-001 | FIXED | All checked-in server profiles use locked width 12; exact offset predictor count 1,419. |
| MAIN-002 | FIXED | Canonical membership-bound split digest is recomputed; split/checkpoint mismatch fails closed. |
| MAIN-009 | FIXED | Every immediate source directory is ledgered exactly once; malformed-only sources return an all-excluded ledger and training fails at the consumer boundary. |
| MAIN-003 | FIXED | Five supported launchers use `POINT_GUIDED_PYTHON`; DDP uses `-m torch.distributed.run`; conflicting `PATH` test passes. |
| MAIN-PROB-001 | NOT_REPRODUCED | In a bounded manual run, both ranks completed a pre-failure collective; after rank 0 failed, the elastic launcher terminated rank 1 in the teardown barrier and exited failed in 3.398 s below the 10 s bound. No committed reproduction script/log was retained; no production change. |
| PERF-001 | ADOPTED | Removed one redundant full-mask copy after an independent contiguous conversion; exact tensor-contract equivalence is tested. |
| PERF-002 | NO_CHANGE | The candidate label scan remains required for direct construction of exported `PointGuidedBatch`; removing it would weaken validation. |
| MAIN-004 | FIXED | Exclusive run/output reservation, held writer locks, unique sibling temporaries, atomic JSON/Torch/prediction replacement, and explicit reuse policy. |
| MAIN-005 | FIXED | Exact key/shape/dtype preflight occurs before strict live-model load; failed loads leave every tensor unchanged. |
| MAIN-PROB-003 | CONFIRMED | Old-schema `metrics.csv` silently accepted new rows under obsolete column names; fail-closed header validation is assigned to MAIN-008. |
| MAIN-PROB-002 | CONFIRMED | Rank 0's single RNG payload is restored on every rank; rank 1 diverges across Python, NumPy, Torch, and DataLoader worker streams. Rank-indexed restore is assigned to MAIN-008. |
| MAIN-PROB-004 | FIXED | W&B finalizes exactly once on success/failure; exceptional accumulation clears partial gradients while preserving the original error. |
| MAIN-008 | FIXED | Versioned immutable/compatible resume protocol; historical configuration is preserved; early-stopping/run progress and rank-indexed RNG state are restored; old CSV schemas fail closed. |
| MAIN-006 | FIXED | Gate-E contexts are bound to their producing model and cross-model use fails before target supervision. |
| MAIN-007 | FIXED | Every module-local train/eval flag is restored on success and exception paths, including mixed parent/child modes. |
| MAIN-012 | FIXED | The query accepts explicit ordinary and AMP dtype pairs, preserves FP32 physical coordinates, promotes low-precision latent state only at sampling, and rejects unsupported pairs before sampling/MLP. |
| MAIN-010 | FIXED | Training and evaluation share a best-effort Git HEAD helper; Git absence is explicit JSON null rather than a failure or fabricated SHA. |
| MAIN-011 | FIXED | Stale task paths were removed, current owners are routed, and all declared entry/read paths have an executable validation. |
| MAIN-DOC-001 | FIXED | Public status separates implemented F1/F2 and G1-G4 software from pending F3/F4 execution, checkpoints, held-out evaluation, GPU evidence, and clinical/reconstruction claims. |
| MAIN-013 | DEFERRED | Explicitly deferred by the Human Gate; untouched. |

## Sol verification executed so far

- Gate A: `41 passed, 19 skipped`; targeted `compileall`; `git diff --check`.
- Launcher contract: `1 passed`; `bash -n`; ambient-command scan; `git diff --check`.
- Safe data path: `49 passed, 21 skipped`; targeted `compileall`; `git diff --check`.
- Artifact/checkpoint integration: `27 passed`; transactional and concurrency
  coverage included.
- Resume/server integration: `21 passed`; the focused resume worker also ran a
  successful CPU two-rank RNG round trip.
- Model/state integration: `55 passed, 2 skipped`; skips require CUDA.
- Provenance/governance integration: `37 passed`; all declared task paths
  validated; `compileall` and `git diff --check` passed.
- Initial full repository suite: `676 passed, 42 skipped, 26 subtests passed`.
- Independent system review then found that MAIN-012's first strict-equality
  implementation broke checked-in AMP profiles. The narrow continuation
  restored the explicit FP16/BF16-state + FP32-coordinate contract without
  narrowing geometry. Focused review passed `75 passed, 2 skipped`, and the
  independent reviewer changed its verdict from BLOCK to PASS.
- Final post-correction repository suite: `678 passed, 42 skipped, 26 subtests
  passed`; one tensor-to-scalar warning; no failures.
- DDP failure investigation used the healthy project Gloo runtime; the injected
  one-rank failure exited through the elastic launcher without a hang in a
  bounded manual run; no committed reproduction script/log was retained.
- Rank-indexed Python, NumPy, CPU Torch, and derived DataLoader worker streams
  were exercised. CUDA rank-state capture/restore is implemented in the schema
  but was not runtime-verified on this CUDA-less host.

## Remaining gates

- Actual-server performance baseline and isolated experiments, subject to
  hardware/data/runtime availability.
