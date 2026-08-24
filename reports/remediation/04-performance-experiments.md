# Performance Experiments

Audit base: `0efeb94af72ffa067769e19afcd19ad358feefd2`  
Result: no real-server A/B experiment was run because the required environment
and Ox Alpha model were unavailable. No protocol-changing optimization was
implemented.

## Decisions

| ID | Decision | Evidence and consequence |
| --- | --- | --- |
| PERF-001 | ADOPT | The redundant mask copy was removed after exact content/dtype/shape/layout/source-independence tests. |
| PERF-002 | REJECT | Removing the remaining label scan would weaken the exported batch trust boundary; no production change. |
| PERF-003 | NEEDS_MORE_EVIDENCE | No real-volume normalization equivalence or objective-difference benchmark. |
| PERF-004 | NEEDS_MORE_EVIDENCE | No server worker/prefetch scaling, RSS, epoch-transition, or determinism evidence. |
| PERF-005 | NEEDS_MORE_EVIDENCE | No CUDA custom-batch pinning, H2D, overlap, or GPU-idle A/B. |
| PERF-006 | NEEDS_MORE_EVIDENCE | No fixed-seed CUDA objective/RNG/control-flow equivalence run. |
| PERF-007 | NEEDS_MORE_EVIDENCE | Decoder input identity/reusability was not proven on representative server execution; caching/batching was not attempted. |
| PERF-008 | NEEDS_MORE_EVIDENCE | No CUDA AMP throughput, memory, stability, objective, or metric comparison. |
| PERF-009 | NOT_AUTHORIZED | Not present in the accepted Human Gate decision set; untouched. |
| PERF-010 | NEEDS_MORE_EVIDENCE | Correctness cleanup retained the public trust-boundary scan; broader fusion lacks real-cost and adversarial-equivalence evidence. |
| PERF-011 | HUMAN_GATE | Provenance-bound preprocessing cache remains unimplemented; it requires source identity/hash, preprocessing version, relevant configuration, and invalidation tests. |
| PERF-012 | HUMAN_GATE | Delayed target materialization remains unimplemented; target-free auditability and measured target-load benefit require review. |
| PERF-013 | HUMAN_GATE | Gate-E scheduling/vectorization remains unimplemented; candidate order, RNG, floating point, and autograd semantics require design approval. |
| PERF-014 | HUMAN_GATE | Economic-stop policy is unchanged; it is a model/protocol decision, not a throughput tweak. |

## Required rerun matrix

On the actual CUDA server, first collect the baseline defined in
`03-server-performance-baseline.md`. Then run one isolated change per
experiment with warm-up and median/p90 measurements:

1. PERF-003 exact-source normalization equivalence.
2. PERF-004 worker counts 0/1/2/4/8 with throughput, idle time, RSS, and
   epoch transition latency.
3. PERF-005 `is_pinned()`, H2D events, and overlap before/after.
4. PERF-006 fixed-seed objective, valid-count, route, and RNG equivalence.
5. PERF-007 decoder-input hashes before any reuse design.
6. PERF-008 AMP throughput/memory/stability and metric deltas.
7. PERF-010 adversarial accept/reject equivalence plus measured scan cost.

An approved experiment (PERF-003/004/005/006/007/008/010) can become `ADOPT`
only after measured bottleneck, measured end-to-end improvement, correctness
equivalence, and acceptable resource cost are all present. PERF-001 was a
separately approved behavior-preserving cleanup and was adopted on exact
contract equivalence rather than a server speedup claim.
