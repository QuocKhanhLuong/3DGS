# Full Static Pipeline Fast-Track Authorization — 2026-08-01

Decision authority: repository owner, by explicit instruction in the active
implementation thread.

## Precedence

This later owner decision supersedes only the earlier implementation-blocking
and per-phase stop requirements for T2, T3, and T5 in
[`2026-07-29-isbi-realignment.md`](2026-07-29-isbi-realignment.md). The original
scientific thesis, legality rules, claim restrictions, evidence requirements,
medical-fidelity requirements, stop/demotion rules, and T4 routing prohibition
remain active.

Where the earlier strategy says T2+ implementation is prohibited pending an
intermediate Human Gate, this document now permits continuous static-pipeline
implementation. It does not retroactively mark any gate as passed.

## Decision

**Continuous implementation of the static reconstruction pipeline is
`AUTHORIZED` for T1-C hardening, T2, T3, and T5.**

The authorized implementation chain is:

```text
legal permanently sparse multi-sequence MRI episodes
→ teacher-free structural and appearance evidence
→ physical support anchors
→ shared anchor-local StructuralField
→ structural and volumetric seed Gaussians
→ bounded anchor–Gaussian propagation
→ static patient-specific Gaussian state
→ target-plane and full-grid reconstruction
→ isolated consolidated evaluation
```

T4 active acquisition and trajectory routing remain `BLOCKED` and are not part
of this authorization.

## What this decision changes

The owner authorizes one continuous engineering sprint and one final integrated
review instead of stopping implementation after every intermediate phase.
T1-C, T2, T3, and T5 may therefore be implemented sequentially on one bounded
static-pipeline branch, provided that their package boundaries, contracts,
configuration switches, tests, and ablations remain separable.

Intermediate smoke tests remain mandatory for legality, geometry, autograd,
numerical stability, and serialization. Intermediate scientific Human Gates do
not pause implementation, but they are consolidated into the final evaluation
and owner review.

## What this decision does not mean

This document is implementation authorization only. It is not:

- a T1-C, T2, T3, or T5 Human Gate pass;
- evidence that anchors, the StructuralField, propagation, or Gaussian memory
  improve reconstruction;
- permission to claim medical validity, novelty priority, calibrated
  uncertainty, clinical readiness, or state of the art;
- permission to use non-manifest, target, validation, or sealed-audit pixels in
  training or patient-state construction;
- permission to implement T4 routing.

Until the consolidated evaluation and an explicit owner decision are committed,
all newly implemented stages remain **implemented or active but scientifically
unvalidated**.

## Authorized scope

### T1-C hardening

T1-C may be repaired where required for the full pipeline, including legal
multi-sequence episode semantics, context-only preprocessing, exact cache
identity, modality-to-appearance mapping, config-driven execution, objective
composition, checkpointing, and provenance.

### T2

T2 may implement physical candidate generation, RAS-mm anchors, cross-plane
consolidation, compact evidence aggregation, one shared anchor-local
StructuralField, stable blending, seed Gaussian initialization, and the minimum
immutable initial patient state.

### T3

T3 may implement bounded static anchor–Gaussian propagation, observability,
uncertainty growth, parent provenance, conservative geometry updates, and
optional fixed-schedule topology proposals required for attribution. It may not
inspect hidden targets or use an active acquisition loop.

### T5

T5 may implement plane reconstruction, chunked full-grid reconstruction,
physical-affine-preserving export, immutable reconstruction packages, isolated
evaluation, medical-fidelity metrics, uncertainty diagnostics, compute
accounting, and patient-level statistics.

## Required final decision package

The owner will review one consolidated evidence package containing:

1. leakage and legality evidence;
2. geometry and physical-coordinate evidence;
3. T1 E0/E1/E2 attribution;
4. interpolation and fixed/free Gaussian baselines;
5. direct-anchor, anchor-field, and propagation ablations;
6. target-plane and full-volume metrics;
7. lesion/ROI, boundary, coverage, and failure metrics where labels exist;
8. parameter, FLOP, runtime, memory, cache, and primitive accounting;
9. uncertainty and unsupported-region diagnostics;
10. patient-level paired statistics and failure cases;
11. an exact commit, resolved configs, manifests, checkpoints, and artifact
    hashes.

Only a later explicit owner decision may mark any scientific gate as passed.

## Implementation references

- [`../plans/2026-08-01-full-static-pipeline-implementation-plan.md`](../plans/2026-08-01-full-static-pipeline-implementation-plan.md)
- [`../designs/2026-07-31-t2-anchor-local-field-plan.md`](../designs/2026-07-31-t2-anchor-local-field-plan.md)
- [`../designs/2026-08-01-t2-fast-track-addendum.md`](../designs/2026-08-01-t2-fast-track-addendum.md)
- [`../plans/2026-08-01-t3-anchor-gaussian-propagation-plan.md`](../plans/2026-08-01-t3-anchor-gaussian-propagation-plan.md)
- [`../plans/2026-08-01-t5-reconstruction-export-evaluation-plan.md`](../plans/2026-08-01-t5-reconstruction-export-evaluation-plan.md)
- [`../experiments/2026-08-01-consolidated-static-pipeline-evaluation-plan.md`](../experiments/2026-08-01-consolidated-static-pipeline-evaluation-plan.md)
