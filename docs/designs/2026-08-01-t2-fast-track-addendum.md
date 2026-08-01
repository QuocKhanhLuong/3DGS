# T2 Fast-Track Addendum — 2026-08-01

Status: **IMPLEMENTATION AUTHORIZED — SCIENTIFIC VALIDATION DEFERRED**

This addendum supersedes only the implementation-blocking language in
[`2026-07-31-t2-anchor-local-field-plan.md`](2026-07-31-t2-anchor-local-field-plan.md).
The T2 architecture, scope, non-claims, package ownership, baselines, and stop
rules in that document remain active.

Under the owner decision in
[`../strategies/2026-08-01-full-static-pipeline-fast-track-authorization.md`](../strategies/2026-08-01-full-static-pipeline-fast-track-authorization.md),
T2 may now be implemented immediately after or alongside the required T1-C
hardening on the same static-pipeline branch.

## Implementation rule

T2 implementation may proceed without an intermediate scientific Human Gate,
but it must remain independently switchable and testable so the final
consolidated evaluation can compare:

```text
fixed support Gaussian
free Gaussian
anchor → Gaussian without field
anchor + shared StructuralField
```

T2 is not considered scientifically passed merely because its tests pass or its
code is merged.

## Bounded scope

Authorized:

- physical candidate generation from legal cached context evidence;
- RAS-mm lifting and physical non-maximum suppression;
- deterministic cross-plane consolidation;
- structural/appearance evidence aggregation;
- partial and evidence-refined anchor frames;
- one shared low-capacity anchor-local StructuralField;
- stable field blending and explicit unsupported output;
- structural and volumetric seed Gaussian initialization;
- immutable initial patient state and provenance;
- matched T2 baselines, synthetic tests, and config switches.

Still excluded from T2:

- propagation and iterative support growth;
- residual assimilation;
- adaptive birth, split, merge, and prune;
- active routing;
- full-grid reconstruction and final evaluation.

Those responsibilities belong to the authorized T3 and T5 plans.

## Required software checks before continuing to T3

The continuous sprint does not stop for scientific review, but T3 code must not
consume T2 outputs until these software guards pass:

1. anchors are finite, deterministic, patient-bound, and within declared RAS-mm
   bounds;
2. physical NMS is invariant to image spacing and plane orientation;
3. aggregation uses only legal cache entries;
4. field blending is finite, differentiable, and permutation-stable;
5. unsupported regions remain explicit;
6. seed Gaussian covariance is positive definite;
7. primitive kind and anchor provenance are retained;
8. gradients reach the encoder, field, and Gaussian initialization path;
9. no target or audit pixel enters bootstrap state;
10. the T2 mechanism can be disabled to recover matched simpler baselines.
