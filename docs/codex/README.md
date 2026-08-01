# Codex Phase Handoffs

This directory is the executable handoff layer for contributors who use Codex
or another coding agent. It complements the scientific documents under
`docs/reconstruction/`; it does not replace them.

## Status vocabulary

Software state, implementation authorization, and Human Gate state are
separate. A merged implementation is not a Human Gate decision and is not a
scientific validation claim.

- `IMPLEMENTED SOFTWARE CONTRACT`: executable contract and focused tests exist.
- `IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PENDING`: the bounded software
  tranche exists, but a Human Gate decision is still required.
- `IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PASSED`: the bounded software
  tranche and its committed Human Gate decision exist; this is still not a
  scientific validation claim.
- `AUTHORIZED FOR IMPLEMENTATION — NOT YET IMPLEMENTED`: owner permission and a
  maintained plan exist, but executable software evidence does not yet exist.
- `BLOCKED`: implementation is not authorized or is explicitly excluded.

Passing unit tests means that the executable contract works. It does **not**
mean that reconstruction accuracy, clinical fidelity, or paper novelty has been
validated.

## Phase matrix

| Phase | Purpose | Status | Runnable entry point |
|---|---|---|---|
| T0 | Canonical coordinates and physical-plane Gaussian renderer | IMPLEMENTED SOFTWARE CONTRACT | `python -m pytest -q tests/render tests/contracts` |
| T0.5 | Legal sparse episodes, prediction receipts, cost, amplitude gauge | IMPLEMENTED SOFTWARE CONTRACT | `python -m pytest -q tests/contracts tests/integration` |
| T1-A | Analytic evidence, explicit feature geometry, fixed supports, safe Gaussian bridge | IMPLEMENTED SOFTWARE CONTRACT | `python -m smagm.cli.t1a --steps 4` |
| T1-B | Teacher-free micro-CNN, structural losses, and cache contracts | IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PASSED | `python -m smagm.cli.t1b --help` |
| T1-C | Sparse context-to-target trainer and E0/E1/E2 experiment orchestration | IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PENDING | `python scripts/train.py --help` |
| T2 | Physical anchor bootstrap, shared tiny StructuralField, and seed Gaussians | AUTHORIZED FOR IMPLEMENTATION — NOT YET IMPLEMENTED | planned in `docs/designs/2026-07-31-t2-anchor-local-field-plan.md` plus the 2026-08-01 addendum |
| T3 | Bounded anchor–Gaussian propagation and optional topology ablations | AUTHORIZED FOR IMPLEMENTATION — NOT YET IMPLEMENTED | planned in `docs/plans/2026-08-01-t3-anchor-gaussian-propagation-plan.md` |
| T4 | Legal active sequence-slice trajectory | BLOCKED | not available |
| T5 | Plane/full-grid reconstruction, export, isolated evaluation, and statistics | AUTHORIZED FOR IMPLEMENTATION — NOT YET IMPLEMENTED | planned in `docs/plans/2026-08-01-t5-reconstruction-export-evaluation-plan.md` |

## Current implementation program

The repository owner has authorized one continuous static-pipeline sprint:

```text
T1-C hardening
→ T2 anchors + StructuralField + seed Gaussians
→ T3 bounded propagation
→ T5 reconstruction/export/evaluation
```

Use:

- `docs/strategies/2026-08-01-full-static-pipeline-fast-track-authorization.md`;
- `docs/plans/2026-08-01-full-static-pipeline-implementation-plan.md`;
- the phase-specific T2, T3, T5 plans;
- `docs/experiments/2026-08-01-consolidated-static-pipeline-evaluation-plan.md`.

T4 remains blocked. Intermediate software tests do not create scientific passes.
T1-C, T2, T3, and T5 remain pending final consolidated evidence and an explicit
owner decision.

The machine-readable quality catalog separates `implementation_status` from
`human_gate_status`. Update each phase from `planned` to `active` only when its
implementation starts, and to `implemented` only when executable software and
its declared evidence exist. Human-gate status remains `pending` until an owner
record is committed.

## Required workflow for the fast-track branch

```text
read fast-track authorization
→ read CODEBASE and nearest theory/plan
→ implement only the current stable package responsibility
→ add focused legality/geometry/autograd/numeric tests
→ preserve config switches for matched ablations
→ continue to the next static stage after software guards pass
→ run complete CPU suite and phase checks
→ freeze configs/checkpoints/predictions
→ run one consolidated isolated evaluation
→ obtain one final owner review
```

Every phase implementation must include:

1. typed interfaces and explicit tensor/coordinate semantics;
2. a small CPU synthetic example;
3. blocking unit/integration tests;
4. a CLI or script with `--help`;
5. configuration and artifact provenance where training is involved;
6. honest non-claims;
7. a switch that recovers the matched simpler ablation;
8. no T4 routing implementation.

## Global non-claims

The repository is an executable research-code scaffold. Until separate
experiments establish otherwise, it does not claim:

- clinical validity or real-patient deployment readiness;
- scanner acceleration from retrospective slice reveal;
- calibrated safety guarantees;
- equivalence to camera-view vanilla 3D Gaussian Splatting;
- successful recovery of unobserved pathology;
- a complete reproduction of any cited prior method;
- that planned or implemented anchors, fields, propagation, or full-grid export
  improve reconstruction;
- that automated software evidence is a scientific Human Gate pass.
