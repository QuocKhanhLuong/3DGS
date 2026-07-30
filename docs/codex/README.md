# Codex Phase Handoffs

This directory is the executable handoff layer for contributors who use Codex
or another coding agent.  It complements the scientific documents under
`docs/reconstruction/`; it does not replace them.

## Status vocabulary

- `IMPLEMENTED`: code, tests, and a runnable command exist on `main`.
- `IN REVIEW`: implementation exists on a branch or pull request and must not be
  treated as an accepted research result.
- `PLANNED`: the interface and stop rules are documented, but runnable code does
  not yet exist.
- `BLOCKED`: the previous phase gate has not passed.

Passing unit tests means that the executable contract works.  It does **not**
mean that reconstruction accuracy, clinical fidelity, or paper novelty has been
validated.

## Phase matrix

| Phase | Purpose | Status | Runnable entry point |
|---|---|---|---|
| T0 | Canonical coordinates and physical-plane Gaussian renderer | IMPLEMENTED | `python -m pytest -q tests/render tests/contracts` |
| T0.5 | Legal sparse episodes, prediction receipts, cost, amplitude gauge | IMPLEMENTED | `python -m pytest -q tests/contracts tests/integration` |
| T1-A | Analytic evidence, explicit feature geometry, fixed supports, safe Gaussian bridge | IN REVIEW | `python -m smagm.cli.t1a --steps 4` |
| T1-B | Teacher-free micro-CNN and structural losses | BLOCKED by T1-A review | not available |
| T1-C | Sparse context-to-target trainer and E0/E1/E2 configs | BLOCKED by T1-B | not available |
| T2 | Physical anchor bootstrap and shared tiny local field | BLOCKED by T1 decision | not available |
| T3 | Anchor-Gaussian propagation and adaptive topology | BLOCKED by T2 | not available |
| T4 | Legal active sequence-slice trajectory | BLOCKED by static T3 baseline | not available |
| T5 | Full-grid reconstruction, uncertainty, and isolated export/evaluation | BLOCKED by T4 | not available |

## Required workflow for every phase

```text
read authoritative reconstruction docs
→ read the phase-specific Codex handoff
→ implement only the named tranche
→ add analytic/synthetic tests
→ run the phase command
→ run the complete CPU CI suite
→ open a PR
→ review scientific claims separately from software correctness
```

Every phase implementation must include:

1. typed interfaces and explicit tensor/coordinate semantics;
2. a small CPU synthetic example;
3. blocking unit/integration tests;
4. a CLI or script with `--help`;
5. configuration and artifact provenance where training is involved;
6. an honest README section listing non-claims;
7. no placeholder implementation for a later phase.

## Global non-claims

The repository is an executable research-code scaffold.  Until separate
experiments establish otherwise, it does not claim:

- clinical validity or real-patient deployment readiness;
- scanner acceleration from retrospective slice reveal;
- calibrated safety guarantees;
- equivalence to camera-view vanilla 3D Gaussian Splatting;
- successful recovery of unobserved pathology;
- a complete reproduction of any cited prior method.
