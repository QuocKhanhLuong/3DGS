# Codex Phase Handoffs

This directory is the executable handoff layer for contributors who use Codex
or another coding agent.  It complements the scientific documents under
`docs/reconstruction/`; it does not replace them.

## Status vocabulary

Software state and Human Gate state are separate. A merged implementation is
not a Human Gate decision and is not a scientific validation claim.

- `IMPLEMENTED SOFTWARE CONTRACT`: executable contract and focused tests exist.
- `IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PENDING`: the bounded software
  tranche exists, but a Human Gate decision is still required.
- `IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PASSED`: the bounded software
  tranche and its committed Human Gate decision exist; this is still not a
  scientific validation claim.
- `IMPLEMENTED CANDIDATE — IN REVIEW`: an explicitly authorized software
  tranche is being hardened and reviewed; its Human Gate remains pending.
- `BLOCKED`: the phase is not authorized or its preceding Human Gate remains
  unresolved.

Passing unit tests means that the executable contract works.  It does **not**
mean that reconstruction accuracy, clinical fidelity, or paper novelty has been
validated.

## Phase matrix

| Phase | Purpose | Status | Runnable entry point |
|---|---|---|---|
| T0 | Canonical coordinates and physical-plane Gaussian renderer | IMPLEMENTED SOFTWARE CONTRACT | `python -m pytest -q tests/render tests/contracts` |
| T0.5 | Legal sparse episodes, prediction receipts, cost, amplitude gauge | IMPLEMENTED SOFTWARE CONTRACT | `python -m pytest -q tests/contracts tests/integration` |
| T1-A | Analytic evidence, explicit feature geometry, fixed supports, safe Gaussian bridge | IMPLEMENTED SOFTWARE CONTRACT | `python -m smagm.cli.t1a --steps 4` |
| T1-B | Teacher-free micro-CNN, structural losses, and cache contracts | IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PASSED | `python -m smagm.cli.t1b --help` |
| T1-C | Sparse context-to-target trainer and E0/E1/E2 experiment orchestration | IMPLEMENTED CANDIDATE — IN REVIEW | `python scripts/train.py --help` |
| T2 | Physical anchor bootstrap and shared tiny local field | BLOCKED | not available |
| T3 | Anchor-Gaussian propagation and adaptive topology | BLOCKED | not available |
| T4 | Legal active sequence-slice trajectory | BLOCKED | not available |
| T5 | Full-grid reconstruction, uncertainty, and isolated export/evaluation | BLOCKED | not available |

The machine-readable quality catalog separates `implementation_status` from
`human_gate_status`. T1-B has a committed Human Gate decision record. Run the
phase-gate evidence with:

```bash
python scripts/check_phase.py --list
python scripts/check_phase.py T1B
python scripts/check_phase.py T1B --run --report-dir quality/reports
```

The runner may report automated evidence, but only a Human decision may close a
Human Gate. T1-B now reports `PASS` only because that committed Human decision
record exists. Do not treat this software tranche as reconstruction success.

T1-C has separate Human authorization to implement and is an executable
candidate under review; its Human Gate remains pending. Run
`python scripts/check_phase.py T1C`; T2 remains blocked regardless of local
T1-C test results until T1-F/T1-R/T1-M evidence and an explicit T2 decision
exist.

## Required workflow for every phase

```text
read strategy/addendum
→ read docs/reconstruction
→ read CODEBASE.md
→ read docs/codex and the current handoff
→ read the relevant quality checklist
→ read the phase-specific Codex handoff
→ implement only the named tranche
→ add analytic/synthetic tests
→ run the phase-gate checklist
→ run the complete CPU CI suite
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

T1-B software demonstrations do not claim reconstruction accuracy, clinical
validity, pathology recovery, successful real-MRI training, scientific
superiority, completed support anchors, or propagation.
