# Phase Gate System

## Purpose

The phase-gate system converts implementation evidence into a short, readable
review packet without replacing scientific review or Human Gates.

```text
implementation on an authorized tranche
→ focused tests during development
→ exact-commit automated phase check
→ Markdown and JSON evidence report
→ independent code and scientific review
→ Human Gate decision
```

The machine-readable source is `quality/checklists.json`; the executable runner
is `scripts/check_phase.py`.

## Authority

The checklist cannot authorize work by itself. Authority remains:

1. `docs/strategies/` for thesis, claims, tranche authorization, and Human Gates;
2. `docs/reconstruction/` for the intended scientific method;
3. `CODEBASE.md` for final software ownership and dependency direction;
4. `docs/codex/` for current executable status;
5. `quality/checklists.json` for evidence expected before a gate decision.

When a checklist conflicts with an authoritative design, stop and update the
checklist through review rather than silently weakening the design.

## Binding levels

### Invariant

A scientific, legal, or physical rule that implementation may not weaken, such
as prediction-before-reveal, canonical RAS-mm geometry, renderer purity, or
sealed final-audit isolation.

### Contract

A typed or algorithmic behavior frozen by phase design, such as the anchor
record, propagation provenance, output-affine round trip, or fixed support
selection.

### Evidence

A command, focused test, report, metric, or artifact that demonstrates an
invariant or contract. Evidence paths can change when implementation is
refactored, but the underlying requirement remains.

## Checklist lifecycle

Future phases are intentionally represented in the catalog without executable
implementation:

- `planned`: requirement and expected owner are recorded, but no executable test
  is claimed;
- `implemented`: software exists and exact tests or commands can run;
- `active`: reserved for an explicitly authorized tranche under development;
- `planned`: requirement exists, but no implementation is authorized.

Human Gate state is separate: current T1-B software is `implemented` with a
committed `passed` Human Gate, T0/T0.5/T1-A are `implemented` with
`retrospective_unrecorded` status, T1-C is `active` with a `pending` Human
Gate, and T2 through T5 are `planned` and `blocked`.

When a phase design freezes, replace each relevant `planned` entry with a
focused `pytest`, `command`, or `file` check. Do not delete an invariant merely
because the implementation chose a different API.

## Running a gate

List phases:

```bash
python scripts/check_phase.py --list
```

Read a phase without executing commands:

```bash
python scripts/check_phase.py T1B
```

Execute all active automated checks and write evidence:

```bash
python scripts/check_phase.py T1B \
  --run \
  --report-dir quality/reports
```

A planned phase reports `BLOCKED`; this is not a software failure. It means the
phase is not yet authorized or lacks bound executable evidence.

## Roles

- **developer**: implements only the authorized tranche and supplies focused
  evidence;
- **QA reviewer**: checks tests, failure behavior, numerical stability, and
  regression evidence;
- **reproducibility auditor**: checks commit, configuration, seeds, environment,
  compute, and artifacts;
- **scientific reviewer**: checks leakage, fairness, mechanism validity,
  terminology, medical risk, and claim discipline;
- **gate owner**: records the final Human Gate decision and conditions.

The configured roles are `pm`, `reviewer`, `qa`, `reproducibility_auditor`,
`medical_data_steward`, `experiment_lead`, `architect`, `dev`, and `researcher`.
The implementing agent and all other agents must not approve a Human Gate.

## Gate decisions

- `PASS`: phase may close and the next tranche may be considered;
- `PASS_WITH_CONDITIONS`: phase may close only with tracked conditions;
- `REWORK`: implementation remains in the same phase;
- `FAIL`: the hypothesis or mechanism is rejected under the declared protocol;
- `BLOCKED`: authorization or prerequisites are missing;
- `PENDING_HUMAN_GATE`: automated evidence passed, but human review is incomplete.

There is no percentage score. A target leak, invalid geometry, audit breach, or
other blocker cannot be compensated by many minor passes.

## Phase summaries

### T0

Validates physical planes, Gaussian tensors, amplitude-gauge provenance,
through-plane rendering, unsupported pixels, and differentiability.

### T0.5

Validates role-free permanent availability, immutable episodes, exact receipts,
prediction-before-reveal, split isolation, cost semantics, and frozen state.

### T1-A

Validates physically aligned analytic evidence, deterministic fixed supports,
local-to-RAS Gaussian construction, and one synthetic gradient path.

### T1-B

Validates common E0/E1/E2 output contracts, per-item geometry, value-independent
support topology, exact cache keys, teacher-free structural objectives, and
three diagnostic render/backward paths.

### T1-C

Validates legal context-to-target training, supported-mask losses,
prediction-before-reveal end to end, matched E0/E1/E2 experiments, independent
weights, and immutable experiment provenance.

### T2

Will validate physical anchor candidates, RAS-mm consolidation, cache-only
evidence aggregation, partial observability, one shared tiny local structural
field, blending, complexity, and naming discipline.

### T3

Will validate separate structural and volumetric memory, patient-state
versioning, bounded propagation, parent provenance, uncertainty growth,
render-before-update assimilation, adaptive topology budgets, and matched
compute.

### T4

Will validate no candidate-pixel access, exact cost and budget, explicit utility
components, legal trajectory ordering, deterministic stopping, matched routing
baselines, and patient-level quality-cost evidence.

### T5

Will validate state-only full reconstruction, chunking, affine/export round
trips, explicit unsupported coverage, uncertainty provenance, corrupt-artifact
failure, serialized prediction isolation, sealed audit evaluation, patient-level
statistics, medical fidelity, and claim-to-artifact traceability.

## Required report interpretation

A passing command demonstrates only the requirement named by that check. In
particular:

- passing unit tests does not establish reconstruction quality;
- a decreasing synthetic loss does not establish medical validity;
- a structural warm-up objective does not prove useful representation;
- a visually dense reconstruction does not prove supported anatomy;
- an uncertainty tensor is not calibrated without a calibration protocol;
- a structural field is not a signed-distance field without sign, distance, and
  Eikonal evidence;
- retrospective active acquisition does not prove clinical workflow benefit.

The Human Gate must state what evidence is still absent before authorizing the
next claim or tranche.
