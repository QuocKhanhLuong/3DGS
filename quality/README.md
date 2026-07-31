# Phase Gate Quality System

This directory is the machine-readable quality layer for implementation phases
T0 through T5. It complements, but does not replace:

1. `docs/strategies/` for authorization, claims, and Human Gates;
2. `docs/reconstruction/` for the theoretical method and cross-phase invariants;
3. `CODEBASE.md` for stable software ownership and dependency direction;
4. `docs/codex/` for the currently executable tranche.

The source of truth is `quality/checklists.json`. Run it with:

```bash
python scripts/check_phase.py --list
python scripts/check_phase.py T1B
python scripts/check_phase.py T1B --run --report-dir quality/reports
```

The default command validates and prints the selected checklist. `--run` executes
active automated checks. Future-phase checks remain `planned` until the phase
contract is frozen and an implementation provides exact evidence.

## Binding levels

- `invariant`: scientific or legal rule that implementation may not weaken;
- `contract`: typed behavior frozen by the phase design and Human Gate;
- `evidence`: command, test, metric, or artifact used to demonstrate a rule.

## Check modes

- `command`: executable repository command;
- `pytest`: focused pytest target;
- `human`: reviewer judgment is required;
- `planned`: requirement exists, but executable evidence is not yet bound;
- `file`: required artifact or document.

## Verdict policy

Automated blockers do not use an aggregate score. One blocker failure prevents
an automated pass. Agents may collect evidence and recommend
`READY_FOR_HUMAN_REVIEW`, but they must not write the final Human Gate decision.
The final phase verdict is one of:

- `PASS`
- `PASS_WITH_CONDITIONS`
- `REWORK`
- `FAIL`
- `BLOCKED`
- `PENDING_HUMAN_GATE`

A later phase may begin only when the previous automated gate passes, the
previous Human Gate is `PASS` or `PASS_WITH_CONDITIONS`, all conditions are
tracked, and the strategy explicitly authorizes the next tranche.

## Reports

With `--report-dir`, the runner writes one JSON and one Markdown report bound to
the current commit. Generated reports are evidence, not scientific conclusions.
Do not commit local reports that contain environment-specific paths or secrets.
