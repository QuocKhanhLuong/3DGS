# AGENTS.md

## Purpose

This repository is a research-code scaffold for sparse multi-sequence MRI
reconstruction with a patient-specific 3D Gaussian representation. Work must
preserve scientific validity, leakage barriers, physical-coordinate semantics,
and the currently authorized implementation tranche.

Keep context small. Do not explore the entire repository by default.

## Instruction and document precedence

Before editing, use the smallest applicable document set:

1. Read this file.
2. Read `README.md` for the repository thesis and current primary task.
3. Read `docs/codex/README.md` for the current phase status and runnable entry
   points.
4. For changes affecting research claims, phase authorization, tranche order,
   or stop decisions, read
   `docs/strategies/2026-07-29-isbi-realignment.md`.
5. Read only the task-specific plan, reconstruction module, phase document, or
   reproducibility note needed for the requested change.

Do not recursively summarize `docs/`. Files under `docs/meetings/`, historical
CVPR strategy documents, and old design notes are not authoritative unless the
user explicitly asks for historical context.

When documents conflict, follow the authority and precedence rules in the ISBI
realignment strategy. Code presence, passing tests, or an old plan does not by
itself authorize a later phase.

## Context-efficient repository navigation

Use this sequence before opening source files:

1. Identify the requested behavior, relevant phase, and likely package area.
2. Use Serena symbolic tools first when Serena MCP is connected:
   `get_symbols_overview`, `find_symbol`, and `find_referencing_symbols`.
3. Otherwise use targeted `rg` queries for filenames, definitions, imports, and
   call sites.
4. Initially inspect no more than 8 candidate files and no more than 250
   relevant lines from any one file.
5. Expand scope only when an unresolved import, contract, reference, or failing
   focused test requires it.

Never begin with recursive full-repository reads, broad `find .`, unrestricted
`grep -R`, or dumping complete large files. Prefer commands such as:

```bash
rg --files src/smagm tests docs/codex
rg -n "class_name|function_name|contract_name" src/smagm tests
sed -n 'START,ENDp' path/to/file.py
git diff --stat
git diff -- path/to/file.py
tail -n 120 path/to/log
```

Do not repeatedly reopen unchanged files. Reuse a concise task summary instead
of retaining large raw excerpts.

## Paths and files to avoid

Do not inspect these paths unless the task explicitly requires them:

- `.git/`
- `.venv/`
- `.pytest_cache/`
- `build/`, `dist/`, `htmlcov/`
- `*.egg-info/`, `__pycache__/`
- `.agenteam/artifacts/`, `.agenteam/events/`, `.agenteam/gates/`
- `.agenteam/handoffs/`, `.agenteam/locks/`, `.agenteam/memory/`
- `.agenteam/reports/`, `.agenteam/runs/`, `.agenteam/state/`
- experiment outputs, logs, caches, generated reports, checkpoints, and local
  datasets that may be added later

Do not read binary or volumetric data unless explicitly requested, including
`*.pt`, `*.pth`, `*.ckpt`, `*.npy`, `*.npz`, `*.nii`, `*.nii.gz`, `*.h5`,
`*.dcm`, images, and rendered artifacts. Inspect metadata, manifests, shapes,
or a bounded synthetic sample instead.

Do not read dependency lock files in full. Search for the exact package or
version needed.

## Scope and phase gates

Before changing code, state internally:

- the active phase and authorization source;
- the files expected to change;
- the contracts that must remain true;
- the smallest focused tests that validate the change.

Implement only the named tranche. Do not create placeholder modules, APIs,
configs, or abstractions for blocked future phases.

Unless the current authoritative status and the user's request explicitly
permit it, do not implement:

- T2 support-anchor bootstrap or anchor-local fields;
- T3 anchor-Gaussian propagation or adaptive topology;
- learned Gaussian birth, split, merge, or prune operations;
- T4 routing, learned utility, multi-wave planning, or adaptive acquisition;
- T5 full-volume export or isolated audit evaluation.

Stop at Human Gates. A local test pass is software evidence, not authorization
to continue to the next stage.

## Scientific and data-leakage contracts

Preserve all applicable contracts when editing:

- permanent sparse availability is distinct from episode context/target roles;
- target values are not revealed before rendering and prediction receipt;
- audit or fully sampled volumes do not enter the main training path;
- deployment acquisition costs retain exact `Decimal` semantics where defined;
- coordinate frames, voxel spacing, plane geometry, tensor axes, units, and
  through-plane profiles remain explicit;
- amplitude gauge, support coverage, unsupported regions, and failure states
  remain visible rather than silently normalized away;
- the renderer is described as a through-plane profile-aware Gaussian
  reference renderer, not ordinary camera-view 3D Gaussian Splatting;
- teacher-free paths remain teacher-free unless a clearly separated privileged
  upper-bound experiment is requested and authorized.

Do not claim clinical validity, scanner acceleration, safety guarantees,
recovery of unseen pathology, calibrated uncertainty, novelty, or superiority
from unit tests or smoke runs.

## Implementation rules

- Target Python 3.10 or newer as declared in `pyproject.toml`.
- Keep public interfaces typed and document non-obvious tensor shapes,
  coordinate systems, units, and invariants.
- Prefer small, composable changes over broad refactors.
- Do not alter adjacent modules merely for style.
- Preserve deterministic CPU execution for contract and synthetic tests.
- Add no production dependency without explicit user approval.
- Use NumPy and PyTorch consistently with existing code; do not introduce a
  parallel framework for a small convenience.
- Fail loudly on invalid geometry, illegal reveal order, shape mismatch,
  unsupported coverage, or non-finite values.
- Do not hide scientific assumptions inside defaults. Expose them in typed
  configuration or clearly named arguments.
- When behavior changes, update the nearest authoritative executable handoff or
  reproducibility note, but do not rewrite unrelated research documents.

## Testing and validation

During iteration, run the smallest relevant test target with concise output:

```bash
python -m pytest -q tests/<area> --tb=short
python -m pytest -q tests/path/test_file.py::test_name --tb=short
```

Run the full CPU suite once when the change is cross-cutting or before declaring
repository-wide validation:

```bash
python -m pytest -q
python -m compileall -q src tests
git diff --check
git status --short
```

Do not repeatedly run the full suite after every edit. Do not start long
training, download datasets or checkpoints, use a GPU, or run unbounded
experiments unless the user explicitly requests it.

For learned components, first add a bounded CPU synthetic or smoke test. A smoke
run verifies execution only; it does not establish reconstruction quality.

## Tool-output discipline

- Never print an entire large source file, Markdown corpus, test log, JSON file,
  or diff when a bounded excerpt is sufficient.
- Use `--tb=short`, quiet test modes, targeted diffs, and the final 120 log lines.
- Summarize repeated failures instead of pasting duplicate traces.
- Do not include generated artifacts or large numerical arrays in chat context.
- If a task becomes broader than initially expected, compact the findings into
  a short handoff: objective, relevant files, contracts, completed work, tests,
  and remaining issue.

## Completion report

At completion, report only:

1. files changed and the purpose of each;
2. tests or checks actually run and their result;
3. scientific or implementation assumptions not verified;
4. the Human Gate or phase boundary at which work stopped.

Do not present an implementation as an accepted research result.