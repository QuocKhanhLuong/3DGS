# Codex Handoff — Full Static Pipeline Fast-Track

Status: **AUTHORIZED EXECUTION HANDOFF**  
Target branch: `feature/full-static-reconstruction-pipeline`  
Excluded phase: T4 routing

## Mission

Continue from current `main`, harden T1-C where necessary, then implement T2,
T3, and T5 in one continuous branch. Do not pause for intermediate scientific
Human Gates. Preserve focused software checks after each stage and produce one
final consolidated evaluation package for owner review.

## Read first

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/strategies/2026-08-01-full-static-pipeline-fast-track-authorization.md`
4. `CODEBASE.md`
5. `docs/plans/2026-08-01-full-static-pipeline-implementation-plan.md`
6. the nearest T1-C/T2/T3/T5 theory and plan;
7. `docs/experiments/2026-08-01-consolidated-static-pipeline-evaluation-plan.md`
8. nearest source files and tests.

## Safe setup

```bash
git fetch origin
git status --short
```

Ignore only generated local agent paths through `.git/info/exclude` when they
appear:

```text
.agenteam/history/
.ateam-worktrees/
```

Stop for any other local change.

Then:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/full-static-reconstruction-pipeline
```

Do not push directly to `main`. Do not merge automatically.

## Required implementation sequence

```text
S0  T1-C audit and hardening
S1  physical anchors
S2  shared anchor-local StructuralField
S3  structural and volumetric seed Gaussian memory
S4  bounded anchor–Gaussian propagation
S5  joint static training variants
S6  plane and full-grid reconstruction
S7  export and immutable prediction packages
S8  isolated evaluation and statistics
S9  unified configs and mandatory ablations
S10 synthetic end-to-end smoke
S11 final consolidated real-data evaluation
```

Use the stable package ownership in `CODEBASE.md`. Do not create phase-named
production modules when an existing package owns the responsibility.

## Hard boundaries

Never:

- open target pixels before render and receipt registration;
- use non-manifest or sealed-audit pixels in training/state construction;
- rerun the encoder during anchor movement or propagation when cache sampling is
  sufficient;
- put patient-specific state into persistent global trainable parameters;
- call an unverified field an SDF;
- hide unsupported pixels or voxels;
- implement T4 candidate scoring, utility, acquisition, or routing;
- mark any scientific Human Gate as passed;
- claim that synthetic smoke proves reconstruction quality.

## Required variants

The final config system must switch among:

```text
E0 analytic evidence
E1 raw shallow CNN
E2 analytic scaffold + micro-CNN

R0 interpolation
R1 fixed-support Gaussian
R2 free Gaussian
R3 direct anchor-to-Gaussian
R4 anchor + shared StructuralField
R5 global field

P0 no propagation
P1 bounded fixed propagation
P2/P3 optional topology variants
```

The `FULL` method is E2 + R4 + the selected propagation variant.

## Continue-without-review software guards

Before consuming a stage in the next stage, its focused tests must prove:

- fail-closed legality;
- correct RAS-mm geometry;
- finite values and positive-definite covariance;
- intended autograd path;
- deterministic hashes and provenance;
- bounded memory/primitive growth;
- explicit unsupported behavior;
- serialization round trip where state is persisted;
- a working switch to the matched simpler ablation.

These guards authorize engineering continuation only, not scientific claims.

## Final commands

```bash
python scripts/check_phase.py --list
python -m pytest -q tests/quality --tb=short
python -m pytest -q
python -m compileall -q src tests scripts
python scripts/train.py --help
python scripts/reconstruct.py --help
python scripts/evaluate.py --help
python scripts/audit.py --help
python scripts/check_phase.py T1C --run --report-dir quality/reports
python scripts/check_phase.py T2  --run --report-dir quality/reports
python scripts/check_phase.py T3  --run --report-dir quality/reports
python scripts/check_phase.py T5  --run --report-dir quality/reports
git diff --check
git diff --cached --check
git status --short
```

Run the synthetic full-pipeline train → reconstruct → evaluate flow declared in
the master implementation plan.

## Preferred commit sequence

```text
fix(t1c)
feat(t2-anchors)
feat(t2-field)
feat(t2-memory)
feat(t3-propagation)
feat(t3-topology)          # optional and config-gated
feat(static-training)
feat(t5-reconstruction)
feat(t5-export)
feat(evaluation)
feat(configs)
test(full-static-pipeline)
docs(codex)
quality
```

## Final report

Return:

1. starting main SHA and final branch SHA;
2. commits and changed files by stage;
3. T1-C repairs;
4. T2 anchor/field/memory contracts;
5. T3 propagation variants and budgets;
6. T5 reconstruction/export/evaluation capabilities;
7. mandatory ablation switches;
8. exact test and phase-runner results;
9. synthetic smoke artifacts;
10. real-data evaluation status and missing inputs, when any;
11. confirmation that T4 is absent;
12. confirmation that all scientific gates remain pending owner review;
13. final clean git status.

Open one pull request into `main`. Do not merge automatically.
