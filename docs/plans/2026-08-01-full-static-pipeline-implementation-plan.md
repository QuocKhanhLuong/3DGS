# Full Static Reconstruction Pipeline — Continuous Implementation Plan

Date: 2026-08-01  
Status: **AUTHORIZED FOR ONE CONTINUOUS IMPLEMENTATION SPRINT**  
Branch target: `feature/full-static-reconstruction-pipeline`  
Final review: one consolidated owner review after implementation and evaluation

## 1. Goal

Implement the full static method from the merged T1-C trainer through T2
anchors/field, T3 propagation, and T5 reconstruction/evaluation without pausing
for intermediate scientific Human Gates.

The final executable flow is:

```text
permanently sparse multi-sequence MRI manifest
→ legal context/target episode
→ context-only preprocessing
→ E0/E1/E2 teacher-free evidence
→ exact feature cache
→ physical structural candidates
→ RAS-mm support anchors
→ cross-plane evidence aggregation
→ shared tiny anchor-local StructuralField
→ structural and volumetric seed Gaussians
→ bounded anchor–Gaussian propagation
→ immutable patient-specific Gaussian state
→ legal target-plane rendering
→ full physical-grid reconstruction
→ serialized prediction package
→ isolated consolidated evaluation
```

T4 routing is excluded.

## 2. Governing documents

Read in this order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/strategies/2026-08-01-full-static-pipeline-fast-track-authorization.md`
4. `CODEBASE.md`
5. `docs/reconstruction/README.md`
6. `docs/reconstruction/FULL_FLOW.md`
7. the nearest phase/module theory document;
8. this implementation plan;
9. the T2, T3, T5, and consolidated evaluation plans;
10. the nearest source files and tests.

This plan changes implementation sequencing only. It does not create a
scientific pass or override legality, leakage, claim, and audit rules.

## 3. Branch and workflow

Use one branch and one final pull request:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/full-static-reconstruction-pipeline
```

Do not implement directly on `main`.

Use small, phase-readable commits. Do not squash until the owner has reviewed
the final diff because commit boundaries are needed for debugging and
attribution.

Intermediate stage failures may be repaired immediately. Do not stop for owner
approval between stages unless:

- target/audit leakage is found;
- a contract conflict makes later state ambiguous;
- patient geometry cannot be validated;
- gradients are irreparably detached;
- a required data or label source is absent;
- the implementation would require T4 routing.

## 4. Stage 0 — Audit and harden merged T1-C

The merged T1-C code is the foundation, not assumed perfect.

Inspect and repair:

- deterministic modality-aware episode sampling;
- one-target reference semantics or fully implemented multi-target semantics;
- target modality to appearance-channel mapping;
- context-only normalization and degenerate-scale policy;
- exact input/cache identity;
- config-driven CLI execution;
- objective composition and training schedule;
- checkpoint safety under gradient accumulation;
- immutable run provenance;
- E0/E1/E2 matched experiment identity.

Required output:

```text
legal sparse episode
→ context-only state
→ commit/render/receipt/reveal
→ finite supported-mask loss
→ backward to intended parameters
```

T1-C fixes remain inside stable `data/`, `training/`, `losses/`, `features/`,
`baselines/`, and CLI ownership. Do not add T2 behavior to T1-C packages.

## 5. Stage 1 — T2 physical anchors

Implement:

```text
src/smagm/anchors/
├── contracts.py
├── candidates.py
├── bootstrap.py
├── consolidation.py
├── aggregation.py
├── frames.py
└── index.py
```

Responsibilities:

- candidate scoring from legal cached context evidence;
- deterministic top-k/threshold selection;
- feature-grid to RAS-mm lifting;
- physical NMS;
- cross-plane consolidation;
- structural/appearance evidence aggregation;
- partial and refined anchor frames;
- observability and provenance;
- bounded spatial queries.

Minimum tests:

- anisotropic spacing;
- rotated and oblique planes;
- deterministic tie-breaking;
- invalid topology exclusion;
- physical duplicate suppression;
- permutation-stable consolidation;
- conflict preservation;
- no target access.

## 6. Stage 2 — T2 shared StructuralField

Implement:

```text
src/smagm/fields/
├── contracts.py
├── local.py
├── blend.py
├── query.py
└── regularization.py
```

Core contract:

```text
normalized anchor-local coordinate + aggregated compact evidence
→ one shared tiny MLP
→ scalar StructuralField value
```

Requirements:

- one shared parameter set across all anchors and patients;
- no encoder rerun;
- no routing or topology ownership;
- nonnegative normalized support weights;
- explicit unsupported mask;
- finite and differentiable blending;
- anchor-order permutation invariance;
- field and gradient diagnostics;
- no unconditional SDF naming.

## 7. Stage 3 — T2 seed Gaussian memory and initial state

Implement:

```text
src/smagm/memory/
├── contracts.py
├── initialize.py
├── appearance.py
├── observability.py
└── index.py

src/smagm/state/
├── patient.py
├── builder.py
└── versioning.py
```

Create two explicit primitive types:

- `STRUCTURAL`: thin, anisotropic, anchor/field-aligned;
- `VOLUMETRIC`: broader modality-specific appearance support.

Requirements:

- positive-definite covariance;
- gauge provenance applied once;
- primitive-to-anchor binding;
- modality appearance validity;
- patient/manifest/config/state hashes;
- no target-derived initialization;
- differentiable renderer integration;
- immutable initial patient state.

## 8. Stage 4 — T3 bounded propagation

Follow
`docs/plans/2026-08-01-t3-anchor-gaussian-propagation-plan.md`.

Implement the default `P1` first:

```text
T2 seed state
→ deterministic bounded frontier proposals
→ child structural/volumetric support
→ parent and anchor provenance
→ uncertainty growth
→ duplicate/budget checks
→ immutable accepted transaction
→ fixed propagation rounds
```

Then add optional `P2/P3` topology variants only behind config switches.

Core files:

```text
src/smagm/memory/propagation.py
src/smagm/memory/topology.py
src/smagm/anchors/adaptation.py
src/smagm/state/update.py
src/smagm/state/serialization.py
```

T3 must not score or acquire new observations.

## 9. Stage 5 — Joint static training integration

Extend the training composition so the full static method can be optimized
through legal target-plane reconstruction.

Training variants must be compositional:

```text
encoder_variant = e0 | e1 | e2
representation_variant = interpolation | fixed | free | direct_anchor |
                         global_field | anchor_field
propagation_variant = p0 | p1 | p2 | p3
```

The resolved config must build only the selected modules.

The training step remains:

```text
sample legal assignment
→ open context only
→ encode/cache
→ build selected representation
→ freeze state
→ expose target geometry
→ commit target
→ render live prediction
→ register receipt from detached audit copy
→ reveal target
→ compute typed objective
→ backward
→ optimizer step
```

Required training safeguards:

- no target before render;
- one encoding pass per context observation per state build;
- patient-specific state not registered as global model parameters;
- independent E0/E1/E2 weights;
- matched initialization and opportunity;
- finite gradient-health reporting;
- exact checkpoint/provenance binding.

## 10. Stage 6 — T5 reconstruction and export

Follow
`docs/plans/2026-08-01-t5-reconstruction-export-evaluation-plan.md`.

Implement:

```text
src/smagm/reconstruction/
src/smagm/contracts/outputs.py
src/smagm/cli/reconstruct.py
scripts/reconstruct.py
```

Required capabilities:

- arbitrary physical-plane reconstruction;
- modality-aware appearance rendering;
- chunked full-grid reconstruction;
- spatial culling;
- support/unsupported outputs;
- field/anchor/Gaussian diagnostics;
- affine/orientation preservation;
- immutable reconstruction package;
- NIfTI/JSON/tensor export;
- exact artifact hashes.

## 11. Stage 7 — Isolated evaluator

Implement:

```text
src/smagm/evaluation/
src/smagm/cli/evaluate.py
src/smagm/cli/audit.py
scripts/evaluate.py
scripts/audit.py
```

Evaluation accepts serialized prediction packages, never mutable patient state.

Required metric groups:

- intensity reconstruction;
- gradient, edge, local contrast, and frequency;
- support and coverage;
- lesion/ROI/boundary fidelity where labels exist;
- uncertainty/reliability association and calibration when valid;
- compute, memory, cache, primitive, and latency accounting;
- patient-level paired statistics.

## 12. Stage 8 — Unified configuration

Create or complete:

```text
configs/data/
configs/model/
configs/training/
configs/reconstruction/
configs/evaluation/
configs/experiments/full_static_pipeline.json
```

The final experiment config must declare:

- dataset/cohort/manifests;
- modality mapping and preprocessing;
- encoder variant;
- anchor candidate and NMS policy;
- field architecture and support kernel;
- seed Gaussian policies;
- propagation variant and budgets;
- objective and schedule;
- output grid and chunking;
- evaluation metrics and statistics;
- seeds, hardware, precision, and output paths.

CLI overrides are small and recorded in the resolved config. Scientific
defaults are not duplicated inside CLI code.

## 13. Stage 9 — Mandatory baseline switches

The same execution path must support:

```text
B0 interpolation
B1 fixed-support Gaussian
B2 free Gaussian
B3 direct anchor-to-Gaussian
B4 anchor + StructuralField
B5 global field
B6 anchor + StructuralField + propagation
```

Ablation switches must remove the mechanism rather than leave unused modules or
extra parameter opportunity in place.

## 14. Stage 10 — Test policy

The owner will review scientific quality once at the end, but coding does not
wait until the end to detect contract failures.

Every stage must add focused CPU tests for:

- legality and fail-closed behavior;
- physical geometry;
- tensor shape/dtype/device;
- deterministic hashes and provenance;
- finite values and positive-definite covariance;
- autograd to intended parameters;
- explicit unsupported behavior;
- serialization round trip;
- component disable/ablation switches;
- no T4 routing.

Do not require large dataset experiments in unit CI.

## 15. Stage 11 — Synthetic end-to-end smoke

Required command:

```bash
python scripts/train.py \
  --config configs/experiments/full_static_pipeline.json \
  --variant full \
  --steps 2 \
  --output-dir /tmp/smagm-full-static-train

python scripts/reconstruct.py \
  --config configs/experiments/full_static_pipeline.json \
  --checkpoint /tmp/smagm-full-static-train/checkpoint.pt \
  --manifest /tmp/smagm-full-static-train/eval_manifest.json \
  --output-dir /tmp/smagm-full-static-reconstruction

python scripts/evaluate.py \
  --plan configs/evaluation/full_static_eval.json \
  --predictions /tmp/smagm-full-static-reconstruction \
  --output-dir /tmp/smagm-full-static-report
```

The smoke must prove execution, geometry, serialization, and gradients. It must
not claim scientific superiority.

## 16. Stage 12 — Final real-data evaluation

Follow
`docs/experiments/2026-08-01-consolidated-static-pipeline-evaluation-plan.md`.

Run one consolidated experiment matrix after code and configs are frozen.
Generate one final evidence package for owner review.

The final matrix must include evidence, representation, and propagation
attribution plus the full model and interpolation floor.

## 17. Quality status during implementation

Do not mark Human Gates as passed.

Recommended lifecycle:

```text
not started: implementation_status = planned
being implemented: implementation_status = active
merged software: implementation_status = implemented
before owner decision: human_gate_status = pending
```

Automated checks may pass while the phase verdict remains
`PENDING_HUMAN_GATE`.

## 18. Required final validation

Run from a clean checkout on the exact final branch commit:

```bash
python scripts/check_phase.py --list
python -m pytest -q tests/quality --tb=short
python -m pytest -q
python -m compileall -q src tests scripts
python scripts/train.py --help
python scripts/reconstruct.py --help
python scripts/evaluate.py --help
python scripts/audit.py --help
git diff --check
git diff --cached --check
git status --short
```

Run phase evidence for every implemented stage:

```bash
python scripts/check_phase.py T1C --run --report-dir quality/reports
python scripts/check_phase.py T2  --run --report-dir quality/reports
python scripts/check_phase.py T3  --run --report-dir quality/reports
python scripts/check_phase.py T5  --run --report-dir quality/reports
```

Expected before final owner review:

```text
automated evidence: PASS where checks are complete
scientific/Human Gate: PENDING_HUMAN_GATE
T4: BLOCKED
```

## 19. Preferred commit sequence

```text
1.  fix(t1c): harden legal multi-sequence trainer contracts
2.  feat(t2): add physical anchors and cross-plane aggregation
3.  feat(t2): add shared anchor-local StructuralField
4.  feat(t2): initialize structural and volumetric Gaussian memory
5.  feat(t3): add bounded anchor-Gaussian propagation
6.  feat(t3): add optional conservative topology transactions
7.  feat(train): integrate full static representation variants
8.  feat(t5): add plane and full-grid reconstruction
9.  feat(t5): add export and immutable reconstruction packages
10. feat(eval): add isolated metrics and patient-level statistics
11. feat(config): add unified static pipeline experiment matrix
12. test(pipeline): add full static smoke and ablation checks
13. docs(codex): record implemented unvalidated software state
14. quality: activate T2 T3 T5 automated evidence
```

## 20. Final deliverables

The final pull request must contain:

- full static source pipeline;
- configs for all mandatory variants;
- focused tests and end-to-end smoke;
- CLIs with `--help`;
- exact provenance and immutable artifacts;
- reconstruction and isolated evaluation packages;
- updated CODEBASE implementation snapshot;
- T2/T3/T5 Codex handoffs;
- active quality checks;
- no T4 code;
- one reproducible consolidated evaluation report or a precise command plan for
  the unavailable real-data portion.

Do not merge automatically. The owner reviews the complete branch and final
scientific evidence together.
