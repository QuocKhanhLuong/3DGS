# T1-A — Executable Analytic-to-Gaussian Reference

Status: `IMPLEMENTED SOFTWARE CONTRACT`
Scope: analytic evidence, feature-grid geometry, deterministic supports, safe
fixed-topology Gaussian bridge, and one synthetic end-to-end gradient path.

## Scientific purpose

T1-A isolates the downstream bridge before introducing a learned encoder.  It
asks only whether compact slice-aligned evidence can be sampled at fixed
physical supports, converted into a valid Gaussian state, and optimized through
the physical-plane renderer.

```text
synthetic observed slice
→ differentiable analytic evidence bank
→ explicit feature-grid-to-plane transform
→ deterministic row-major support points
→ independent per-support Gaussian head
→ gauge-safe fixed Gaussian state
→ physical target-plane renderer
→ reconstruction loss
→ gradient to the head and sampled evidence
```

T1-A does not implement support-anchor learning, local structural fields,
propagation, active routing, or real-data training.

## Run

```bash
python -m pip install -e .
python -m smagm.cli.t1a --steps 4
python -m pytest -q tests/features tests/baselines
python -m pytest -q
```

The current reading order also includes [`CODEBASE.md`](../../CODEBASE.md) for
stable software ownership and [`quality/checklists.json`](../../quality/checklists.json)
for executable and Human Gate evidence requirements.

Expected demo output contains finite values for:

```text
initial_loss
final_loss
gradient_norm
support_count
supported_fraction
```

The demo is successful when it runs, produces supported pixels, and retains a
non-zero gradient.  Loss improvement over four steps is informative but is not
a blocking scientific result.

## Files owned by this tranche

```text
src/smagm/features/
├── analytic.py
└── contracts.py

src/smagm/baselines/
├── fixed_support.py
└── fixed_gaussian.py

src/smagm/cli/t1a.py

tests/features/
tests/baselines/
```

## Codex continuation prompt

```text
You are working on QuocKhanhLuong/3DGS T1-A only.

Read:
- CODEBASE.md
- docs/reconstruction/README.md
- docs/reconstruction/modules/EVIDENCE_ENCODER.md
- docs/codex/README.md
- quality/checklists.json
- the nearest T1-A source files and tests

Inspect all current T0/T0.5 contracts and the T1-A files before editing.

Your task is to improve or debug the T1-A executable reference without adding
T1-B, T2, T3, T4, or T5 placeholders.

Hard constraints:
1. Preserve canonical RAS-mm and pixel-centre semantics.
2. Support selection must be deterministic and value-independent.
3. E0/E1/E2 must later be able to use identical support topology and identical
   Gaussian-head architecture.
4. Every raw-to-runtime Gaussian conversion applies the amplitude gauge exactly
   once; render_plane applies it zero times.
5. Keep render_plane pure and differentiable.
6. Unsupported pixels remain explicit and are never silently filled with zero.
7. Do not claim accuracy, clinical validity, propagation, or active acquisition.

Before finishing, run:
- python -m smagm.cli.t1a --steps 4
- python -m pytest -q tests/features tests/baselines
- python -m pytest -q
- python -m compileall -q src tests
- git diff --check

Return changed files, test output, unresolved risks, and confirmation that no
T1-B or T2+ code was added.
```

## Retrospective evidence and maintained baseline

T1-B software and its committed Human Gate are passed. T1-A remains the
maintained analytic attribution and regression baseline. The T1-B software
merge and Human Gate did not replace the T1-A baseline or establish
reconstruction accuracy. T1-C was subsequently authorized and implemented
under its separate handoff; T2 remains blocked.

The retrospective T1-A evidence requirements remain:

- all CPU CI checks pass on an exact commit;
- feature-grid alignment is independently reviewed;
- support topology is shown to be value-independent;
- covariance and amplitude parameterizations remain valid under optimization;
- the rendered loss has a finite gradient to the Gaussian head and evidence;
- no hidden target data enter state construction.
