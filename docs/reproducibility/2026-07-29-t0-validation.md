# T0 CPU validation record

Date: 2026-07-29  
Scope: synthetic T0 contract and reference-kernel validation only  
Status: locally verified in a dirty, uncommitted worktree

## Source state

- Branch: `main`
- Base commit: `d79fbf7a2f7ccfcc35e0b0ddcc4135f0bcdba2f1`
- Worktree: dirty; T0 implementation and the intentional deletion of four
  deprecated root documents are not committed.
- Authoritative specification: `docs/` only.

## Environment

- Hardware architecture: Apple arm64 CPU
- Operating system: macOS 26.0.1 (build 25A362)
- Python: 3.10.0
- PyTorch: 2.13.0
- NumPy: 2.2.6
- pytest: 9.1.1
- Dependency lock:
  `requirements/t0-cpu-macos-arm64-py310.lock`
- Lock SHA-256:
  `491441198a7e562c98e3e25a571a3106cdb23572263a7e1b6fa4a787c4ceacdc`
- Package declaration SHA-256:
  `3730c5757f00b5cc3a6204bde90aac140e713645a87cb6e74ddc622818cb66ee`

## Determinism

- Recorded renderer test seed: `20260729`
- Device: CPU reference path
- Inputs: generated synthetic tensors and temporary synthetic byte payloads
- No patient data, external dataset, learned checkpoint, or GPU kernel was used.

## Verification results

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 44 passed, 26 subtests passed |
| `.venv/bin/python -m unittest discover -s tests -v` | 44 passed |
| `.venv/bin/python -m compileall -q src tests` | passed |
| `.venv/bin/python -m pip check` | no broken requirements |
| `git diff --check` | passed |

## Scientific boundary

This record verifies the legal observation boundary, patient-space coordinate
contracts, Gaussian tensor validation, differentiable thin-plane renderer, and
sampled finite-slab/PSF reference on synthetic inputs. It is not evidence for
reconstruction quality, clinical validity, scanner-side acquisition, dataset
legality, cross-site generalization, or the CVPR headline claim. Those require
separate immutable experiment manifests, patient-level splits, baselines,
statistics, and external evaluation.
