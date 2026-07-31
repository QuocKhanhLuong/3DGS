# T1-C Legal Episodic Trainer Handoff

Status: `IMPLEMENTED SOFTWARE TRANCHE — HUMAN GATE PENDING`

T1-C connects the existing T0.5 legality contracts and T1-B fixed-topology
evidence path into one legal context-to-target optimizer step. It is software
contract evidence, not a reconstruction-accuracy result.

## Executable flow

```text
SparseAvailabilityManifest
→ deterministic EpisodeAssignment
→ open and decode every context payload
→ fit context-only normalization
→ E0/E1/E2 encode and exact cache
→ deterministic fixed supports
→ common fixed Gaussian head
→ gauge-safe FrozenPatientState
→ expose target geometry and commit target
→ EpisodeController.render_and_register
→ reveal and decode target
→ target-valid ∩ renderer-supported loss
→ backward and optimizer step
```

The target tensor exists only in the post-reveal `LegalEpisodeStep`. Target
pixels and target-derived features never enter preprocessing fitting, the
feature cache, support selection, Gaussian construction, or frozen state.

## Implemented ownership

- `smagm.data`: byte-only NumPy decoding, context-only normalization,
  registration records, deterministic assignment schedules;
- `smagm.losses`: typed intensity/gradient/frequency reconstruction results and
  explicit objective composition;
- `smagm.training`: legal episode orchestration, matched schedules, gradient
  health, precision/accumulation policy, checkpoints, metrics, and provenance;
- `smagm.cli.train`: bounded CPU synthetic E0/E1/E2 diagnostic.

## Commands

```text
PYTHONPATH=src python -m smagm.cli.train --help
PYTHONPATH=src python -m smagm.cli.train --variant e0
PYTHONPATH=src python -m smagm.cli.train --variant e1
PYTHONPATH=src python -m smagm.cli.train --variant e2
python scripts/train.py --variant e2
python -m pytest -q tests/data/test_t1c_data.py tests/losses/test_reconstruction_losses.py tests/training --tb=short
python scripts/check_phase.py T1C
```

`--allow-dirty` is development-only. Exact gate evidence must run on a clean
commit and records commit, dirty state, config, manifest, split, assignment
schedule, seed, environment, checkpoint, receipt, state, and audit hashes.

## Non-claims and boundary

Synthetic execution and finite gradients do not establish reconstruction
quality, real-MRI training success, lesion fidelity, clinical validity,
calibrated uncertainty, or superiority of E2. T1-F, T1-R, and T1-M evidence
remain absent. T2 and later phases remain blocked. No anchor, field,
propagation, topology, routing, or full-volume package is introduced.
