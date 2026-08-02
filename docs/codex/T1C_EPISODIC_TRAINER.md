# T1-C Legal Episodic Trainer Handoff

Status: `IMPLEMENTED CANDIDATE — IN REVIEW`

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
python scripts/train.py --config configs/experiments/t1c_synthetic.json --variant e0 --steps 2 --output-dir /tmp/smagm-t1c-e0
python scripts/train.py --config configs/experiments/t1c_synthetic.json --variant e1 --steps 2 --output-dir /tmp/smagm-t1c-e1
python scripts/train.py --config configs/experiments/t1c_synthetic.json --variant e2 --steps 2 --output-dir /tmp/smagm-t1c-e2
python -m pytest -q tests/data/test_t1c_data.py tests/losses/test_reconstruction_losses.py tests/training --tb=short
python scripts/check_phase.py T1C
```

The command rejects a dirty checkout so a development run cannot be presented
as final evidence. Exact gate evidence must run on a clean commit and records
resolved config, manifest, split, assignment schedule, modality mapping,
preprocessing, encoder/head, renderer/gauge, checkpoint, receipt, state,
environment, opened-file ledger, and artifact hashes. A resumable checkpoint is
accepted only when its immutable run identity, manifest, split registry, and
scheduled assignment hashes match; it is never written mid-accumulation.

When `--output-dir` is supplied, the run writes the declared resolved config,
provenance, metrics, summary, checkpoint, serialized episode ledger, and an
artifact-digest manifest. These artifacts document software execution only.

## Non-claims and boundary

Synthetic execution and finite gradients do not establish reconstruction
quality, real-MRI training success, lesion fidelity, clinical validity,
calibrated uncertainty, or superiority of E2. T1-F, T1-R, and T1-M evidence
remain absent. T2, bounded-static T3 P0/P1, and T5 are now separately
authorized active implementation candidates under the fast-track decision. T4
routing remains absent and blocked. This T1-C handoff still owns only the
legal context-to-target episode path; it does not make a scientific claim.
