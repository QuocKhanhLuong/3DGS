# T1-B Teacher-Free Encoder Handoff

T1-B is an executable software tranche for comparing three compact evidence
paths behind the same fixed-support and Gaussian-head contract. It is not a
reconstruction-accuracy result.

## Scope and variants

- E0 uses the differentiable analytic bank and a fixed, measurable adapter to
  the common output dimensions.
- E1 uses a shallow raw-image micro-CNN.
- E2 uses the analytic channels as input to the same class of shallow
  micro-CNN.

All variants emit `Z_str` with 16 channels, `Z_app` with 8 channels, one
reliability channel in `[0, 1]`, a tuple of one bound
`FeatureGridToPlaneTransform` per batch item, and a boolean valid-feature mask.
The channel counts and output stride are configurable; the default stride is 1.
Strides 1, 2, and 4 use the locked `align_corners=False` half-pixel mapping.
Odd shapes are right/bottom padded explicitly and padded feature centres remain
invalid.

The Gaussian head consumes canonical-RAS lower factors. Raw local covariance
factors are converted through the selected support basis, symmetrized, and
re-factored with the typed `covariance_epsilon` policy before runtime use.

Reliability may modulate amplitudes or loss weights, but it never selects a
support. Support indices are deterministic, value-independent, and shared
across E0/E1/E2. There is no support-to-support communication, learned birth,
anchor, propagation, routing, or full-volume export in this tranche.

## Teacher-free objectives

`smagm.losses.structural` provides masked structural consistency under declared
intensity perturbations, appearance sensitivity regularization, bounded
reliability regularization, a structural variance-floor diagnostic, and an
optional registered cross-modality structural comparison. Empty legal masks
raise `EmptyComparisonError`; unregistered cross-modality pairs are rejected.
No teacher, pseudo-label, segmentation label, or target pixel is used by these
objectives.

## Cache contract

`FeatureCacheKey` binds observation ID, canonical source-plane hash, encoder
variant/configuration/state hashes, preprocessing hash, the bound feature-grid
transform, the valid-feature-mask hash, dtype, and output channel contract.
Retrieval is exact-key only. Target-derived insertion is rejected.

## Synthetic checks

```text
PYTHONPATH=src python -m smagm.cli.t1b --help
PYTHONPATH=src python -m smagm.cli.t1b --variant e0
PYTHONPATH=src python -m smagm.cli.t1b --variant e1
PYTHONPATH=src python -m smagm.cli.t1b --variant e2
```

The CLI reports finite tensor shapes, reliability bounds, deterministic support
count, renderer coverage, loss, parameter counts, and gradient norms. These
reports are software-contract diagnostics only. They do not claim clinical
validity, pathology recovery, real-MRI training success, reconstruction
accuracy, or scientific superiority.

## Phase boundary

T1-A is implemented as an executable software contract. T1-B is in
development. T1-C and T2+ remain blocked. This handoff does not add scaffolding
for those later phases.
