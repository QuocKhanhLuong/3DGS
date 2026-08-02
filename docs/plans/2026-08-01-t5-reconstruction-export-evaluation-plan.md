# T5 Implementation Plan — Reconstruction, Export, and Isolated Evaluation

Date: 2026-08-01  
Status: **IMPLEMENTATION AUTHORIZED — SCIENTIFIC VALIDATION DEFERRED**  
Depends on: executable static patient state from T1-C, T2, and T3

Product alignment (2026-08-02): BraTS21 execution is a simulated
sparse-acquisition task. The renderer queries physical target planes/full grids
and serializes intensity, support, unsupportedness, and uncertainty. The
isolated evaluator uses serialized prediction packages and immutable target
packages, explicit normalized data range, global SSIM policy, and no
zero-filling of unsupported pixels. ROI and uncertainty analyses remain
evaluator-only and are skipped when their semantics are unavailable.

## 1. Purpose

T5 converts a complete static patient-specific Gaussian state into reproducible
2D and 3D outputs, preserves physical metadata, and evaluates serialized
predictions without allowing dense targets or labels to flow back into training,
state construction, checkpoint selection, or model adaptation.

The T5 flow is:

```text
frozen patient-specific state
→ arbitrary physical-plane reconstruction
→ chunked full-grid reconstruction
→ support and uncertainty diagnostics
→ immutable reconstruction package
→ physical-affine-preserving export
→ isolated target/label loading
→ patient-level metrics and statistics
→ consolidated evidence report
```

## 2. Stable package ownership

Implement or complete:

```text
src/smagm/reconstruction/
├── plane.py
├── volume.py
├── field.py
├── uncertainty.py
├── package.py
└── export.py

src/smagm/evaluation/
├── metrics.py
├── medical_fidelity.py
├── budget.py
├── uncertainty.py
├── audit.py
└── statistics.py

src/smagm/contracts/
└── outputs.py

src/smagm/cli/
├── reconstruct.py
├── evaluate.py
└── audit.py

scripts/
├── reconstruct.py
├── evaluate.py
└── audit.py
```

Evaluation code may consume only serialized outputs and declared audit inputs.
It must not receive a mutable patient state or optimizer.

## 3. Typed output contracts

Define immutable records for:

### PlaneReconstruction

- patient ID;
- modality;
- physical plane and plane hash;
- intensity prediction;
- support mass;
- unsupported mask;
- support uncertainty/reliability diagnostic;
- renderer version and config hash;
- patient-state version;
- artifact hash.

### VolumeReconstruction

- patient ID;
- modality;
- canonical output grid;
- affine/orientation metadata;
- intensity volume;
- support volume;
- unsupported volume;
- uncertainty/reliability volume;
- chunking configuration;
- state/config/artifact hashes.

### ReconstructionPackage

- exact repository commit;
- resolved config hash;
- manifest, split, assignment, and patient-state hashes;
- encoder, field, Gaussian, and propagation identities;
- modality mapping;
- requested planes and grids;
- output artifact inventory and digests;
- execution status;
- non-claims;
- runtime and hardware accounting.

## 4. Plane reconstruction

Plane reconstruction must:

- accept a declared physical plane in canonical RAS millimetres;
- call the pure renderer through a stable reconstruction interface;
- choose modality appearance channels through the resolved mapping;
- preserve unsupported output explicitly;
- never fill unsupported pixels with zero and call them confident predictions;
- return support and uncertainty diagnostics separately from intensity;
- remain deterministic for the same state, plane, config, dtype, and device.

Plane reconstruction is used for legal sparse targets, cross-validation planes,
and diagnostic arbitrary-plane queries.

## 5. Full-grid reconstruction

Full-grid reconstruction must be chunked and spatially culled.

Requirements:

- the output grid is declared before execution;
- physical affine, orientation, spacing, and shape are explicit;
- chunk boundaries do not alter values beyond declared numeric tolerance;
- spatial indexing evaluates only nearby Gaussian/anchor support;
- modality channels are reconstructed independently under the same geometry;
- support and unsupported masks are reconstructed with intensity;
- memory use and chunk runtime are logged;
- no dense ground-truth volume is needed to generate predictions.

The reference output is a canonical RAS-oriented NIfTI-compatible grid while
retaining source geometry provenance.

## 6. Structural field output

The reconstruction package may export:

- StructuralField values;
- field support weights;
- gradients and normals where finite and supported;
- optional zero-level or threshold surfaces;
- anchor and Gaussian locations for diagnostics.

Do not name exported values `SDF` unless the final evidence establishes sign,
distance calibration, Eikonal behavior, and gradient-norm validity.

## 7. Uncertainty and support terminology

Before calibration, expose:

- support mass;
- unsupported mask;
- nearest-observation distance;
- modality observability;
- field disagreement;
- propagation depth;
- reliability or support-uncertainty diagnostics.

A calibrated predictive uncertainty claim requires a declared calibration
cohort, target, proper score or coverage analysis, and a frozen calibration
mapping. Raw support mass is not calibrated uncertainty.

## 8. Export

Export formats:

```text
NIfTI  — intensity, support, unsupported mask, optional diagnostics
JSON   — provenance, configs, metrics, state and artifact hashes
PT/NPZ — tensors for reproducible analysis
PLY    — optional anchor/Gaussian diagnostics only
```

Requirements:

- preserve affine and orientation;
- preserve modality identity;
- use atomic writes;
- hash every artifact;
- prevent overwrite unless explicitly requested;
- record software/schema versions;
- export predictions before loading audit targets.

## 9. Isolated evaluation barrier

Evaluation order is mandatory:

```text
freeze model and configs
→ reconstruct and serialize predictions
→ close reconstruction process
→ open evaluator with immutable prediction package
→ load declared targets/labels
→ compute metrics
→ write evaluation report
```

The evaluator must reject:

- mutable patient state;
- model parameters requiring gradients;
- optimizer or trainer objects;
- prediction packages without state/config hashes;
- targets whose patient/split/grid identity mismatches the prediction;
- sealed audit data before the declared freeze record exists.

## 10. Reconstruction metrics

Report at patient level and then aggregate.

### Global intensity

- MAE;
- RMSE;
- NMSE;
- PSNR;
- SSIM;
- NCC.

### Structure and frequency

- gradient MAE/NCC;
- edge agreement;
- local contrast error;
- frequency-band error;
- boundary distance where an admissible reference exists.

### Coverage-aware reporting

Every metric must be accompanied by:

- supported fraction;
- unsupported fraction;
- metric on all declared evaluable voxels;
- metric stratified by support/reliability bins;
- failure count and reason;
- no silent exclusion of difficult or unsupported regions.

## 11. Medical-fidelity evaluation

Where patient-disjoint labels or ROIs exist, report:

- lesion/ROI intensity fidelity;
- lesion/ROI contrast-to-background;
- boundary error;
- small-structure retention;
- frozen downstream-model consistency when predeclared;
- coverage inside and around the ROI;
- failure cases.

T1 development validation and the sealed T5 final audit must remain distinct.
The final-audit cohort cannot guide architecture, threshold, checkpoint, or
analysis-plan changes after it is opened.

## 12. Uncertainty evaluation

When calibration is implemented, report:

- error versus uncertainty correlation;
- calibration curves;
- proper scores where applicable;
- interval or set coverage;
- coverage-risk curves;
- selective prediction curves;
- error stratified by propagation depth, support, modality observability, and
  distance to observed planes.

If calibration is not implemented, report only diagnostic association and avoid
calibration claims.

## 13. Compute and complexity accounting

For every model variant, record:

- trainable parameters;
- encoder FLOPs and runtime;
- anchor count;
- field-query count;
- structural and volumetric primitive counts;
- propagation rounds and accepted operations;
- cache bytes;
- peak CPU/GPU memory;
- target-plane render latency;
- full-volume reconstruction latency;
- artifact size;
- hardware and software environment.

Matched comparisons must disclose unequal opportunity rather than hiding it.

## 14. Patient-level statistics

Use patient-level paired analysis for all primary comparisons.

Required outputs:

- per-patient metric table;
- mean/median and dispersion;
- paired differences;
- confidence intervals from a predeclared procedure;
- seed aggregation;
- multiplicity policy for multiple primary endpoints;
- equivalence/non-inferiority margins where used;
- missing/failure handling;
- subgroup summaries only when predeclared and sufficiently supported.

Do not claim significance or equivalence from overlapping point estimates.

## 15. Required tests

### Reconstruction

- arbitrary-plane geometry correctness;
- chunked versus unchunked equivalence;
- spatial-culling equivalence;
- modality mapping;
- finite output;
- explicit unsupported behavior;
- deterministic artifact hashes.

### Export

- affine/orientation preservation;
- atomic write behavior;
- no accidental overwrite;
- schema validation;
- round-trip tensor/NIfTI consistency within tolerance.

### Evaluation isolation

- mutable patient state rejection;
- no model/trainer imports in isolated evaluator path;
- prediction-before-target ordering;
- patient/split/grid mismatch rejection;
- sealed-audit freeze-record requirement.

### Metrics

- analytic identity and known-error cases;
- coverage-aware denominators;
- unsupported-region handling;
- patient-level aggregation;
- deterministic confidence intervals under fixed seed.

### Scope

- no active routing;
- no audit target access in reconstruction/training packages;
- no evaluation feedback into mutable state.

## 16. CLI and artifact layout

Reconstruction:

```bash
python scripts/reconstruct.py \
  --config configs/experiments/full_static_pipeline.json \
  --checkpoint artifacts/model.pt \
  --manifest data/eval_manifest.json \
  --output-dir experiments/reconstructions/full-static
```

Evaluation:

```bash
python scripts/evaluate.py \
  --plan configs/evaluation/full_static_eval.json \
  --predictions experiments/reconstructions/full-static \
  --output-dir experiments/reports/full-static
```

Audit:

```bash
python scripts/audit.py \
  --package experiments/reconstructions/full-static \
  --report experiments/reports/full-static
```

Each command must provide `--help`, validate configuration, write resolved
configs and provenance, and fail closed on identity mismatch.

## 17. Completion condition

T5 software is complete when:

1. a frozen patient state reconstructs declared planes and a full physical grid;
2. outputs preserve affine/orientation and unsupported diagnostics;
3. an immutable package is serialized before target loading;
4. isolated evaluation produces patient-level metrics and statistics;
5. all artifacts bind to exact commit/config/manifest/checkpoint/state hashes;
6. the complete static pipeline remains switchable into required ablations;
7. T4 routing remains absent;
8. one consolidated evaluation report can be reproduced from clean checkout.

Software completion is not scientific approval. The owner performs the final
review using the consolidated evaluation plan.
