# Consolidated Static Pipeline Evaluation Plan — 2026-08-01

Status: **PLANNED FINAL EVALUATION — PRODUCT EXECUTION BLOCKED BY CUDA ENVIRONMENT**

## Product protocol clarification — 2026-08-02

The first real-data execution is BraTS21 as a simulated sparse-acquisition
task: five aligned physical axial context positions per modality, one
strictly-interior gap target, context-only robust percentile normalization,
E2/R4 with the retained P1 bounded propagation path for the full run. P0
remains a code-level regression switch, not a product launch stage. Metrics use an explicit normalized data
range and global SSIM policy. The inventory is complete, but no CUDA full-run,
checkpoint/resume, or online W&B result may be reported until the host
exposes a working NVIDIA driver. This document remains an evaluation plan, not
a scientific pass.

## 1. Objective

Run one final evaluation suite after the complete static pipeline is executable.
The suite replaces intermediate scientific stop-and-review cycles, but it does
not remove software tests or matched attribution requirements.

The evaluation must answer five questions:

1. Does legal teacher-free evidence outperform simpler evidence paths?
2. Do physical anchors add value beyond deterministic or free Gaussian support?
3. Does the shared anchor-local StructuralField add value beyond direct
   anchor-to-Gaussian mapping or a global field?
4. Does bounded propagation improve over the static seed state under matched
   primitive, compute, and optimization opportunity?
5. Does the final method improve full-volume and medically relevant fidelity
   without hiding unsupported regions or leakage?

## 2. Evaluation unit

The primary unit is the patient. Splits must be patient-disjoint.

Each evaluated patient has:

- a declared permanently sparse input manifest;
- a fixed context/target or reconstruction input schedule;
- physical plane metadata;
- modality identities;
- a serialized prediction package;
- an evaluator-isolated dense target or admissible sparse target set;
- optional lesion/ROI/boundary labels;
- no target or audit pixel access by training or state construction.

## 3. Cohorts

### Training cohort

Used for global model optimization under permanently sparse manifests.

### Development validation cohort

Patient-disjoint. May be used for declared checkpoint-selection and architecture
decisions before the final freeze. Every access and resulting decision is
logged.

### Sealed final-audit cohort

Patient-disjoint and unopened until:

- architecture is frozen;
- variants and baselines are frozen;
- checkpoints and checkpoint-selection rules are frozen;
- preprocessing and modality mappings are frozen;
- evaluation metrics and margins are frozen;
- statistical analysis is frozen;
- prediction serialization is complete.

## 4. Mandatory variants

Use one config composition system so only the named causal component changes.

### Evidence attribution

```text
E0 — analytic differential evidence only
E1 — raw-image shallow CNN
E2 — analytic scaffold + teacher-free micro-CNN
```

### Representation attribution

```text
R0 — sparse interpolation floor
R1 — deterministic fixed-support Gaussian
R2 — free Gaussian without anchors or StructuralField
R3 — direct physical anchor-to-Gaussian mapping, no field
R4 — anchor + shared StructuralField + seed Gaussians
R5 — anchor + global coordinate field + seed Gaussians
```

### Propagation attribution

```text
P0 — no propagation; T2 seed state
P1 — bounded fixed propagation
P2 — propagation with conservative move/birth, when implemented
P3 — propagation with bounded adaptive topology, when implemented
```

### Full method

```text
FULL — E2 + R4 + the selected propagation variant + T5 reconstruction
```

Optional privileged upper bounds must be clearly separated:

```text
U0 — frozen pretrained encoder
U1 — teacher-distilled encoder
U2 — dense or complete-volume supervision
```

Privileged variants cannot be reported as the main method.

## 5. Matching rules

Within each attribution group, hold constant wherever scientifically meaningful:

- patient split and manifests;
- context/target assignment schedule;
- modality policy and normalization;
- physical output planes and volume grid;
- renderer and amplitude gauge;
- support or primitive budget;
- trainable parameter budget or declared difference;
- optimizer and optimizer-step opportunity;
- random seeds;
- checkpoint-selection rule;
- evaluation mask;
- compute-accounting procedure;
- hardware and precision;
- preprocessing and registration inputs.

When exact matching is impossible, report the mismatch and add a complexity or
compute-normalized comparison.

## 6. Primary comparisons

The minimum causal comparisons are:

1. `E2 - E0`: value of learned evidence beyond analytic features.
2. `E2 - E1`: value of the analytic scaffold beyond a raw shallow CNN.
3. `R1 - R0`: whether Gaussian reconstruction beats interpolation.
4. `R3 - R1`: whether physical anchors add value over fixed supports.
5. `R4 - R3`: value of the shared StructuralField.
6. `R4 - R2`: value of anchor/field constraints over free Gaussians.
7. `R4 - R5`: local shared field versus global coordinate field.
8. `P1 - P0`: value of bounded propagation.
9. `FULL - R0`: end-to-end gain over the interpolation floor.

Optional topology comparisons are secondary until P1 is stable.

## 7. Reconstruction endpoints

Primary global endpoints:

- NMSE;
- PSNR;
- SSIM;
- gradient error;
- supported fraction;
- unsupported fraction.

Primary medical-fidelity endpoints where labels exist:

- lesion/ROI intensity fidelity;
- lesion/ROI contrast fidelity;
- boundary distance or boundary agreement;
- small-structure retention;
- ROI support/coverage.

Secondary endpoints:

- MAE/RMSE;
- NCC;
- local contrast error;
- frequency-band error;
- edge agreement;
- failure rate;
- calibration or support-risk diagnostics;
- runtime, memory, cache, and primitive count.

## 8. Coverage and unsupported regions

No method may improve its metric by silently refusing difficult regions.

For every endpoint, report:

- full declared evaluable region;
- supported region;
- unsupported fraction;
- error stratified by support/reliability bin;
- error versus distance to observed planes;
- error versus propagation depth;
- failure count and reason.

Unsupported pixels/voxels are never converted into confident zero-valued
predictions for scoring.

## 9. Leakage audit

The final evidence package must include:

- immutable sparse manifests and hashes;
- opened-file ledgers;
- context/target event order;
- prediction receipt records;
- zero non-manifest training access;
- zero target-before-render access;
- zero sealed-audit access before freeze;
- negative controls that intentionally attempt prohibited access and fail;
- exact repository commit and dirty-state report.

Any leakage invalidates affected runs and blocks scientific claims.

## 10. Geometry audit

Verify:

- canonical RAS-mm coordinates;
- source-affine provenance;
- plane and grid hashes;
- feature-to-plane transforms;
- anisotropic spacing;
- rotated/oblique plane handling;
- full-volume affine/orientation preservation;
- chunked reconstruction equivalence;
- physical NMS and propagation step sizes.

## 11. Representation diagnostics

Report:

- anchor count and physical distribution;
- contributing planes and modalities per anchor;
- field support and overlap;
- field value/gradient statistics;
- structural versus volumetric primitive counts;
- covariance scale/orientation distributions;
- primitive-to-anchor and parent provenance;
- propagation depth and accepted/rejected proposals;
- uncertainty/reliability diagnostics;
- state size and serialization cost.

These diagnostics explain behavior but do not replace reconstruction metrics.

## 12. Statistical analysis

Use patient-level paired analysis.

Before opening the final audit, freeze:

- primary comparisons;
- primary endpoints;
- direction of benefit;
- confidence-interval procedure;
- equivalence/non-inferiority margins where applicable;
- multiplicity policy;
- missing/failure handling;
- number of seeds and seed aggregation;
- subgroup definitions.

Report point estimates, paired differences, confidence intervals, and full
per-patient tables. Do not infer equivalence from a non-significant difference.

## 13. Decision rules

### Evidence encoder

- If E2 does not improve over E0, remove or demote the learned encoder claim.
- If E2 does not improve over E1, remove or demote the analytic-scaffold claim.

### Anchor and field

- If R1 does not beat R0, repair the basic reconstruction setup before claiming
  value from anchors or propagation.
- If R3 does not improve over R1, demote anchor selection as a contribution.
- If R4 does not improve over R3 and R2 under matched conditions, remove or
  simplify the StructuralField claim.
- If R5 matches or beats R4 with lower complexity, demote the local-field claim.

### Propagation

- If P1 does not improve over P0 under matched primitive/compute opportunity,
  remove propagation from the main claim.
- If topology variants improve only by increasing primitive or compute
  opportunity, report them as unmatched complexity variants rather than causal
  wins.

### Medical fidelity

- If global metrics improve while lesion/ROI or boundary fidelity regresses
  beyond the predeclared margin, fail the medical-fidelity claim.

## 14. Final report layout

```text
00_run_inventory.json
01_legality_audit.md
02_geometry_audit.md
03_evidence_attribution.csv
04_representation_attribution.csv
05_propagation_attribution.csv
06_full_volume_metrics.csv
07_medical_fidelity.csv
08_coverage_uncertainty.csv
09_compute_complexity.csv
10_patient_level_statistics.md
11_failure_cases.md
12_claim_decision_matrix.md
artifacts/
configs/
manifests/
checkpoints/
predictions/
```

Every table row must bind to run ID, commit, resolved config, manifest,
checkpoint, prediction package, patient split, seed, and hardware record.

## 15. Final owner review

After the report is complete, the owner records one explicit decision for each
claim:

- PASS;
- PASS_WITH_CONDITIONS;
- REWORK;
- FAIL;
- DEMOTE_TO_ABLATION.

No automated runner or implementation agent may write these scientific
verdicts.
