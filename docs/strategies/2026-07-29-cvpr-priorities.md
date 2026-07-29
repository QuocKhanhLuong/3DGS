# CVPR Priority Strategy — Sparse Active Multi-Sequence Reconstruction

> **Historical, non-authoritative strategy.** For current work, follow
> [`2026-07-29-isbi-realignment.md`](2026-07-29-isbi-realignment.md). Where the
> two documents conflict, the ISBI realignment controls; in particular, this
> document's active-policy-first thesis is not the current research direction.

Date: 2026-07-29  
Role: AgenTeam PM  
Decision horizon: next implementation tranche through CVPR submission readiness

## Executive decision

The project should serve researchers and clinical-imaging teams who need to reconstruct a useful patient-specific multi-sequence volume under a limited observation budget while knowing where the result is unsupported.

The CVPR thesis should be narrowed to:

> **Leakage-safe, closed-loop selection of registered multi-sequence planes using calibrated marginal reconstruction gain over a PSF-correct Gaussian patient state.**

The representation is enabling infrastructure, not the headline novelty. Slice-aware Gaussian rendering, SDF–Gaussian coupling, anchor-local fields, teacher-free sparse reconstruction, and generic information-gain routing all have close recent prior art. The remaining defensible contribution is their rigorously controlled use in a legal active-observation protocol that improves the complete quality–budget curve, generalizes across cohorts, and preserves lesion-sensitive fidelity.

The smallest scientifically coherent implementation tranche is **T0 — Legal Physical Forward Operator**. T0 should include both:

1. a closed-form infinitesimally thin-plane reference; and
2. a slow sampled finite-slab renderer with a configurable normalized through-plane PSF.

T0 should **not** stop at thin-plane rendering. GaussianPile and GSVR make PSF/thickness fidelity a baseline expectation, while the project documents already expose thickness in the plane contract. A sampled PSF/slab oracle is the minimum needed to determine when the thin approximation is valid and to compare future optimized kernels. T0 does not need an analytic finite-slab formula, measured scanner-PSF estimation, motion correction, or CUDA acceleration.

## Problem

### User problem

Current reconstruction plans ask a future user to trust three things that are not yet executable:

- a hidden candidate image was never inspected before its action was committed;
- every observation was rendered in the correct patient-space geometry and slice-formation model;
- active selection improves clinically relevant reconstruction rather than merely producing smoother images or exploiting a favorable budget.

Without those guarantees, a visually plausible result cannot support a CVPR-level active-reconstruction claim.

### Product and paper problem

The current four-phase design is too broad to attribute scientifically. It combines:

- a teacher-free analytic-plus-micro-CNN encoder;
- local structural fields and anchor-organized Gaussians;
- dual structural/volumetric Gaussian banks;
- adaptive topology;
- uncertainty;
- balanced multi-wave routing;
- learned utility and stopping;
- multi-sequence full-volume reconstruction.

Recent work collides with nearly every component-level novelty claim. Building the entire design before testing the remaining policy-level gap would maximize engineering effort while leaving the paper thesis vulnerable.

## Evidence base and precedence

This strategy uses only the authoritative `docs/` tree:

- [`../research/2026-07-29-cvpr-external-signals.md`](../research/2026-07-29-cvpr-external-signals.md)
- [`../designs/2026-07-29-cvpr-internal-health.md`](../designs/2026-07-29-cvpr-internal-health.md)
- every specification under [`../reconstruction/`](../reconstruction/)

There was no prior file under `docs/strategies/`. The locked reconstruction direction is therefore treated as the current strategy, with `PROOFREAD_NOTES.md` taking precedence where its implementation order differs from older phase prose. Deprecated root documents are excluded.

## External collision × internal blocker map

| External signal | Internal blocker or strategy conflict | Product decision |
|---|---|---|
| GaussianPile and GSVR already cover slice-aware Gaussians, finite thickness, and PSF-aware rendering. | No executable observation operator exists; thin-plane mathematics and amplitude normalization remain unsettled. | Treat rendering as a validity contract and matched baseline substrate, not novelty. Put thin and sampled PSF/slab references in T0. |
| GauSS-MI and ActiveGAMER already use Gaussian uncertainty/information gain for active planning. | Current uncertainty is a collection of uncalibrated heuristics; learned routing lacks a legal counterfactual target source. | Require an oracle-headroom gate, calibrated marginal gain, and matched routing baselines before multi-wave or learned routing. |
| GSDF, SurfaceSplat, GaussianUDF, Scaffold-GS, and 3D-SLNR collide with SDF hybrids, anchors, and local tiny fields. | “SDF” behavior is unverified; the design has incompatible covariance variants and excessive representation scope. | Use `StructuralField` until signed-distance tests pass. Keep SDF constraints, dual banks, and adaptive topology as causal ablations, not the lead claim. |
| GaussianSVR, MedGS, 3DGSMR, and related medical-Gaussian work remove broad “first” and teacher-free claims. | The main sparse-supervision claim is not protected by an executable manifest/commit/reveal barrier or isolated audit process. | Make leakage safety and patient-level audit provenance non-negotiable gates, but do not claim they alone are algorithmic novelty. |
| Scanner-side active MRI selects k-space; the current action is a registered `sequence × plane` reveal. | The current documents can be read as claiming scan acceleration without a scanner-valid acquisition model. | Develop first as **retrospective budgeted progressive observation** on registered multi-sequence volumes. Claim active plane/stack acquisition only after a real prescription protocol is documented. |

## Proposal

Build a narrow evidence ladder in which each tranche answers one decision before unlocking the next:

```text
Claim and action contract
→ legal physical forward operator
→ matched static reconstruction baselines
→ oracle policy headroom
→ analytic/calibrated active policy
→ external and lesion-sensitive validation
```

The team should stop or re-scope whenever a gate shows that the next layer cannot add a meaningful, attributable result.

## Top five prioritized recommendations

### 1. Freeze the paper claim, action semantics, and data legality contract

**Priority:** P0 — blocking  
**Effort:** Small to medium  
**Who benefits:** Reviewers can evaluate one falsifiable contribution; implementers avoid building toward an invalid scanner-acquisition claim; medical-data owners get an auditable boundary.

**Rationale**

The strongest remaining gap is active multi-sequence observation, but the current `sequence × slice` action is a retrospective reveal unless a scanner or reacquisition protocol says otherwise. At the same time, candidate-image leakage and dense-audit contamination are critical validity risks.

**Requirements**

- Lock the primary wording to **retrospective budgeted progressive observation** for the initial BraTS-style track.
- Define the action as a registered `(modality, physical plane)` with explicit cost, thickness, orientation, and availability.
- Define patient-level train/validation/test partitions; keep longitudinal time points in the same partition.
- Use permanently sparse main-training manifests and an immutable commit/reveal ledger.
- Isolate fully sampled audit volumes in a separate process that receives only serialized reconstructions.
- Choose a controlled development cohort and one external-site test-only cohort; the current evidence supports BraTS plus a TCIA cohort, subject to final license and access verification.
- Pre-register the budget grid, primary quality–budget metric, lesion-sensitive metrics, and failure statuses.

**Success criteria**

- One approved one-sentence thesis and one task name are used consistently in plans, experiments, and paper claims.
- A data-provenance sheet records dataset license, access constraints, patient grouping, split hashes, sparse-manifest construction, and audit isolation.
- A leakage-positive control is specified: it must fail when forbidden candidate or audit pixels are intentionally exposed.
- No experiment can start without an action-cost definition and manifest hash.

**Explicitly out of scope**

- scanner-side k-space acquisition claims;
- new clinical acquisition or patient recruitment;
- a new dataset contribution;
- slice-level random splitting;
- clinical deployment or diagnostic claims.

### 2. Implement T0 — Legal Physical Forward Operator

**Priority:** P0 — first implementation tranche  
**Effort:** Medium  
**Who benefits:** Every later representation and routing experiment gets a trusted legality, geometry, and differentiability substrate.

**Rationale**

The renderer is not paper novelty, but it is the correctness oracle for all later results. A thin-only implementation cannot determine whether observed gains come from ignoring slice thickness or from the proposed representation. A production-grade PSF kernel is unnecessary now; a slow independent reference is sufficient.

**Must-have scope**

- immutable sparse observation, manifest, commit/reveal, and deterministic file-open ledger contracts;
- canonical RAS millimeters, pixel-center plane semantics, `[v,u]` tensor mapping, source-convention provenance, and independent signed-affine validation;
- explicit `PhysicalPlane`, `TargetGrid`, observation batch, and general-SPD `GaussianBatch` contracts;
- normalized additive MRI composition returning intensity, support mass, and `unsupported_mask`;
- closed-form thin-plane Gaussian reference;
- sampled finite-slab reference with a normalized configurable 1D PSF along the plane normal;
- synthetic analytic, affine-equivariance, slab-convergence, leakage, chunking, and float64 gradient checks;
- CPU-first deterministic package and environment lock.

**PSF decision**

T0 includes **reference PSF/finite-slab support**, not only a thin plane. The minimum implementation may use deterministic quadrature and a small PSF interface supporting:

- a delta profile, reproducing the thin-plane limit;
- a box profile, representing uniform slab integration;
- a supplied normalized discrete profile for protocol-specific tests.

The thin path remains the analytic oracle and speed baseline. Any later analytic or accelerated slab path must match the sampled reference in forward values and gradients.

**Success criteria**

- target pixels cannot be opened before a committed reveal token, and the positive-control leakage test fails as intended;
- LPS/RAS landmarks, source-affine agreement, signed normal, pixel-center, and equivalent-frame tests pass;
- the thin renderer matches an independent closed-form image under documented numerical tolerances;
- slab quadrature converges toward a stable result as samples increase and collapses to the thin result for a delta PSF;
- gradients pass `gradcheck` for center, covariance factor, support amplitude, and appearance;
- unsupported regions are explicit and never silently converted into confident zero intensity;
- all CPU reference tests are deterministic from a recorded seed.

**Explicitly out of scope**

- analytic Gaussian–PSF convolution;
- estimation of scanner-specific PSFs;
- slice motion, bias-field, or outlier modeling;
- NIfTI/DICOM adapters beyond the first chosen boundary format;
- spatial acceleration, custom CUDA, or `gsplat` integration;
- encoder, local field, routing, adaptive topology, or final export.

### 3. Establish a matched static reconstruction and baseline gate

**Priority:** P1 — required before routing  
**Effort:** Large  
**Who benefits:** Researchers learn whether the representation has any value beyond a correct forward operator; reviewers receive fair and attributable comparisons.

**Rationale**

The active policy cannot be evaluated if reconstruction quality changes with different forward models, primitive budgets, or hidden optimization budgets. The closest prior work makes a PSF-aware free Gaussian a mandatory baseline. SDF, anchors, and dual banks should earn their place through causal ablations.

**Requirements**

- Run deterministic interpolation floors: nearest, linear, cubic/B-spline, and an edge-aware variant where available.
- Implement or containerize a mature task-specific reference such as NeSVoR.
- Implement a matched PSF-aware free-Gaussian baseline using the T0 operator.
- Add a compact INR or dense-voxel baseline with matched parameter, wall-clock, and observation budgets.
- Treat GaussianPile/GSVR formulations as the closest forward-model references; label official executions and paper reimplementations separately.
- Compare free Gaussian, structural initialization only, persistent structural constraint, surface-only, volume-only, and dual-bank variants only after the free-Gaussian baseline is stable.
- Hold observations, PSF, target grid, normalization, initialization opportunity, optimizer budget, and hardware accounting constant.

**Success criteria**

- Every baseline consumes identical legal observations and the same declared slice-formation model.
- Static results report patient-level confidence intervals, failure counts, latency, peak VRAM, state size, and primitive counts.
- The proposed representation must beat the best matched static baseline on the predeclared primary metric without degrading lesion-sensitive fidelity; otherwise it remains an ablation and is removed from the headline claim.
- Reimplemented baselines pass synthetic forward-model parity checks before entering the leaderboard.

**Explicitly out of scope**

- routing, learned gain, stopping, and graph repair;
- adaptive birth/split/merge/prune;
- claiming `SDF` without sign, Eikonal, and distance-calibration evidence;
- a large appearance decoder or segmentation training head.

### 4. Prove policy headroom, then build the simplest calibrated active policy

**Priority:** P1 — paper differentiator  
**Effort:** Medium to large  
**Who benefits:** Users receive demonstrably better reconstructions per observation; the team avoids spending months on routing when uniform sampling is already near optimal.

**Rationale**

Generic uncertainty and Gaussian information gain are prior art. The project needs evidence that candidate ordering matters, that a legal predictor can rank true marginal full-volume gain, and that multi-sequence balance improves more than simple greedy policies.

**Requirements**

- Measure a retrospective one-step oracle on the isolated development audit split; use beam search only for small candidate sets.
- Compare random, uniform spacing, central-first, modality-balanced uniform, uncovered-mass, predictive-uncertainty, expected residual reduction, GauSS-MI-style mutual information, and ActiveGAMER-style rendering information gain.
- Start with an analytic predicted marginal-gain score from legal current-state descriptors.
- Calibrate predicted gain against post-commit audit improvement and report rank correlation and NDCG.
- Add learned gain only after its supervision source is declared legal.
- Add balanced multi-wave routing only after a single-wave or greedy marginal-gain policy demonstrates headroom and cross-modality imbalance.

**Success criteria**

- Before policy engineering, the retrospective oracle shows a pre-registered meaningful gap over the best simple heuristic; a suggested development gate is at least 5% relative improvement in normalized quality–budget AUC.
- The proposed legal policy beats uniform, random, uncertainty-greedy, and MI-style baselines on both the controlled and external tracks, with a paired patient-level 95% confidence interval excluding zero for the primary AUC metric.
- Actual-versus-predicted marginal gain has useful positive rank correlation and NDCG on held-out patients.
- The policy reduces redundancy and reaches 90%/95% of the full-observation reference with fewer or cheaper observations without increasing lesion-miss rate.
- If the oracle gap is small, routing is re-scoped to a negative result or removed; if a simple analytic policy closes the gap, learned and multi-wave routing are not built.

**Explicitly out of scope**

- reinforcement learning;
- receding-horizon planning before one-step ranking works;
- D*-style graph repair;
- adaptive topology as a routing prerequisite;
- claims of globally optimal dynamic routes.

### 5. Build the CVPR evidence package and reproducibility system

**Priority:** P1 — submission gate  
**Effort:** Large  
**Who benefits:** Reviewers can reproduce the result and assess clinical failure modes; future researchers can distinguish algorithmic gains from data, compute, or implementation effects.

**Rationale**

PSNR/SSIM on one registered cohort is insufficient. The lead claim requires external generalization, clinically sensitive fidelity, calibrated unsupported regions, full quality–budget curves, and transparent compute.

**Requirements**

- Controlled registered multi-sequence development track plus one external-site test-only clinical cohort.
- Per-modality NMSE, PSNR, SSIM/MS-SSIM, MAE, gradient/frequency error, arbitrary-plane consistency, and boundary/lesion-region error.
- Frozen external segmentation audit reporting whole-tumor, tumor-core, and enhancing-tumor Dice, HD95/surface Dice, lesion recall, and false-negative lesions.
- Calibration, sparsification/AUSE, risk–coverage, lesion-miss-by-uncertainty, and unsupported-volume metrics.
- Full quality–budget curves, normalized AUC, budget-to-target, modality allocation, redundancy, regret, and stopping/failure status.
- At least three seeds for learned variants, patient-level paired intervals, and a declared statistical test.
- Run artifacts containing git/dirty state, environment lock, resolved config, seeds, split and manifest hashes, opened-file ledger, context/target IDs, coordinate and normalization records, outputs, checkpoints, runtime, VRAM, and compute hardware.

**Success criteria**

- A clean environment can reproduce the primary table and quality–budget figure from immutable manifests.
- All headline claims map to a metric, baseline, ablation, and artifact hash.
- The external test remains untouched until the method and thresholds are frozen.
- Improvements persist on the external cohort and do not trade higher global PSNR for worse lesion recall or miscalibrated confidence.
- The final compute report includes preprocessing, per-query scoring, encoding, rendering, assimilation, and routing rather than GPU training time alone.

**Explicitly out of scope**

- training a new segmentation model;
- broad multi-dataset domain-generalization claims beyond one external cohort;
- clinical utility, outcome, or safety claims;
- release promises for data the team cannot legally redistribute;
- presentation UI or full-volume visualization tooling.

## Prioritization summary

| Rank | Recommendation | Priority | Effort | Unlocks |
|---:|---|---|---|---|
| 1 | Freeze claim, action, and data legality | P0 | Small–medium | Valid task and evidence contract |
| 2 | T0 legal physical forward operator with thin + sampled PSF/slab references | P0 | Medium | Trustworthy implementation substrate |
| 3 | Matched static reconstruction baseline gate | P1 | Large | Representation attribution |
| 4 | Oracle headroom and simplest calibrated active policy | P1 | Medium–large | Defensible paper differentiator |
| 5 | External, lesion-sensitive, calibrated, reproducible evidence package | P1 | Large | CVPR submission readiness |

## Must-have, nice-to-have, and out-of-scope portfolio

### Must-have for the primary paper

- honest retrospective-action wording unless a real acquisition protocol is supplied;
- immutable sparse manifests, commit/reveal, opened-file audit, and isolated dense audit;
- canonical physical geometry and thin plus sampled PSF/slab reference rendering;
- matched static baselines;
- oracle policy-headroom test;
- analytic or learned marginal-gain policy that beats simple and MI-style baselines;
- two-cohort evaluation, lesion-sensitive audit, uncertainty calibration, full quality–budget curves, and compute reporting.

### Nice-to-have only after their gates clear

- learned gain prediction after legal analytic routing;
- balanced multi-wave scheduling after imbalance is measured;
- structural-field constraints after free-Gaussian results;
- dual banks after surface-only and volume-only ablations;
- analytic finite-slab convolution after sampled-reference parity;
- a pretrained or teacher-distilled encoder as a clearly privileged upper bound.

### Out of scope for the first implementation and initial paper path

- custom CUDA or perspective-camera 3DGS rasterization;
- scanner-side k-space active MRI;
- adaptive Gaussian/anchor topology;
- D*-style local graph repair;
- reinforcement learning and long-horizon planning;
- unverified true-SDF claims;
- training segmentation heads;
- clinical deployment or prospective patient study;
- broad “first Gaussian medical reconstruction” or component-aggregation novelty claims.

## CVPR-appropriate team and gate structure

The team needs independent ownership of scientific validity, not only implementation modules. One person may hold more than one role in a small team, but gate approvers should not approve their own work alone.

### 1. Research and Novelty Lead

**Owns**

- living novelty/collision ledger;
- one-sentence thesis and related-work boundary;
- closest-baseline availability and reimplementation labels;
- claim-to-evidence matrix.

**Gate N0 — Claim viability**

Pass only when the action model, one primary claim, mandatory baselines, and no-go conditions are written. Reopen N0 when a new paper collides with the claim.

### 2. Medical Data Legality and Acquisition Lead

**Owns**

- dataset license and data-use record;
- acquisition realism and cost model;
- patient/longitudinal splitting;
- sparse-manifest construction;
- commit/reveal capability boundary;
- isolated audit protocol and leakage-positive controls.

**Gate D0 — Legal observation**

This role has veto authority. Pass only when unauthorized pixels are inaccessible by construction, split hashes are patient-safe, and the task wording matches the physical or retrospective action.

### 3. Geometry and Rendering Lead

**Owns**

- canonical RAS-mm and source-affine contracts;
- thin-plane derivation;
- sampled finite-slab and PSF reference;
- normalized additive composition;
- support/unsupported semantics;
- forward and gradient parity for future optimized kernels.

**Gate G0 — Physical operator**

Pass only when coordinate invariance, analytic thin-plane, slab convergence, gradient, and leakage-boundary tests pass. No encoder or representation result is scientific evidence before G0.

### 4. ML and Representation Lead

**Owns**

- E0/E1/E2 encoder comparisons;
- fixed-topology free-Gaussian baseline;
- structural-field and dual-bank ablations;
- gain predictor only after a legal target source exists;
- capacity and FLOP matching.

**Gate M1 — Attributable representation**

Pass only when the proposed component beats its matched simpler baseline under the same operator, observations, compute, and optimization opportunity. Otherwise remove or demote the component.

### 5. Experiments and Reproducibility Lead

**Owns**

- baseline containers and license records;
- immutable experiment manifests and configurations;
- run hashing, seeds, hardware and compute logs;
- patient-level statistics;
- controlled and external evaluation;
- leaderboard inclusion rules.

**Gate E1 — Static benchmark ready**

All baselines must pass geometry/PSF parity and consume identical legal observations.

**Gate E2 — Policy and external evidence**

Pass only when the oracle gap exists, the policy beats required baselines on frozen budgets, and the external cohort remains test-only until freeze.

### 6. Adversarial Reviewer

**Owns**

- independent attempt to falsify leakage, geometry, novelty, fairness, calibration, and lesion-safety claims;
- inspection of negative and failed cases;
- verification that every headline sentence maps to evidence.

This role should not own the component under review.

**Gate R0 — Submission red team**

Pass only when no critical issue remains in the claim-to-evidence matrix, leakage-positive control, affine/PSF tests, baseline fairness, external evaluation, statistics, and compute report.

### Gate sequence and stop rules

```text
N0 Claim viability
  + D0 Legal observation
        ↓
G0 Physical operator
        ↓
E1 Static benchmark ready
        ↓
M1 Attributable representation
        ↓
Oracle-headroom check
        ↓
E2 Policy and external evidence
        ↓
R0 Submission red team
```

Stop rules:

- fail N0 if the only remaining novelty is a combination of known components;
- fail D0 if query pixels can influence scoring before commit or if audit data can enter mutable state;
- fail G0 if thin/slab behavior, physical coordinates, or gradients are not independently validated;
- stop representation expansion if the PSF-aware free Gaussian matches the complex state;
- stop routing expansion if the oracle gap is small;
- stop the CVPR claim if gains disappear on the external cohort or lesion fidelity/calibration degrades.

## Smallest tranche implementation brief

### T0 name

**Legal Physical Forward Operator**

### T0 objective

Given only committed context observations and candidate-plane metadata, produce differentiable thin-plane and sampled PSF/slab Gaussian renderings without coordinate ambiguity or hidden-pixel access.

### T0 acceptance package

- contract tests;
- legality and positive-control leakage tests;
- independent thin-plane analytic tests;
- finite-slab/PSF convergence tests;
- affine-equivariance and signed-normal tests;
- forward/gradient chunking parity;
- deterministic CPU run manifest.

### T0 exit decision

- **Pass:** begin the matched free-Gaussian static baseline.
- **Fail on geometry or leakage:** repair T0; do not proceed.
- **Fail because slice profiles are unknown:** keep the generic sampled PSF interface and label the development task retrospective; do not invent scanner physics.

## CVPR go/no-go criteria

**Go** only if:

- the task/action claim is honest and legally enforced;
- the matched static representation is competitive with PSF-aware Gaussian and INR baselines;
- the oracle shows meaningful policy headroom;
- the proposed policy improves normalized quality–budget AUC on controlled and external cohorts under matched reconstruction/compute;
- predicted gain and unsupported-region uncertainty are calibrated against hidden audit error;
- lesion-sensitive and downstream fidelity do not regress;
- the full result is reproducible with transparent compute and dataset provenance.

**No-go or re-scope** if:

- the query cannot be defended beyond an arbitrary slice reveal and the paper still claims scan acceleration;
- a PSF-aware free Gaussian explains the gains;
- uniform, uncertainty, or MI-style routing is near the oracle;
- gains vanish on the external cohort;
- global image metrics improve while lesion recall, failure rates, or calibration worsen;
- baseline code or data restrictions prevent a fair, auditable comparison.
