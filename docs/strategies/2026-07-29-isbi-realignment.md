# ISBI 2027 Strategy Realignment — Sparse Support-Anchor Gaussian Reconstruction

Date: 2026-07-29
Status: authoritative Stage-1 strategy correction; awaiting the human gate before T0.5 implementation
Owner: PM / scientific-program lead
Primary venue: ISBI 2027, with medical-imaging validity standards aligned to ISBI/MICCAI

## 1. Authority and precedence

This document is the governing research and implementation strategy for the
current repository.

When another document conflicts with this strategy:

1. this ISBI realignment controls the paper thesis, claims, tranche order,
   gates, and stop decisions;
2. the approved T0 and T0.5/T1 plans control implementation detail within their
   authorized tranche;
3. reconstruction phase and module documents remain useful design references
   only where they are consistent with items 1 and 2.

In particular, `docs/strategies/2026-07-29-cvpr-priorities.md` and any other
CVPR-first strategy document are non-authoritative on conflict. They are
historical research records, not permission to make active routing the paper
headline or to implement T2+ work.

## 2. Venue framing and single paper thesis

The target is **ISBI 2027 / medical imaging**, not a CVPR-first system. The
scientific question is reconstruction validity under permanently sparse,
multi-sequence MRI evidence. General-CV components are relevant prior art, but
they do not by themselves determine the contribution or the evaluation
standard.

The project will keep one falsifiable representation thesis across T1–T3:

> Under permanently sparse, patient-separated multi-sequence MRI supervision,
> a compact teacher-free evidence path coupled to physical support anchors, one
> shared tiny anchor-local structural field, anchored Gaussian birth, and
> iterative anchor–Gaussian propagation produces a more faithful
> patient-specific 3D reconstruction than matched simpler fixed-support and
> free/grid Gaussian alternatives, without using non-manifest or audit pixels.

The complete causal chain is:

```text
permanently sparse multi-sequence MRI slices
→ compact teacher-free structural evidence
→ physical support anchors
→ shared tiny anchor-local structural field
→ anchored seed Gaussians
→ anchor–Gaussian propagation
→ latent patient-specific 3D Gaussian representation
→ full-volume reconstruction
```

This thesis is not approved because every block can be implemented. It survives
only when matched experiments show that each added causal mechanism earns its
complexity and when medical-imaging validity gates pass.

## 3. Research-direction decisions

### 3.1 Primary contribution

The primary thesis is the **permanently sparse support-anchor Gaussian
representation**. T1 establishes the legal and attribution bridge; T2 and T3,
if later authorized, test the physical support-anchor field and propagation
mechanism that could support the final representation claim.

Novelty must be assessed on the coupled mechanism:

```text
sparse observed support seeds
→ local structural field
→ anchored Gaussian birth
→ iterative anchor–Gaussian propagation
→ permanently sparse cross-patient supervision
```

Finding prior art for SDFs, anchors, Gaussians, local MLPs, or active routing
separately is not sufficient to demote the representation. Demotion requires
either:

1. verified direct prior art for the full mechanism or a technically
   equivalent causal chain; or
2. a matched ablation showing that the complex representation does not improve
   over the corresponding simpler baseline.

The current collision ledger reports no confirmed full-mechanism collision,
but identifies close Gaussian MRI/SVR work. This is permission to test a
hypothesis, not evidence for a priority or first-of-kind claim.

### 3.2 T0

T0 is completed enabling infrastructure: legality contracts, canonical
physical geometry, differentiable Gaussian state, and a CPU reference
renderer. T0 is **not** the central novelty and must not replace the
representation thesis.

The current renderer must be called the:

> through-plane profile-aware Gaussian reference renderer

### 3.3 Active routing

Active trajectory selection is a later extension at T4. It is not part of the
current tranche and is not the paper headline. No routing, multi-wave planning,
learned information gain, or adaptive acquisition may be implemented now.

The headline may not switch from static representation to routing without:

1. a competitive static representation baseline;
2. completed static-representation attribution;
3. an explicit human decision approving the thesis change.

### 3.4 Next approved tranche

T0.5 and T1 are the next **scientifically approved tranche**, in this order:

```text
T0.5 — legal episodic training corrections
T1   — teacher-free encoder + fixed-topology Gaussian baseline
```

This strategy approval does not bypass the Stage-1 human gate. No T0.5 code is
authorized until that gate is explicitly approved, and T1 cannot start until
T0.5 passes all blocking tests and its next human gate.

## 4. Tranche boundaries

### T0.5 may contain

- separation of `SparseAvailabilityManifest` from immutable
  `EpisodeAssignment`;
- receipt-gated render-before-reveal semantics;
- separation of offline training roles from exact-`Decimal` deployment
  acquisition cost;
- a named amplitude-gauge/coverage policy and regression tests;
- corrected renderer naming;
- exact-clean-commit CPU CI and leakage-positive tests.

### T1 may contain

- an analytic differential bank;
- E0 analytic-only and E1 raw shallow-CNN baselines;
- E2 analytic-scaffold teacher-free micro-CNN variants;
- structural losses with explicit registration confidence and anti-collapse
  diagnostics;
- deterministic, matched physical support points;
- a safe fixed-topology Gaussian constructor;
- an episodic trainer that enforces prediction receipts;
- matched configs, smoke runs, and E0/E1/E2 attribution experiments;
- E3/E4 only as clearly separated privileged upper bounds.

### Not authorized in this run

- T2 support-anchor bootstrap or anchor-local field implementation;
- T3 anchored Gaussian propagation;
- learned birth, split, merge, prune, or adaptive topology;
- T4 routing, learned utility, multi-wave planning, or adaptive acquisition;
- T5 full-volume export;
- placeholder modules or APIs for any of the above.

## 5. Human-gated execution sequence

The stages are sequential. Passing a local test does not authorize the next
stage; each stated human gate is blocking.

### Stage 1 — strategy, roles, and design correction

Deliver:

- authoritative ISBI realignment;
- full-mechanism novelty correction;
- corrected team-role instructions;
- permanently sparse training protocol;
- corrected reconstruction order and root README;
- T0.5/T1 design delta.

Then stop at the **Human Gate 1**. Approval authorizes Stage 2 only.

### Stage 2 — T0.5 legal episodic training

Implement T0.5 only, write blocking tests in parallel, run exact-clean-commit
CI, and obtain an independent T0.5 review.

Then stop at the **Human Gate 2**. T1 remains unauthorized until approval.

### Stage 3 — T1-A synthetic contracts

Implement analytic features, feature-grid alignment, deterministic fixed
supports, the safe Gaussian constructor, and the end-to-end renderer-to-encoder
gradient contract using synthetic data.

Then stop at the **Human Gate 3**.

### Stage 4 — T1 learned components and smoke experiment

Implement the micro-CNN, switchable structural losses, episodic trainer,
E0/E1/E2 configs, and a bounded smoke experiment.

Then stop at the **Human Gate 4**.

### Stage 5 — matched T1 decision

Run the matched T1 experiments, report paired results and medical-fidelity
evidence, and obtain an independent reviewer decision of `PASS`, `PARTIAL`, or
`FAIL`.

Stop after the T1 decision. T2 is a separate future authorization.

## 6. Blocking gates

### Gate T0.5-L — legal episodic sparse training

Pass only when all are true:

- episode context/target roles are separate from permanent availability;
- assignments are immutable, patient-consistent, disjoint, manifest-bound, and
  canonically hashed;
- one manifest supports multiple legal assignments without mutation;
- target reveal requires a single-use prediction receipt bound to target,
  episode, assignment, ledger, state version, physical-plane hash, renderer
  version, and prediction digest;
- wrong-target, wrong-episode, wrong-state, wrong-ledger, mismatched-plane,
  missing, and reused receipts are rejected;
- offline role assignment consumes no acquisition budget;
- deployment bootstrap and subsequent commitments consume exact declared
  `Decimal` cost;
- gauge-equivalent intensity parameterizations cannot arbitrarily change
  supported/unsupported classification;
- leakage-positive controls pass;
- exact-clean-commit CI passes.

### Gate T1-F — feature validity

Pass only when:

- pixel-center and stride-1/2/4 feature alignment tests pass;
- analytic outputs are finite on constant and low-signal inputs;
- structural channels do not collapse;
- registered matching locations are more similar than mismatched locations;
- local differential cues remain recoverable;
- anti-collapse statistics are logged per channel.

### Gate T1-R — reconstruction attribution

Pass only when E2b improves both E0 and E1 under the same:

- patients, manifests, and episode assignments;
- physical support positions and primitive count;
- renderer profile and coverage policy;
- optimizer opportunity and steps;
- parameter, FLOP, runtime, cache, and hardware accounting.

Feature-proxy improvements alone do not pass this gate.

### Gate T1-M — medical fidelity

Pass only when the reconstruction improvement does not cause a meaningful
lesion/ROI fidelity regression on the patient-disjoint T1 lesion-validation
audit cohort. Before the evaluator is opened, freeze its sparse reconstruction
input manifest, checkpoint, config, primary lesion/ROI and boundary estimands,
paired non-inferiority margins, confidence-interval procedure, multiplicity
policy, coverage metrics, and failure rules. Global image metrics,
supported-only metrics, or visually selected examples cannot substitute for
this gate.

This one-shot T1 gate cohort is not the sealed T5 final-audit cohort. Its dense
targets and labels are evaluator-only and cannot drive training, normalization,
support placement, hyperparameter tuning, early stopping, or checkpoint
ranking. The T5 cohort remains unopened and cannot select or authorize T2.

No work may proceed to T2 until the human explicitly accepts all applicable
T0.5-L, T1-F, T1-R, and T1-M evidence.

## 7. Stop rules and consequences

| Trigger | Required decision |
|---|---|
| Verified direct prior art implements the full mechanism or a technically equivalent causal chain | Reopen novelty review and demote or materially revise the representation claim before further expansion. |
| `E0 ≈ E2b` under the matched T1 contract | Remove the learned encoder from the novelty path. |
| `E1 ≈ E2b` under the matched T1 contract | Remove the analytic-scaffold claim. |
| Auxiliary losses improve feature diagnostics but not target reconstruction | Demote auxiliary losses to diagnostics. |
| Fixed-topology Gaussian baseline does not beat the interpolation floor | Do not implement propagation; repair data, state, geometry, coverage, or renderer assumptions first. |
| Anchor-local field later fails to beat an equally budgeted simpler global/free Gaussian alternative | Remove or simplify the field claim. |
| Propagation later fails to beat fixed topology under matched compute and primitive opportunity | Remove propagation from the main claim. |
| No competitive static representation baseline exists | Do not implement routing. |
| Any non-manifest or audit pixel enters training | Invalidate the run and block all reconstruction claims from it. |
| Leakage-positive control does not fail closed | Block the tranche. |
| Global metrics improve while lesion/ROI fidelity regresses meaningfully | Fail T1-M; do not advance the representation. |
| A field lacks sign convention, Eikonal test, gradient-norm statistics, or distance calibration | Call it `StructuralField`, not SDF. |

An `≈` decision must use the predeclared paired equivalence/non-inferiority
criterion and uncertainty analysis; it cannot be declared from overlapping
point estimates alone.

## 8. Claim-to-evidence matrix

| Candidate claim | Minimum admissible evidence | Evidence that is insufficient | Decision if evidence fails |
|---|---|---|---|
| Training is permanently sparse | Immutable manifests, patient-level split hashes, opened-file ledger, and explicit zero non-manifest/audit pixel counts | A sparse sampler over a loader that can see dense or audit data | Invalidate the run |
| Context-to-target learning is legal | Immutable episode assignments plus receipt-gated event order and leakage-positive controls | `commit` followed by immediate reveal; a permanent `CONTEXT/TARGET` metadata flag | Fail T0.5-L |
| T1 teacher-free evidence adds reconstruction value | Paired E2b improvement over E0 and E1 under matched downstream and compute conditions | Better proxy loss, feature visualization, or an unmatched larger network | Remove or simplify the encoder claim |
| Analytic scaffolding adds value | Paired E2b improvement over E1 attributable under matched architecture/opportunity | Derivative channels look interpretable | Remove the analytic-scaffold claim |
| Support-anchor field adds representation value | Future matched incremental ablations against deterministic supports and an equally budgeted free/global alternative | Prior-art component lists or qualitative anchor plots | Do not retain the field as a contribution |
| Anchor–Gaussian propagation adds value | Future matched improvement over fixed topology with equal compute and primitive opportunity | More primitives, more steps, or adaptive compute | Remove propagation from the main claim |
| Reconstruction improves medically relevant fidelity | Patient-level T1 lesion-validation audit under a frozen sparse input manifest, paired margins/intervals, lesion/ROI, boundary, coverage, and failure metrics; sealed T5 final audit remains separate | PSNR/SSIM alone, supported-only metrics, repeated gate-set tuning, or opening the T5 final audit | Fail T1-M / later medical gate |
| The renderer models the current acquisition abstraction | Analytic/reference tests for geometry, finite through-plane profiles, masking, and autograd | Current reference implementation alone as scanner calibration evidence | Retain reference-only wording |
| Uncertainty is calibrated | A declared target, calibration protocol, proper scoring or coverage analysis, and gauge-invariant estimator | Raw `support_mass` or encoder reliability map | Call it support/reliability diagnostic only |
| The field is an SDF | Sign convention, Eikonal test, gradient-norm statistics, and distance calibration | A scalar local field or near-surface visualization | Use `StructuralField` |
| The full mechanism is novel | Living primary-source M1–M8 collision ledger plus a verified search and matched differentiation | Novelty of individual anchors, Gaussians, MLPs, SDFs, or routing | No novelty priority claim; reopen review |
| Active routing improves acquisition | Future budget-matched deployment study after static reconstruction passes | Training episode reassignment or uncalibrated support mass | No routing claim or implementation |

## 9. Prohibited claims

Until the corresponding evidence exists, team artifacts, code comments,
experiment reports, and paper drafts must not claim:

- that the current work is CVPR-first or that active routing is the primary
  contribution;
- that T0 is the central novelty;
- “first Gaussian MRI reconstruction”, “first sparse-slice Gaussian volume
  reconstruction”, “first”, “first-ever”, “unique”, “unprecedented”, or
  state-of-the-art priority;
- novelty merely because anchors, Gaussians, SDFs, local MLPs, propagation, or
  routing are present;
- `PSF-correct`, `scanner-accurate PSF`, or `complete physical MRI forward
  operator` for the current renderer;
- scanner-side or clinical acquisition validity from the reference renderer;
- `SDF` before all SDF evidence in the stop rule exists;
- raw `support_mass` as calibrated uncertainty, or any routing/coverage
  decision that depends arbitrarily on the global amplitude gauge;
- E3 frozen-pretrained or E4 teacher-distilled privileged upper bounds as the
  main method;
- success from feature proxy losses without sparse target reconstruction;
- reconstruction success using supported-only metrics without coverage and
  failure reporting;
- medical or lesion fidelity from global PSNR/SSIM alone;
- permanently sparse training if a non-manifest pixel, dense parent, or audit
  pixel entered training, preprocessing, registration, support selection,
  normalization, caching, model selection, or checkpoint selection;
- a complete full-volume method, active acquisition method, or adaptive
  topology before its later tranche is implemented and approved.

## 10. Program decision at Stage 1

The PM approves the revised **ISBI representation thesis** and the bounded
T0.5-then-T1 tranche as the program strategy. The recommendation at this stage
is:

**PARTIAL — proceed to Human Gate 1, not to implementation.**

Rationale:

- no confirmed full-mechanism collision currently requires demotion;
- close Gaussian MRI work makes component-level novelty claims untenable;
- T1 can falsify unnecessary encoder complexity before T2/T3 investment;
- buildability is not evidence of reconstruction or medical validity;
- T0.5 legal contracts must pass before any T1 experiment can be trusted.

This approval explicitly preserves the user's Human Gate. It does not authorize
T0.5 source changes, T1 source changes, or any T2+ module in the current Stage-1
run.
