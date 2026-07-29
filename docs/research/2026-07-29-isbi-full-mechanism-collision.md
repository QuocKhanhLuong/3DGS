# ISBI 2027 Novelty Correction and Full-Mechanism Collision Ledger

- Date: 2026-07-29
- Search cutoff: 2026-07-29, Asia/Ho_Chi_Minh
- Status: Stage-1 research correction; living ledger
- Primary venue frame: ISBI 2027 / medical imaging
- Authority: `docs/` only; deprecated root design documents are not evidence
  for research direction

## 1. Decision

The static representation remains the primary research hypothesis:

```text
permanently sparse observed multi-sequence MRI slices
→ compact teacher-free structural evidence
→ physical support anchors
→ shared tiny anchor-local structural field
→ anchored Gaussian birth
→ iterative anchor–Gaussian propagation
→ latent patient-specific 3D Gaussian representation
→ permanently sparse cross-patient supervision
→ full-volume reconstruction
```

The literature reviewed below creates serious collisions with several large
subsets of this chain, especially sparse-slice MRI reconstruction with explicit
3D Gaussian primitives. It does **not**, in the scoped search completed for
this correction, establish a direct collision with the complete mechanism.
This is a provisional research decision, not a claim that no such work exists.

Consequently:

- T0 is enabling legality and physical-rendering infrastructure only;
- T0.5 and T1 are the approved next tranche;
- T1 is an attribution baseline, not the final novelty claim;
- the representation thesis must be tested before active acquisition;
- active routing, learned information gain, and multi-wave planning are later
  extensions and are not the headline of the current static-representation
  study.

## 2. Unit of novelty analysis

Novelty is evaluated on the **coupled mechanism**, not by asking whether each
ingredient has appeared independently.

The ledger uses the following mechanism links:

| ID | Required link |
|---|---|
| M1 | Permanently sparse, legally fixed multi-sequence MRI observations |
| M2 | Compact teacher-free structural evidence learned across patients |
| M3 | Observed support is lifted to physical support anchors |
| M4 | One shared tiny anchor-local structural field organizes local geometry |
| M5 | Gaussian primitives are born in relation to those physical anchors |
| M6 | Anchors and Gaussians iteratively propagate patient-specific support |
| M7 | Training supervision remains permanently sparse across patients |
| M8 | The resulting latent patient-specific Gaussian state reconstructs a full volume |

A paper is a **direct full-mechanism collision** only when its executable
method contains M1–M8, or technically equivalent links with the same causal
roles. A paper that collides with one or several links is important related
work and an ablation obligation, but is not by itself evidence to demote the
complete representation thesis.

## 2.1 Reproducible search provenance

This is a bounded collision search, not a systematic review. The conditional
negative result in Section 3 is reproducible only within the scope below.

### Search surfaces

Primary-source discovery and verification used:

- arXiv abstract records and author-provided HTML/PDF;
- IEEE/CVF Open Access accepted-paper pages for CVPR;
- MICCAI Society Open Access accepted-paper pages and DOI records;
- MIDL/PMLR or author manuscripts when an accepted proceedings page was not
  yet indexed;
- IEEE Xplore/DOI metadata for named ISBI/TMI works.

Secondary aggregators and search snippets were used only to discover candidate
titles. A work entered the evidence table only after its author manuscript,
accepted paper, or official venue record was inspected.

### Representative query families

Queries were run with spelling and venue variants:

```text
MRI ("3D Gaussian" OR "Gaussian splatting") reconstruction sparse slice
"slice-to-volume" MRI Gaussian primitives PSF
multi-contrast MRI implicit representation sparse anisotropic slices
(anchor OR scaffold) Gaussian splatting refiner propagation sparse view
feed-forward image features 3D anchors anchor-aligned Gaussians refiner
site:openaccess.thecvf.com CVPR 2026 AnchorSplat
site:papers.miccai.org MRI Gaussian splatting reconstruction
site:arxiv.org sparse multi-sequence MRI Gaussian reconstruction
```

The search cutoff is the date at the top of this file. Later paper revisions,
new supplements, and later publications are outside the present verdict and
must trigger the update protocol in Section 10.

### Inclusion and collision rules

A candidate was included when its primary source contained at least one of:

1. sparse/degraded 2D medical observations reconstructed into a continuous 3D
   volume;
2. Gaussian primitives used as a medical volume representation;
3. learned image features lifted to physical/3D anchors that generate
   Gaussians;
4. iterative Gaussian refinement or propagation;
5. multi-sequence or multi-contrast sparse MRI supervision.

The M1–M8 entries are based on executable method semantics, not title or
abstract keyword overlap. `No` does not mean the idea is absent from all
literature; it means the inspected primary method did not contain that causal
link.

### Primary-source evidence locations

| Work | Evidence inspected |
|---|---|
| GaussianSVR | Sections 2.1–2.3: Gaussian volume, slice acquisition model, and self-supervised coarse-to-fine joint optimization |
| Fast and Explicit SVR | Sections 2.1–2.5: physical forward model, explicit Gaussian representation, analytic PSF convolution, and optimization |
| M-Gaussian | Sections 3.1–3.5: registered point-cloud/grid initialization, magnetic Gaussians, global neural residual field, and progressive training |
| GaussianPile | Sections 4.1–4.3: focus-aware model, Gaussian piling, and differentiable optimization |
| MRI novel-view resampling | Section 3 and 3.1–3.2: sparse source planes, random Gaussian initialization, rendering, and scan-specific optimization |
| AnchorSplat | CVPR 2026 Sections 3.3–3.4, especially Eqs. (4)–(12): feature extraction, 3D anchors, anchor-aligned Gaussian prediction, and rendering-error refiner |
| Multi-contrast / multi-view INR baselines | Author abstract and method description for the continuous subject-specific multi-contrast or multi-view representation |

## 3. Direct full-mechanism collision result

**Result as of 2026-07-29: no confirmed M1–M8 collision was found in the scoped
primary-source search.**

This result has only the following meaning:

- the representation hypothesis remains eligible for T1–T3 testing;
- the project must compare against the close Gaussian MRI/SVR methods below;
- novelty language must remain conditional until the searches and matched
  experiments are complete;
- any newly discovered full or near-full collision reopens this gate.

It does not authorize “first”, “first-ever”, “unique”, “unprecedented”, or
state-of-the-art claims.

## 4. Highest-risk collisions

Legend: `yes` means the paper supplies the link in substantially the same role;
`partial` means a nearby but scientifically different mechanism; `no` means
the inspected method does not supply the link.

| Primary source | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | Collision assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [GaussianSVR, ISBI 2026 / arXiv:2601.22990](https://arxiv.org/abs/2601.22990) | partial | no | no | no | no | partial | no | yes | **Highest medical-imaging risk.** It reconstructs a fetal MRI volume from motion-corrupted 2D stacks with a self-supervised slice acquisition model and jointly optimizes localized 3D Gaussian and transform parameters. Its multi-resolution optimization is not the proposed observed-support-anchor field and anchor–Gaussian propagation chain, and it is scan-specific rather than permanently sparse cross-patient representation learning. |
| [Fast and Explicit SVR, MIDL 2026 / arXiv:2512.11624](https://arxiv.org/abs/2512.11624) | partial | no | no | no | no | no | no | yes | Explicit anisotropic Gaussian primitives reconstruct sparse/degraded MRI slices with an analytic Gaussian-PSF forward model. This directly pressures any broad claim that Gaussian primitives plus slice physics are novel. It does not provide teacher-free evidence, physical support anchors, an anchor-local field, propagation, or cross-patient sparse supervision. |
| [M-Gaussian, arXiv:2603.00145](https://arxiv.org/abs/2603.00145) | partial | no | partial | partial | no | partial | no | yes | Multi-stack thick-slice MRI is registered and devoxelized to a point cloud; uniform-grid Gaussians plus a coordinate MLP residual field are progressively optimized. The residual MLP is a global high-frequency correction, not a shared anchor-local structural field conditioned on compact observed evidence. No permanently sparse episodic cross-patient learning or anchored propagation is described. |
| [Rendering Novel Views of MRI Using 3D Gaussian Splatting, arXiv:2606.26236](https://arxiv.org/abs/2606.26236) | partial | no | partial | no | no | no | no | partial | Sparse anisotropic spine MRI is converted into a Gaussian volumetric representation for anatomy-aligned resampling and grading. This collides with the MRI resampling/Gaussian output envelope, but not with the representation-learning chain. |
| [GaussianPile, CVPR 2026 / arXiv:2603.20611](https://arxiv.org/abs/2603.20611) | partial | no | partial | no | partial | no | no | yes | Slice-aware anisotropic Gaussian placement, finite-thickness projection, compression, and volumetric reconstruction form a strong representation/forward-operator collision. Evaluated modalities are microscopy and ultrasound, and the method does not supply teacher-free cross-patient sparse evidence, the anchor-local field, or anchor–Gaussian propagation. |
| [AnchorSplat, CVPR 2026 / arXiv:2604.07053](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html) | no | partial | partial | no | yes | partial | no | partial | **Highest general-CV mechanism risk.** A cross-scene feed-forward model extracts image features, predicts/back-projects 3D geometric priors, downsamples them into anchors, predicts four offset-constrained Gaussians per anchor, and refines their attributes from render errors. It is RGB scene novel-view synthesis using a pretrained MVS module, U-Net, transformer decoder, and point-transformer refiner—not permanently sparse multi-sequence MRI, physical MRI slice support, a tiny anchor-local structural field, or permanently sparse cross-patient target supervision. |
| [Single-subject Multi-contrast MRI SR via INR, arXiv:2303.15065](https://arxiv.org/abs/2303.15065) | partial | partial | no | partial | no | no | no | yes | Jointly reconstructs complementary anisotropic multi-contrast MRI views in a patient-specific continuous implicit representation. It is a necessary baseline for cross-contrast completion, but it has no Gaussian/anchor mechanism and is optimized per subject. |
| [SIMS-MRI, arXiv:2603.22627](https://arxiv.org/abs/2603.22627) | partial | no | no | partial | no | no | no | yes | Single-subject multi-view anisotropic MRI reconstruction combines a hash-encoded implicit field with learned inter-view alignment. It directly challenges benefits attributed only to continuous patient-specific representation, not the proposed anchor–Gaussian mechanism. |
| [3DGSMR, arXiv:2502.06510](https://arxiv.org/abs/2502.06510) | no | no | no | no | no | no | no | yes | Uses explicit 3D Gaussians for self-supervised reconstruction from undersampled k-space. It invalidates a generic “first Gaussian MRI reconstruction” claim but addresses a different observation domain and lacks the support-anchor chain. |

### 4.1 Why GaussianSVR is not yet a full collision

GaussianSVR is the closest confirmed medical-imaging paper because its method
contains:

```text
2D MRI slice stacks
→ simulated slice acquisition model
→ self-supervised Gaussian and motion optimization
→ patient-specific 3D Gaussian volume
```

The inspected primary paper instead initializes/optimizes a free Gaussian
volume and slice transformations in a coarse-to-fine scan-specific procedure.
It does not describe:

```text
compact teacher-free cross-patient structural evidence
→ observed physical support anchors
→ shared anchor-local structural field
→ anchored Gaussian birth
→ anchor–Gaussian propagation
```

This difference must be demonstrated experimentally. It is not enough to
rename free Gaussian centers as anchors or call coarse-to-fine optimization
propagation.

### 4.2 Why M-Gaussian is not yet a full collision

M-Gaussian is especially important because it combines multi-stack MRI,
point-cloud initialization, Gaussian primitives, a neural field, progressive
training, and full-volume output. Its inspected method uses a registered
foreground point cloud to initialize a uniform Gaussian grid and adds a
Fourier-coordinate neural residual field for high-frequency detail.

That residual field does not have the proposed role:

```text
one shared low-capacity rule
+ anchor-local coordinates
+ compact evidence at an observed physical support
→ local structural field used to constrain Gaussian birth and propagation
```

Therefore M-Gaussian is a **near-mechanism warning**, not a confirmed direct
collision. It creates a mandatory matched baseline and makes vague
“Gaussians + local MLP + MRI” novelty wording indefensible.

### 4.3 Why AnchorSplat is a high-risk partial collision

The accepted [CVPR 2026
paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html)
contains a close general-CV causal subchain:

```text
posed/unposed RGB images
→ pretrained multi-view geometry and CNN image features
→ back-projected and farthest-point-sampled 3D anchors
→ transformer anchor features
→ offset-constrained anchor-aligned Gaussian prediction
→ rendering-error-driven Gaussian attribute refiner
→ scene novel-view rendering
```

This directly collides with the broad formulation
“image features → 3D anchors → anchor-aligned Gaussian birth → iterative
Gaussian refinement.” Accordingly, the project cannot claim that anchor-aligned
Gaussian generation or learned Gaussian refinement is novel by itself.

The current full-mechanism verdict remains conditional but unchanged because
AnchorSplat does not implement the medical and representation-specific links:

- input observations are RGB scene views, not a permanently sparse legal
  multi-sequence MRI manifest;
- its anchors derive from pretrained MVS depth/pose geometry rather than
  acquired physical MRI slice support and registered RAS-mm plane semantics;
- its decoder is an 84M-parameter transformer stack, not one shared tiny
  anchor-local structural field;
- each anchor directly predicts Gaussians; there is no separately validated
  structural-field interface organizing birth;
- its refiner adjusts existing Gaussian attributes from input-view render
  errors, which is related to but not equivalent to iterative
  anchor–Gaussian support propagation into sparsely observed anatomy;
- training uses scene-view rendering/depth losses, not render-before-reveal
  permanently sparse cross-patient context-to-target supervision;
- the output target is RGB scene novel-view synthesis, not multi-sequence
  full-volume MRI reconstruction with unsupported-region and lesion-fidelity
  accounting.

AnchorSplat nevertheless raises the novelty bar for T2–T3. Future comparisons
must isolate whether the tiny structural field and propagation provide value
beyond a medicalized feed-forward anchor-to-Gaussian decoder/refiner.

## 5. Component-level related work

The following sources establish that major components are mature. They narrow
the claims but do not determine the full-mechanism verdict.

### Sparse and self-supervised MRI reconstruction

- [NeSVoR, IEEE TMI 2023](https://doi.org/10.1109/TMI.2023.3238170)
  models MRI slice-to-volume reconstruction as a continuous implicit neural
  representation with slice acquisition physics, motion, bias, noise, and
  outlier handling.
- [Single-subject Multi-contrast MRI SR via INR,
  arXiv:2303.15065](https://arxiv.org/abs/2303.15065) establishes
  subject-specific continuous reconstruction using complementary
  multi-contrast anisotropic views.
- [INR meets Multi-Contrast MRI Reconstruction,
  arXiv:2509.04888](https://arxiv.org/abs/2509.04888) jointly reconstructs
  complementary undersampled multi-contrast k-space. Its measurement domain
  differs, but it pressures claims about cross-contrast complementarity.

### Gaussian medical representations

- [GaussianSVR](https://arxiv.org/abs/2601.22990),
  [Fast and Explicit SVR](https://arxiv.org/abs/2512.11624),
  [M-Gaussian](https://arxiv.org/abs/2603.00145), and
  [3DGSMR](https://arxiv.org/abs/2502.06510) establish that Gaussian
  representations for MRI reconstruction are no longer novel in isolation.
- [GaussianPile](https://arxiv.org/abs/2603.20611) establishes slice-aware
  Gaussian placement and a finite-thickness imaging model for slice-based
  volumetric reconstruction.
- [Gaussian Pancakes, MICCAI
  2024](https://doi.org/10.1007/978-3-031-72089-5_26) establishes
  geometry/depth-regularized surface-aligned Gaussians in medical endoscopic
  reconstruction.
- [AnchorSplat, CVPR
  2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html)
  establishes the feed-forward image-feature-to-anchor-to-Gaussian-to-refiner
  subchain in general scene reconstruction.
- [EndoSparse, MICCAI
  2024](https://doi.org/10.1007/978-3-031-72089-5_24) establishes sparse-view
  medical reconstruction with Gaussian splatting and privileged foundation
  priors. Those priors are not legal main-path substitutes for this project's
  teacher-free encoder.

### Anchors, local fields, and propagation

Sparse anchors, local coordinate fields, adaptive Gaussian density, and
coarse-to-fine propagation all have extensive general-CV precedents. Their
existence means the paper cannot claim novelty for any one noun. The research
question is whether conditioning a shared tiny local field on permanently
sparse, physically anchored multi-sequence MRI evidence provides a measurable
reconstruction advantage over:

1. analytic-only fixed supports;
2. a raw shallow CNN with the same supports;
3. free or grid-initialized Gaussian MRI reconstruction;
4. a global coordinate residual field;
5. interpolation and established MRI SVR.

## 6. Required differentiation and matched comparisons

The representation thesis survives only if the end-to-end causal additions
earn their complexity.

### T1 attribution

T1 must hold observations, episode assignments, physical support positions,
primitive count, renderer profile, optimization steps, and compute accounting
constant across:

```text
E0  analytic only
E1  raw shallow CNN
E2a analytic + micro-CNN, reconstruction only
E2b analytic + micro-CNN, all teacher-free losses
```

E3/E4 are upper bounds and cannot be used to define the main method.

### T2–T3 attribution, only after human approval

Future work must add mechanisms incrementally:

```text
fixed physical supports
→ learned observed-support anchors
→ anchor-local structural field
→ anchored Gaussian birth
→ iterative anchor–Gaussian propagation
```

At every transition, support count, renderer, legal observations, and optimizer
opportunity must be matched. Comparisons should include at least a free/grid
Gaussian MRI baseline and a patient-specific INR baseline, plus the closest
available implementations of GaussianSVR, M-Gaussian, and Fast and Explicit
SVR when their code/data contracts permit a fair comparison.

## 7. Representation demotion rules

The representation is demoted only when at least one of these conditions is
met:

1. a primary source is verified to implement the complete M1–M8 mechanism, or
   a technically equivalent causal chain; or
2. a matched ablation shows that the proposed representation is not
   meaningfully better than the corresponding simpler baseline.

Operational consequences:

- `E0 ≈ E2b`: remove the learned encoder from the novelty path;
- `E1 ≈ E2b`: remove the analytic-scaffold claim;
- auxiliary feature diagnostics improve but reconstruction does not: demote
  auxiliary losses to diagnostics;
- fixed Gaussian reconstruction does not beat interpolation: stop before
  propagation and repair data/state/renderer assumptions;
- anchor-local field does not beat an equally budgeted global/free Gaussian
  alternative: remove or simplify the field;
- propagation does not beat fixed topology under matched compute and primitive
  opportunity: remove propagation from the main claim;
- no competitive static representation baseline: do not implement routing.

## 8. Medical-imaging validity obligations

An ISBI/MICCAI-quality result must establish more than plausible global image
quality:

- patient-level splits and physically isolated audit evaluation;
- explicit registration assumptions and failure analysis;
- no non-manifest or audit pixels in training;
- context/target roles assigned per episode without changing availability;
- render-before-reveal receipts;
- affine and physical-plane correctness;
- unsupported coverage reported rather than silently filled;
- lesion/ROI and boundary fidelity, not only PSNR/SSIM;
- missing-modality behavior and registration-confidence weighting;
- matched primitive, renderer, observation, optimization, and compute budgets;
- leakage-positive controls that are demonstrated to fail.

## 9. Claim boundary

### Permitted now

- “We investigate” the complete support-anchor Gaussian mechanism.
- T0 is a legal, through-plane profile-aware Gaussian **reference** renderer.
- T1 tests whether compact teacher-free evidence improves a fixed-topology
  Gaussian reconstruction bridge.
- The complete representation remains a hypothesis pending T1–T3 gates.

### Prohibited now

- “first Gaussian MRI reconstruction”;
- “first sparse-slice Gaussian volume reconstruction”;
- “novel because it uses anchors / a local MLP / Gaussians / propagation”;
- “PSF-correct”, “scanner-accurate PSF”, or “complete physical MRI forward
  operator” for the current renderer;
- “true SDF” without a sign convention, Eikonal test, gradient-norm
  statistics, and distance calibration;
- raw support mass as calibrated uncertainty;
- lesion fidelity without isolated lesion/ROI audit evidence;
- active acquisition or routing as the current primary contribution.

## 10. Living collision ledger protocol

This file must be updated, rather than replaced by an untraceable one-time
search.

For each new candidate work, add:

```text
date discovered
primary-source URL / DOI / arXiv identifier
paper version and venue status
M1–M8 mapping with evidence locations
closest executable mechanism
new baseline or ablation obligation
gate action: no change / reopen / demote
reviewer and human decision
```

Reopen the novelty gate when:

- a new or revised work matches four or more links including M4–M7;
- a work combines observed support, a local field, and propagated Gaussians in
  sparse MRI;
- a work uses permanently sparse cross-patient supervision for a patient-
  specific Gaussian reconstruction state;
- code or supplements reveal a mechanism absent from an abstract;
- a T1–T3 matched ablation invalidates the claimed causal advantage.

Search surveillance must include both medical-imaging venues
(ISBI/MICCAI/MIDL/TMI/MRM) and general-CV venues (CVPR/ICCV/ECCV), while the
paper's scientific framing remains ISBI/medical imaging.

## 11. Current gate recommendation

**Recommendation: PARTIAL / proceed only with the approved T0.5 and T1
attribution tranche.**

Rationale:

- no confirmed full-mechanism collision was found;
- Gaussian MRI/SVR itself is already a crowded and fast-moving area;
- the unique value, if any, must come from the coupled legal sparse-support,
  compact evidence, anchor-local field, anchored birth, and propagation
  mechanism;
- T1 cannot validate M3–M6, but it can remove an unjustified encoder branch
  before the project pays the cost of T2–T3;
- no routing work is warranted before a competitive static reconstruction
  baseline exists.
