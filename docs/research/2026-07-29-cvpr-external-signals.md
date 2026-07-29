# CVPR External Signals for Sparse Active 3D Reconstruction

Date: 2026-07-29  
Role: AgenTeam Researcher  
Decision horizon: CVPR-level method and experimental plan

## Executive verdict

The project addresses a timely problem, but the current paper story is too broad and several of its apparent component novelties now have direct 2024–2026 collisions.

The most important new signal is **GaussianPile (CVPR 2026)**. It already proposes slice-aware anisotropic Gaussians, a differentiable imaging-system-aware slice operator, finite-thickness point-spread-function (PSF) modeling, additive volumetric composition, and joint reconstruction/compression for slice-based biomedical volumes. It reports up to \(11\times\) faster fitting than NeRF-style alternatives and \(16\times\) compression on ultrasound and microscopy ([official CVPR paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_GaussianPile_A_Unified_Sparse_Gaussian_Splatting_Framework_for_Slice-based_Volumetric_CVPR_2026_paper.html), [arXiv](https://arxiv.org/abs/2603.20611)).

The project therefore should **not** claim novelty for:

- applying Gaussian primitives to sparse medical slices;
- physical-plane or finite-thickness Gaussian rendering by itself;
- combining an SDF with Gaussian splatting;
- organizing local Gaussians around anchors;
- using a shared tiny MLP over local SDF supports;
- teacher-free/self-supervised slice-to-volume reconstruction by itself;
- using uncertainty or information gain for active Gaussian reconstruction.

The most defensible CVPR direction is narrower:

> **Closed-loop, leakage-safe, multi-sequence active plane acquisition driven by calibrated marginal reconstruction gain over a PSF-correct Gaussian patient state.**

The paper must make the active acquisition protocol, not the representation inventory, the main contribution. It must also prove that a `sequence × slice` query is a real acquisition/reacquisition action or explicitly rename the task as **retrospective budgeted progressive observation**. Standard 3D MRI is acquired in k-space; current active-MRI work selects measurements or sampling patterns, not already reconstructed axial slices ([Active MRI Acquisition with Diffusion-Guided Bayesian Experimental Design](https://arxiv.org/abs/2506.16237), [fastMRI data contract](https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md)).

Overall external-risk rating: **high until the contribution and acquisition model are narrowed**.

## Corpus and staleness assessment

The following complete corpus was reviewed:

- `README.md`;
- `architecture.md`;
- `pipeline.md`;
- `MASTER_KNOWLEDGE.md`;
- `KNOWLEDGE_PACKAGE.md`;
- every file under `docs/reconstruction/`, including all phase and module specifications.

Per the project-owner clarification received during this review, `architecture.md`, `pipeline.md`, `MASTER_KNOWLEDGE.md`, and `KNOWLEDGE_PACKAGE.md` are deprecated. They were read for historical context but are excluded from the authoritative staleness, scope, and consistency conclusions below. The `docs/` tree is the source of truth.

There was no pre-existing file under `docs/research/` to refresh. All reviewed reconstruction documents had workspace modification dates of 2026-07-29, so none is older than the two-week staleness threshold.

They are nevertheless **externally stale in content** because the current documents do not account for several now-essential 2025–2026 baselines:

- GaussianPile (CVPR 2026);
- GaussianSVR (2026);
- GSVR / analytic-PSF Gaussian primitives (MIDL 2026);
- MedGS (2025);
- the 2025 tri-plane–Gaussian medical reconstruction paper;
- active-Gaussian planning methods GauSS-MI and ActiveGAMER;
- SurfaceSplat and the broader SDF–Gaussian line.

## Findings

### 1. Direct novelty collisions

| Current project element | Closest primary-source collision | Consequence |
|---|---|---|
| Slice-aware anisotropic Gaussians and physical-plane rendering | GaussianPile introduces slice-aware piling, a differentiable PSF-aware projection operator, additive volumetric rendering, compression, and CUDA acceleration ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_GaussianPile_A_Unified_Sparse_Gaussian_Splatting_Framework_for_Slice-based_Volumetric_CVPR_2026_paper.html)). | This is no longer a standalone novelty. GaussianPile must be implemented or used as a matched reference. |
| Gaussian MRI slice-to-volume reconstruction without ground-truth volumes | GaussianSVR uses a simulated slice-acquisition model and self-supervised multi-resolution optimization ([arXiv 2026](https://arxiv.org/abs/2601.22990)). GSVR derives exact Gaussian–PSF convolution and jointly optimizes primitives and per-slice motion ([paper](https://arxiv.org/abs/2512.11624), [official code](https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR)). | “Teacher-free” or “self-supervised Gaussian SVR” is prior art. The project needs an active/multi-sequence result beyond reconstruction. |
| Sparse sequential medical frames represented by Gaussians | MedGS uses Folded-Gaussians with shared geometry and separate image/segmentation attributes, trained on sparse sequential medical slices ([paper](https://arxiv.org/abs/2509.16806), [official repository](https://github.com/gmum/MedGS)). | The structural/appearance separation and sparse-frame interpolation story is not sufficient novelty. |
| 3D Gaussian representation for MRI | 3DGSMR uses explicit 3D Gaussians for self-supervised isotropic 3D MRI reconstruction from undersampled complex k-space ([arXiv 2025](https://arxiv.org/abs/2502.06510)). | “First Gaussian MRI reconstruction” is unavailable; this is a different forward problem but mandatory related work. |
| Hybrid global field plus Gaussian primitives for sparse medical slices | A 2025 medical reconstruction paper combines tri-plane global features with 3D Gaussians for sparse MRI/ultrasound reconstruction and segmentation ([arXiv](https://arxiv.org/abs/2512.22800)). | The paper must compare local SDF/anchor fields against a tri-plane–Gaussian global-field baseline. |
| SDF plus Gaussian mutual constraint | GSDF jointly optimizes SDF and 3DGS with mutual guidance ([NeurIPS 2024](https://openreview.net/forum?id=r6V7EjANUK)); SurfaceSplat uses SDF for global structure and Gaussians for detail under sparse views ([ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/html/Gao_SurfaceSplat_Connecting_Surface_Reconstruction_and_Gaussian_Splatting_ICCV_2025_paper.html)); GaussianUDF infers distance fields through splatted Gaussians ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_GaussianUDF_Inferring_Unsigned_Distance_Functions_through_3D_Gaussian_Splatting_CVPR_2025_paper.html)). | “SDF defines the Gaussian manifold” needs a medical-volume-specific mathematical or empirical distinction, not a generic hybrid claim. |
| Anchor-organized local Gaussians with growth/pruning | Scaffold-GS uses anchor points to distribute local Gaussians and grows/prunes anchors based on importance ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Scaffold-GS_Structured_3D_Gaussians_for_View-Adaptive_Rendering_CVPR_2024_paper.html)). | Adaptive anchors and local Gaussian spawning are prior art at the representation level. |
| Shared tiny local SDF with adaptive supports and prune/expand | 3D-SLNR represents a global SDF as local band-limited SDFs sharing a tiny MLP and adapts support geometry with prune-and-expand ([CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_3D-SLNR_A_Super_Lightweight_Neural_Representation_for_Large-scale_3D_Mapping_CVPR_2025_paper.pdf)). | The project is an adaptation of an explicit recent design; cite and ablate it directly. |
| Active information-gain routing over a Gaussian map | GauSS-MI quantifies per-Gaussian uncertainty and uses Shannon mutual information for real-time next-best-view selection ([RSS 2025](https://www.roboticsproceedings.org/rss21/p030.html)). ActiveGAMER uses rendering-based information gain and coarse-to-fine exploration over a 3DGS map ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_ActiveGAMER_Active_GAussian_Mapping_through_Efficient_Rendering_CVPR_2025_paper.html)). | “Observability-driven active Gaussian reconstruction” is not novel alone. The multi-sequence plane action, legal sparse training, calibration, and budget objective must be the distinction. |

### 2. What remains potentially novel

The **intersection** below appears less directly occupied:

1. a patient-specific Gaussian volume state;
2. registered multi-sequence plane actions with modality-specific cost;
3. strict no-peeking action commitment;
4. a learned or analytic predictor of *marginal full-volume reconstruction gain*;
5. balanced multi-source coverage under a global budget;
6. calibration of both the next-query gain and the final unsupported-region uncertainty;
7. training from complementary permanently sparse manifests rather than complete target volumes.

This is a credible research gap, but it is not yet a contribution until the project demonstrates all of the following:

- the action is physically or operationally valid;
- the gain predictor generalizes across patients and sites;
- balance improves the quality–budget frontier beyond greedy uncertainty and GauSS-MI-style mutual information;
- the result survives matched-capacity renderer and representation baselines;
- pathology-sensitive fidelity does not degrade while global PSNR improves.

The current “SDF → low-DoF Gaussians → observability → multi-wave” chain is a useful systems hypothesis. It should be presented as a **causal ablation chain**, not four simultaneous novelty claims.

### 3. Acquisition-model realism

There are three materially different tasks:

| Task | What is selected | Physical validity | Recommended wording |
|---|---|---|---|
| Scanner-side active MRI | k-space samples, masks, trajectories, or a separately prescribed 2D acquisition | Highest; matches current active-MRI research ([active Bayesian design](https://arxiv.org/abs/2506.16237), [fastMRI](https://fastmri.med.nyu.edu/)). | “Active MRI acquisition” only if the forward operator and scanner constraints are modeled. |
| Multi-stack / slice reacquisition | an actual 2D stack, slab, or plane with thickness, PSF, motion, and scan time | Plausible for fetal MRI, ultrasound, and targeted reacquisition; established SVR literature models these effects ([NeSVoR](https://pmc.ncbi.nlm.nih.gov/articles/PMC10287191/), [GSVR](https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR)). | “Active plane/stack acquisition” with explicit protocol. |
| Retrospective slice reveal | an already reconstructed slice is hidden on disk and revealed after routing | Valid as information-budgeted progressive observation, but not evidence of scan-time reduction. | “Budgeted progressive observation” or “active slice selection from a registered pool.” |

The current repository implements the third abstraction in its documents. A CVPR paper can use it, but must not infer clinical scan acceleration from it.

There is also a source conflict that must be resolved empirically: GaussianPile illustrates MRI as a zero-thickness slice regime, whereas NeSVoR and GSVR explicitly model thick-slice MRI with a PSF. Both can be correct for different protocols. The renderer must read or estimate each acquisition’s actual slice profile, thickness, spacing, and orientation rather than hard-code either assumption.

### 4. Strongest practical baseline set

#### Representation and reconstruction baselines

| Baseline | Why it is necessary | Practical status |
|---|---|---|
| Linear, cubic/B-spline, and edge-aware interpolation | Establishes the non-learning floor and exposes whether smooth outputs drive PSNR. | Easy, deterministic, and mandatory. |
| FC-SVR | A CVPR 2024 single-stack MRI reconstruction baseline with official code and pretrained weights ([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Young_Fully_Convolutional_Slice-to-Volume_Reconstruction_for_Single-Stack_MRI_CVPR_2024_paper.html), [code](https://github.com/seannz/svr)). | Code is available but pinned to PyTorch 1.13.1 and requires FreeSurfer 7, so containerize it. |
| NeSVoR | Strong continuous implicit MRI SVR baseline with PSF, motion, bias, outlier variance, and uncertainty ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10287191/), [current docs](https://nesvor.readthedocs.io/en/latest/commands/reconstruct.html)). | Most mature task-specific reference; Docker/documented commands exist. |
| GSVR / analytic Gaussian primitives | Closest Gaussian MRI forward-model baseline: exact Gaussian–PSF convolution, motion correction, and fast patient fitting ([paper](https://arxiv.org/abs/2512.11624), [code](https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR)). | Public but young: five commits; modern CUDA stack and brittle FAISS dependency. |
| GaussianSVR | Direct self-supervised Gaussian fetal-MRI SVR comparison ([arXiv](https://arxiv.org/abs/2601.22990)). | Paper says code will be available upon acceptance; reproducibility remains uncertain as of this review. |
| GaussianPile | Closest slice-based Gaussian renderer and compression baseline ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_GaussianPile_A_Unified_Sparse_Gaussian_Splatting_Framework_for_Slice-based_Volumetric_CVPR_2026_paper.html)). | Accepted primary source; no clearly discoverable official implementation was found in this search, so reimplementation risk is high. |
| MedGS | Sparse-frame medical Gaussian reconstruction/interpolation and shared-geometry multitask baseline ([paper](https://arxiv.org/abs/2509.16806), [code](https://github.com/gmum/MedGS)). | Code exists; environment is Python 3.8, CUDA 12.4, and custom rasterizer extensions. |
| Tri-plane + Gaussian medical reconstruction | Tests whether a simple global field is enough without local SDF machinery ([arXiv](https://arxiv.org/abs/2512.22800)). | Publication/code maturity is uncertain; use as a conceptual reimplementation if code remains absent. |
| Free Gaussian with the same plane operator | Isolates the value of the SDF constraint from the renderer. | Must share initialization, primitive budget, optimizer, and forward model with the proposed method. |
| Dense voxel / compact INR with the same legal observations | Tests whether Gaussians are actually the right memory. | Use matched parameter count and wall-clock budget. |

3DGSMR is a relevant related method, but it consumes undersampled k-space rather than sparse reconstructed planes. It belongs in a separate forward-model track, not the main leaderboard ([paper](https://arxiv.org/abs/2502.06510)).

#### Routing baselines

Every routing method must start from identical anchors and use the same reconstruction/update rule:

1. random;
2. uniform physical spacing;
3. central-first;
4. modality-balanced uniform;
5. fixed learned population trajectory;
6. maximum support-distance / uncovered mass;
7. maximum predictive uncertainty;
8. maximum expected one-step residual reduction;
9. GauSS-MI-style Gaussian mutual information adapted to planes ([RSS 2025](https://www.roboticsproceedings.org/rss21/p030.html));
10. ActiveGAMER-style rendering information gain ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_ActiveGAMER_Active_GAussian_Mapping_through_Efficient_Rendering_CVPR_2025_paper.html));
11. single-wave version of the proposed graph;
12. multi-wave without balancing;
13. full balanced multi-wave;
14. retrospective one-step oracle and, on small candidate sets, beam-search oracle.

The oracle is not a deployable competitor; it measures remaining policy headroom and whether learning routing is worth the complexity.

### 5. Dataset expectations

No single public dataset fully validates the proposed claim. Use at least two tracks.

#### Track A — controlled registered multi-sequence reconstruction

- **BraTS adult glioma** provides the exact common four-sequence tuple \(T1n, T1c, T2w, T2f\) and current preprocessing/tooling expectations ([BraTS documentation](https://brats.readthedocs.io/en/stable/index.html)). It is suitable for controlled retrospective slice-reveal experiments and lesion-sensitive evaluation.
- BraTS is already registered and standardized, so success on it does **not** establish robustness to raw affine, scanner, motion, or registration errors.

#### Track B — external clinical generalization

- **UCSF-PDGM** contains 501 subjects acquired under a standardized 3T preoperative glioma protocol with conventional and advanced sequences ([official TCIA collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)).
- **UPENN-GBM** contains 500 subjects, DICOM and NIfTI access, acquisition-parameter tables, and a CC BY 4.0 license ([official TCIA collection](https://www.cancerimagingarchive.net/collection/upenn-gbm/)).
- **UCSD-PTGBM** includes longitudinal and advanced multiparametric post-treatment data under CC BY 4.0, with subject identifiers that support patient-safe splitting ([dataset paper and TCIA link](https://www.nature.com/articles/s41597-025-06499-z)).

At least one external-site dataset should be test-only. Do not mix slices from the same patient or longitudinal time points across splits.

#### Optional forward-model track

- **fastMRI brain** provides raw multi-coil k-space and official slice-level data contracts ([official repository](https://github.com/facebookresearch/fastMRI), [dataset site](https://fastmri.med.nyu.edu/)). Use it only if the paper makes scanner-side acquisition claims.
- Fetal MRI/FeTA is appropriate if the task pivots to real stack/plane acquisition and motion-aware SVR; FC-SVR and NeSVoR already provide public pipelines for this setting ([FC-SVR code](https://github.com/seannz/svr), [NeSVoR docs](https://nesvor.readthedocs.io/en/latest/)).

### 6. Evaluation expectations

#### Reconstruction quality

Report per modality and macro-average:

- NMSE, PSNR, SSIM, and MS-SSIM;
- MAE in normalized and restored physical/intensity units;
- high-frequency error or gradient error;
- boundary-band error;
- lesion-region error;
- frequency-stratified error;
- arbitrary-plane consistency, not only held-out axial slices.

PSNR/SSIM are necessary for comparability but insufficient. MedGS itself uses a leave-frame-out protocol and PSNR/SSIM for interpolation ([paper](https://arxiv.org/abs/2509.16806)); the proposed paper must go beyond this with clinically sensitive and downstream evaluations.

#### Downstream fidelity

Run a frozen, externally trained segmentation model on real and reconstructed volumes and report:

- whole-tumor, tumor-core, and enhancing-tumor Dice;
- HD95 and surface Dice;
- lesion-wise recall and false-negative lesion count;
- performance difference relative to the original audit volume.

The downstream model must not train on reconstructions from the test patients.

#### Active-routing quality

Use common budgets expressed in three ways:

- absolute number of acquired planes;
- percentage of the legal candidate pool;
- estimated acquisition time/cost, including modality switches.

Report:

- full quality–budget curve and normalized area under it;
- budget to reach 90% and 95% of the full-observation reference;
- actual versus predicted marginal gain correlation and ranking NDCG;
- redundant-plane rate;
- modality allocation and spatial coverage;
- convergence success, premature-stop rate, and insufficient-observation rate;
- policy regret against the retrospective oracle.

One final budget number is not sufficient evidence for an active method.

#### Uncertainty and safety

Report:

- calibration error and reliability diagrams;
- sparsification/error-retention curves and AUSE;
- risk–coverage curves;
- interval coverage where predictive intervals are available;
- lesion-miss rate stratified by predicted uncertainty;
- unsupported-volume fraction.

The uncertainty must be validated against hidden audit error, not merely visualized.

#### Efficiency

CVPR 2026 explicitly introduced compute reporting to encourage efficiency transparency ([author guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)). Report:

- training and inference GPU model, CUDA/PyTorch versions, and precision;
- per-patient initialization time;
- per-query scoring, rendering, encoding, assimilation, topology, and graph-repair time;
- peak training and inference VRAM separately;
- primitive and anchor counts over the budget;
- bytes per primitive and total state size;
- FLOPs/latency of the analytic scaffold and micro-CNN;
- number of runs, seeds, mean, confidence interval, and paired statistical tests.

### 7. Mandatory ablation implications

The full current ablation list is too large for one paper. Organize it into three causal blocks.

#### A. Forward operator and representation

1. trilinear/cubic interpolation;
2. dense voxel or compact INR;
3. free Gaussian + zero-thickness plane evaluation;
4. free Gaussian + measured/analytic PSF;
5. SDF initialization only;
6. persistent SDF constraint;
7. free quaternion/full covariance versus derived normal/tangent covariance;
8. surface-only, volume-only, and dual-bank states.

The first gate is whether an analytic PSF-aware free Gaussian already matches the proposed structural memory. GSVR and GaussianPile make this a necessary test.

#### B. Evidence and sparse-supervision legality

1. raw intensity;
2. analytic differential bank only;
3. raw shallow CNN;
4. analytic bank + micro-CNN;
5. pretrained encoder upper bound;
6. complete-volume privileged-training upper bound;
7. sparse-only main training;
8. deliberate leakage-positive control that proves the audit catches forbidden access.

The main result must include the sparse-manifest hash, exact opened-file ledger, and patient-level split.

#### C. Active policy

1. uniform/random;
2. uncertainty greedy;
3. mutual-information plane score;
4. predicted marginal reconstruction gain;
5. single wave;
6. multi-wave without balance;
7. balanced multi-wave;
8. fixed budget versus calibrated stopping.

Do not add adaptive topology or incremental graph repair until the static-representation active policy beats all greedy baselines. Those mechanisms are efficiency refinements, not prerequisites for the first scientific claim.

### 8. Dependency and tooling maturity

| Tool | Maturity signal | Recommendation |
|---|---|---|
| `gsplat` | Active, Apache-2.0, PyPI-installable, 1,200+ commits, documented tests, batching, N-D features, sparse gradients, and distributed rasterization. Stable release is v1.5.3; main adds CUDA 13/NumPy 2 support and stronger AOT CI ([official repository](https://github.com/nerfstudio-project/gsplat), [docs](https://docs.gsplat.studio/main/), [releases](https://github.com/nerfstudio-project/gsplat/releases)). | Preferred generic Gaussian backend. Pin a commit/release. Its camera rasterizer and alpha compositing are not the MRI plane operator, so keep the medical forward operator separate. |
| Original Graphdeco 3DGS | Canonical reference, but requests CUDA 11.8, compiled submodules, and about 24 GB VRAM for paper-quality training; the official issue tracker contains recurring compiler/CUDA mismatch reports ([repository](https://github.com/graphdeco-inria/gaussian-splatting), [2026 installation issue](https://github.com/graphdeco-inria/gaussian-splatting/issues/1313)). | Use only as a correctness reference, not the project foundation. |
| GSVR code | Modern Python 3.12, PyTorch 2.7, CUDA 12.8, exact PSF, and direct NIfTI output, but only five commits and FAISS-GPU is explicitly documented as brittle ([official repository](https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR)). | Vendor only the mathematical baseline or run it in its own container; do not couple core code to its environment. |
| MedGS code | Public, 35 commits, custom differentiable rasterizer, Python 3.8, CUDA 12.4, and A100-reported experiments ([official repository](https://github.com/gmum/MedGS)). | Containerized comparison. Expect porting effort and verify fallbacks do not silently render the wrong head. |
| NeSVoR | Current documentation, packaged CLI, Docker/source installation, PSF/motion/bias/noise support, and multiple reconstruction modes ([documentation](https://nesvor.readthedocs.io/en/latest/)). | Highest-priority mature INR baseline. |
| MONAI | Active medical-imaging stack; release 1.6.0 is current and includes metrics/transforms fixes ([official releases](https://github.com/Project-MONAI/MONAI/releases)). | Use for well-tested transforms/metrics where helpful, but keep physical-coordinate contracts in the project’s own small core. |

Recommended environment policy:

- keep physical geometry, manifests, ledgers, and analytic CPU reference kernels dependency-light;
- pin a reproducible Linux/CUDA/PyTorch container for GPU work;
- maintain a slow PyTorch/NumPy reference plane renderer and compare it numerically with the CUDA kernel;
- never import an external project as a mutable Git submodule without a pinned commit and license record;
- record GPU architecture because custom splatting kernels are sensitive to CUDA/compiler combinations.

### 9. Community and reproducibility signals

- The generic 3DGS ecosystem is active but fragmented. `gsplat` is the most maintainable base; the original Graphdeco repository still has hundreds of open issues and recurring installation reports, including CUDA/driver/compiler mismatches on 2026 hardware ([Graphdeco issues](https://github.com/graphdeco-inria/gaussian-splatting/issues), [gsplat repository](https://github.com/nerfstudio-project/gsplat)).
- Medical Gaussian code is young. MedGS is public but tied to a narrow environment; GSVR is public but has only a handful of commits; GaussianSVR still promises code only upon acceptance. Claims of “reproducible strongest baseline” must distinguish executed official code from a paper-only reimplementation.
- The GSVR repository explicitly identifies FAISS-GPU as its most brittle dependency and documents CPU fallback ([official README](https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR)). This is a practical warning against putting approximate-neighbor infrastructure on the critical correctness path.
- CVPR’s current guidance emphasizes compute transparency and careful dataset provenance; papers cannot claim a new dataset contribution without making it available by camera-ready time ([CVPR 2026 author guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)).

### 10. Reproducibility and review risks

| Risk | Severity | Required control |
|---|---|---|
| Accidental candidate-image leakage | Critical | OS-level loader separation, immutable commit ledger, opened-file audit, and a leakage-positive test. |
| Invalid “active MRI acquisition” wording | Critical | Specify whether actions are k-space patterns, real prescribed planes/stacks, or retrospective reveals. |
| Full-volume audit contamination | Critical | Separate process/container and serialized reconstruction before audit volume access. |
| Baseline forward-model mismatch | Critical | Same PSF, geometry, observations, starting anchors, intensity normalization, and compute budget. |
| Registration/resampling leakage | High | Fit transforms only from allowed metadata/observations; report use of pre-registered BraTS separately. |
| Patient or timepoint split leakage | High | Grouped subject-level and longitudinal split hashes. |
| Cherry-picked query budgets | High | Predeclare the full budget grid and report curve AUC. |
| Single trajectory seed | High | Multiple seeds/policies per patient and paired confidence intervals. |
| “SDF” without signed-distance evidence | High | Eikonal, sign, and distance calibration tests; otherwise call it a structural level-set field. |
| Unsupported Gaussian kernel correctness | High | Analytic synthetic tests, finite differences, affine equivariance, slab quadrature comparisons, CPU–CUDA parity. |
| Novelty by component aggregation | High | One primary claim and causal ablations showing why every added component is necessary. |
| Only PSNR/SSIM | High | Lesion ROI, downstream segmentation, uncertainty, and failure analysis. |
| Unavailable 2026 baseline code | Medium | Clearly label paper reimplementations and perform sensitivity analysis. |

## Comparison of strategic alternatives

| Alternative | Main claim | Strength | Risk | Recommendation |
|---|---|---|---|---|
| A. Keep the full SDF + dual Gaussian + micro-CNN + multi-wave paper | A unified sparse medical reconstruction system | Ambitious and potentially high impact | Very high novelty collision; too many moving parts; attribution will be weak | Do not lead with this package. Treat it as a long-term system. |
| B. Active observability over a validated PSF-aware Gaussian volume | Leakage-safe calibrated active multi-sequence plane acquisition | Most distinct remaining gap; clean quality–budget evaluation; representation can be controlled | Requires a defensible action model and strong uncertainty/gain calibration | **Recommended CVPR direction.** |
| C. Representation-only SDF-constrained Gaussian MRI | Low-DoF structural manifold improves sparse reconstruction | Easier engineering and static evaluation | Direct GSDF, SurfaceSplat, 3D-SLNR, Scaffold-GS, GaussianPile, GSVR, and MedGS collisions | Viable only as an ISBI/MIDL-style adaptation unless results are unusually strong. |

## Recommendation

1. **Freeze the paper claim before full implementation.** Use Alternative B.
2. **Replace the current renderer novelty claim with a validated forward-model contract.** Implement a slow analytic PSF reference and benchmark against GSVR/GaussianPile formulations.
3. **Build a static reconstruction baseline first.** Free Gaussian, GSVR-style Gaussian, compact INR, and interpolation must be reproducible before routing.
4. **Demonstrate an active-policy oracle gap.** If uniform or uncertainty greedy is already close to the oracle, multi-wave learning is not justified.
5. **Make the first routing contribution analytic and calibrated.** Add learned gain only after proving a legal sparse supervision source.
6. **Use BraTS for controlled development and a TCIA cohort for external validation.**
7. **Treat SDF, dual banks, topology adaptation, and local graph repair as ablations or later work until each clears a measurable gate.**

Go/no-go conditions for a CVPR submission:

- **Go** if the balanced policy improves quality–budget AUC on two datasets, remains better under matched representation/compute, predicts true gain with useful rank correlation, and preserves lesion fidelity with calibrated uncertainty.
- **No-go / re-scope** if gains disappear against GSVR/GaussianPile-style rendering, if the oracle gap is small, or if the query action cannot be defended as a real acquisition or clearly labeled retrospective task.

## External Signals

### Trends

- Slice-based volumetric Gaussian reconstruction is now an established 2025–2026 direction; GaussianPile, GSVR, GaussianSVR, MedGS, and 3DGSMR eliminate broad “first Gaussian medical reconstruction” claims.
- Imaging-physics-aware forward operators, especially exact or differentiable PSF modeling, are becoming baseline expectations rather than optional refinements.
- SDF–Gaussian hybrids and anchor-organized Gaussians are mature enough that the project must show domain-specific value rather than novelty by combination.
- Active Gaussian reconstruction has moved from generic uncertainty to rendering information gain and mutual information; a new policy must beat GauSS-MI/ActiveGAMER-style scores.

### Ecosystem

- `gsplat` is the strongest general backend, but the MRI plane/slab operator remains custom research code.
- NeSVoR is the most mature task-specific INR baseline.
- GSVR and MedGS are executable but young and environment-sensitive; GaussianPile and GaussianSVR reproducibility is still uncertain.
- A pinned CUDA container plus a CPU/PyTorch reference renderer is essential.

### Community

- Official 3DGS repositories show sustained activity but recurring CUDA/compiler friction.
- Current medical Gaussian repositories are small enough that independent numerical validation is necessary.
- CVPR 2026 raises the expectation for compute and dataset-provenance reporting.

### Benchmark and ablation implications

- Add GaussianPile, GSVR, GaussianSVR, MedGS, NeSVoR, FC-SVR, tri-plane Gaussian, free Gaussian, and classical interpolation baselines.
- Separate representation, encoder legality, and routing ablations into causal blocks.
- Report full quality–budget, calibration, lesion fidelity, external-site generalization, and compute curves.
- Reframe the paper around active observability; do not claim slice-aware Gaussian rendering, SDF–Gaussian coupling, anchors, or teacher-free reconstruction as standalone novelties.
