# Software ownership

## Implemented point-guided frontend

`src/smagm/features/point_guided/` owns the current research frontend:

| Area | Owner | Depends on |
| --- | --- | --- |
| configuration and typed volume/point records | `config.py`, `contracts.py` | PyTorch, canonical geometry |
| MedicalNet ResNet10 and semantic prior | `medicalnet_resnet10.py`, `semantic_prior.py` | configuration |
| static diagnostic base planes and spectral anchor | `triplane_projection.py`, `swt_haar.py`, `spectral_anchor.py` | configured selected shared feature and static `BaseTriPlanes` |
| geometry-aware point spectral evidence | `spectral_query.py`, `cross_plane_consistency.py` | typed static `SpectralAnchor`, refined points, deterministic feature-grid geometry |
| deterministic point placement and 3-D sampling | `points.py`, `sampling.py` | contracts |
| directional descriptor and bounded offset prediction | `directional.py`, `offset_predictor.py`, `refinement.py` | sampling |
| semantic/spatial affinity and sparse PoU | `semantic_affinity.py`, `spatial_affinity.py`, `pou.py` | contracts |
| bounded Gate-C trajectory | `state_init.py`, `reward.py`, `trajectory_cost.py`, `trajectory_solver.py`, `updater.py`, `writeback.py`, `trajectory.py` | static B/A and fixed Phase-7 evidence; no target data |
| Gate-D implicit decoder | `decoder.py`, `model.py` | final `Z_K` plus typed geometry only; chunked `96 -> 64 -> 32 -> 1` absolute prediction, no observation bypass |
| Gate-E supervision | `losses.py`, `reward_supervision.py`, `training_objective.py`, `model.py` | target-after-inference Charbonnier/3-D SSIM/DHW-gradient objective, bounded measured reward targets, and trace-local E5–E8 terms; no optimizer or train loop |
| training-only semantic grounding | `semantic_supervision.py` | BraTS `{0,1,2,4}` to normal/edema/core targets, ignore-index masking, CE, and validation-only Dice; never an inference input |
| full-volume point-guided data | `src/smagm/data/brats21_point_guided.py` | additive NIfTI `[X,Y,Z] -> [D,H,W]` adapter, input-derived mask/normalization, geometry validation, and deterministic subject splits; legacy sparse-plane adapter unchanged |
| server training/evaluation | `src/smagm/training/point_guided.py`, `src/smagm/cli/point_guided_train.py`, `src/smagm/cli/point_guided_eval.py` | target-free context then Gate-E objective plus configurable semantic auxiliary loss; AMP/DDP, logs, strict checkpoints, and post-inference metrics |
| public frontend composition and diagnostic output | `model.py`, `contracts.py` | all locked components, `BaseTriPlanes`, `SpectralAnchor`, `PointSpectralEvidence`, optional typed trajectory result |
| Gate-D-and-later type-only contracts | `interfaces.py` | typed frontend records only |

Dependency direction is inward: `model -> locked components -> contracts ->
canonical geometry`. The locked frontend remains isolated from legacy anchors,
fields, memory, routing, reconstruction, and evaluation packages. The additive
server owner imports only the explicit point-guided model/data/checkpoint APIs.

## Implemented PLAN scope through Phase 7, Gate C, Gate D, and Gate E

The following locked engineering boundaries are implemented at the current
revision:

| Phase | Authorized owner | Narrow boundary |
| --- | --- | --- |
| 1-2 | `medicalnet_resnet10.py`, `semantic_prior.py`, `config.py` | one shared pre-MaxPool/Layer1 feature API and explicit freeze/detach/tap ablations; never a second encoder |
| 3 | `config.py`, semantic contracts/tests | exactly `normal brain`, `edema`, `tumor-core candidate` as the production coarse semantic state |
| 4 | `triplane_projection.py` | static feature-only `Bxy/Bxz/Byz` projection from the configured selected shared feature |
| 5 | `model.py`, `contracts.py` | one-pass diagnostic composition of `B`; no anchor `A`, dynamic state, decoder, or T1ce output |
| 6 | `swt_haar.py`, `spectral_anchor.py`, `model.py`, `contracts.py` | fixed two-level SWT-Haar and one shared per-band projection into typed static `Axy/Axz/Ayz` |
| 7 | `spectral_query.py`, `cross_plane_consistency.py`, `model.py`, `contracts.py` | live-operator/full-affine feature-grid geometry, bilinear refined-point queries, deterministic reliability, and XY/XZ/YZ-preserving 168-d `f_spec`; no Gate-C state |
| Gate C C1–C7 | `state_init.py`, `reward.py`, `trajectory_cost.py`, `trajectory_solver.py`, `updater.py`, `writeback.py`, `trajectory.py`, `model.py` | bounded dynamic `Z`: reward-cost route diagnostics and local 4-mm writes; no losses or target data |
| Gate D D1 | `decoder.py`, `model.py` | final-Z-only, full-affine, chunked XY/XZ/YZ query and shared `96 -> 64 -> 32 -> 1` absolute prediction; no observation bypass, loss, or training |
| Gate E E1–E9 | `losses.py`, `reward_supervision.py`, `training_objective.py`, `model.py` | target-after-inference supervision only: bounded local/spill counterfactual measurement and typed objective; no optimizer, scheduler, loop, checkpoint, or Gate F/G policy |

`B` is not a spectral anchor and is not an alternate coordinate system. It
keeps the frontend's `[B, C, D, H, W]`, RAS-mm, `[d, h, w]`, spacing, and
input-affine conventions. Phase 7 derives feature-grid geometry internally
from the live spatial operators and returns typed point spectral evidence
`f_spec`; it does not expose dynamic state.

## Implemented Gate B: Phase 7 point spectral evidence

Gate A is closed and Phase 6 is implemented by the additive point-guided
frontend: fixed
two-level 2-D stationary/undecimated Haar on each static B plane, reflect
padding, seven bands in `LL2,LH1,HL1,HH1,LH2,HL2,HH2` order, and one shared
`Conv2d(64,8,1)` for all bands and planes. This yields the static 56-channel
anchor `Axy/Axz/Ayz`; MAIN has no normalization, with only optional
`GroupNorm(7,56)` as `band_gn`. Haar filters must be fixed PyTorch
buffers/constants, not external wavelet-library components or learned filters.
If reflect padding is invalid for a singleton required plane dimension, the
implementation fails closed rather than changing padding mode. SWT preserves
the B-plane grids for the implemented point query.

Gate B is closed and Phase 7 deterministically derives feature-grid geometry
from the volume geometry and actual spatial transforms, then bilinearly queries
each A plane at refined RAS-mm points. The helper handles rotation, shear,
anisotropic spacing, and translation; it does not hard-code `/2`, learn a
registration/warp, or use target-derived geometry. Phase 7's only fusion is
the fixed 24-d `[LL2,E1,E2]` descriptor, pairwise cosine / mean agreement /
softmax reliability, and XY-XZ-YZ reliability-weighted 168-d concatenation.
It has no confidence MLP, second encoder, cross-attention, hard plane drop,
canonical 104-d fusion, or learned compression.

Gate C C1–C7 is implemented only as bounded dynamic `Z0`/`Z_t`, adaptive
selection, explicit reward costs, local updates, and compact diagnostics.
Gate D D1 is complete only through its explicit reconstruction endpoint.
Gate E is complete only as target-after-inference supervision. The additive
server owner now provides the real full-volume adapter, training loop,
semantic auxiliary, checkpoints, and held-out evaluation boundary without
changing the frontend. Software readiness does not claim F3/F4 execution or
a trained checkpoint.
Authorization never permits reuse
of legacy `anchors`, `fields`, `memory`, `routing`, reconstruction, training,
or data systems.

## Existing code reused only by explicit boundary

`src/smagm/contracts/coordinates.py` remains the RAS-mm reference. The
existing BraTS21 implementation under `src/smagm/data/` is a sparse-plane
preparation path and stays outside the frontend until a separately tested
volume adapter exists. Existing Gaussian/anchor/training code is legacy
research code, not a dependency of the new point-guided model.

## Navigation

`CODEGRAPH.json` is the access-control map. Resolve one task before reading or
editing code with `python scripts/codegraph.py --task <name>`.

## Additive PFGR-Lite frontend

`src/smagm/features/point_guided/pfgr_lite/` is the separately versioned
PFGR-Lite W1-W5 package. W1's typed declarations/configuration/provenance are
target-free and lazy; its `PFGRLiteModel` wraps the legacy frontend without a
trajectory or RewardNet, consumes one private four-result MedicalNet seam, and
produces graph-preserving static B0/B1/B2/B-light `Z0` planes from true
shallow/Layer1/deep feature-cell lattices. `initialize_state` clones Z0
without detach. `decode_final` requires W2's canonical query-lattice
injection and never falls back to the legacy query path. W2 owns query,
footprint, sparse writer, and teacher; W3 owns stages/bank/value; W4 owns
proposals/policy/calibration/checkpoints/inference; W5 owns services,
diagnostics, CLI, and runbook. W1 fix-round software is pending review and
the W2-W5 ledger remains pending implementation/evidence.

Run `python scripts/codegraph.py --task pfgr_lite` for the package navigation
scope; future package paths are not permanently blocked and per-worker
dispatch controls write ownership. W1 fixtures may explicitly set
`engineering_only=True` and a smaller N, while production declares N=2048 and
processes subject batches serially at B=1. Global producer hashes cover frozen
algorithm/component identities; subject affine/shape, resolved observation
mask, and values are context/action/replay identity fields. Passing W1 CPU
tests is software evidence only.
