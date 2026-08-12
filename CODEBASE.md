# Software ownership

## Implemented point-guided frontend

`src/smagm/features/point_guided/` owns the current research frontend:

| Area | Owner | Depends on |
| --- | --- | --- |
| configuration and typed volume/point records | `config.py`, `contracts.py` | PyTorch, canonical geometry |
| MedicalNet ResNet10 and semantic prior | `medicalnet_resnet10.py`, `semantic_prior.py` | configuration |
| static diagnostic base planes | `triplane_projection.py` | configured selected shared feature |
| deterministic point placement and 3-D sampling | `points.py`, `sampling.py` | contracts |
| directional descriptor and bounded offset prediction | `directional.py`, `offset_predictor.py`, `refinement.py` | sampling |
| semantic/spatial affinity and sparse PoU | `semantic_affinity.py`, `spatial_affinity.py`, `pou.py` | contracts |
| public frontend composition and diagnostic output | `model.py`, `contracts.py` | all locked components, `BaseTriPlanes` |
| Gate-C future-only contracts | `interfaces.py` | typed frontend records only |

Dependency direction is inward: `model -> locked components -> contracts ->
canonical geometry`. The frontend may not import anchors, fields, memory,
routing, training, reconstruction, evaluation, or CLI packages.

## Implemented PLAN scope through Phase 5

The following locked engineering boundaries are implemented at the current
revision:

| Phase | Authorized owner | Narrow boundary |
| --- | --- | --- |
| 1-2 | `medicalnet_resnet10.py`, `semantic_prior.py`, `config.py` | one shared pre-MaxPool/Layer1 feature API and explicit freeze/detach/tap ablations; never a second encoder |
| 3 | `config.py`, semantic contracts/tests | exactly `normal brain`, `edema`, `tumor-core candidate` as the production coarse semantic state |
| 4 | `triplane_projection.py` | static feature-only `Bxy/Bxz/Byz` projection from the configured selected shared feature |
| 5 | `model.py`, `contracts.py` | one-pass diagnostic composition of `B`; no anchor `A`, dynamic state, decoder, or T1ce output |

`B` is not a spectral anchor and is not an alternate coordinate system. It
keeps the frontend's `[B, C, D, H, W]`, RAS-mm, `[d, h, w]`, spacing, and
input-affine conventions, but it does not yet expose a derived feature-grid
geometry/provenance record. The current executable output does not contain
`A` or `f_spec`.

## Authorized next scope: Phases 6 and 7, not implemented

Gate A is closed and Phase 6 is authorized, but no Phase 6 module exists yet.
Its only authorized owner is the additive point-guided frontend: fixed
two-level 2-D stationary/undecimated Haar on each static B plane, reflect
padding, seven bands in `LL2,LH1,HL1,HH1,LH2,HL2,HH2` order, and one shared
`Conv2d(64,8,1)` for all bands and planes. This yields the static 56-channel
future anchor `Axy/Axz/Ayz`; MAIN has no normalization, with only optional
`GroupNorm(7,56)` as `band_gn`. Haar filters must be fixed PyTorch
buffers/constants, not external wavelet-library components or learned filters.
If reflect padding is invalid for a singleton required plane dimension, the
future implementation must fail closed rather than change padding mode.

Gate B is closed and Phase 7 is authorized, but no Phase 7 module exists yet.
It may add deterministic feature-grid geometry bookkeeping derived from the
volume geometry and actual spatial transforms, then geometry-aware bilinear
query of a refined RAS-mm point from all three A planes. The helper must handle
rotation, shear, anisotropic spacing, and translation; it must not hard-code
`/2`, learn a registration/warp, or use target-derived geometry. Phase 7's
only fusion is the fixed 24-d `[LL2,E1,E2]` descriptor, pairwise cosine / mean
agreement / softmax reliability, and XY-XZ-YZ reliability-weighted 168-d
concatenation. It does not authorize a confidence MLP, a second encoder,
cross-attention, a hard plane drop, canonical 104-d fusion, or learned
compression.

Gate C remains blocked and default-deny: dynamic `Z0`/`Z_t`, selection,
routing, updates, history, stopping, decoding, losses, training, and T1ce
synthesis are not owners of this frontend. Authorization never permits reuse
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
