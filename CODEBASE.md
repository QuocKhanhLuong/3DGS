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
| future-only contracts | `interfaces.py` | typed frontend records only |

Dependency direction is inward: `model -> locked components -> contracts ->
canonical geometry`. The frontend may not import anchors, fields, memory,
routing, training, reconstruction, evaluation, or CLI packages.

## Completed PLAN scope through Phase 5

The following locked engineering boundaries are implemented at the current
revision:

| Phase | Authorized owner | Narrow boundary |
| --- | --- | --- |
| 1–2 | `medicalnet_resnet10.py`, `semantic_prior.py`, `config.py` | one shared pre-MaxPool/Layer1 feature API and explicit freeze/detach/tap ablations; never a second encoder |
| 3 | `config.py`, semantic contracts/tests | exactly `normal brain`, `edema`, `tumor-core candidate` as the production coarse semantic state |
| 4 | `triplane_projection.py` | static feature-only `Bxy/Bxz/Byz` projection from the configured selected shared feature |
| 5 | `model.py`, `contracts.py` | one-pass diagnostic composition of `B`; no anchor `A`, dynamic state, decoder, or T1ce output |

`B` is not a spectral anchor and is not an alternate coordinate system. It
inherits the frontend's `[B, C, D, H, W]`, RAS-mm, `[d, h, w]`, spacing, and
affine contracts. Wavelet anchor `A`, cross-plane fusion, dynamic tri-planes,
trajectory, stopping, decoding, and reconstruction remain interface-only and
research-gated.

## Existing code reused only by explicit boundary

`src/smagm/contracts/coordinates.py` remains the RAS-mm reference. The
existing BraTS21 implementation under `src/smagm/data/` is a sparse-plane
preparation path and stays outside the frontend until a separately tested
volume adapter exists. Existing Gaussian/anchor/training code is legacy
research code, not a dependency of the new point-guided model.

## Navigation

`CODEGRAPH.json` is the access-control map. Resolve one task before reading or
editing code with `python scripts/codegraph.py --task <name>`.
