# Software ownership

## Active point-guided frontend

`src/smagm/features/point_guided/` owns the current research frontend:

| Area | Owner | Depends on |
| --- | --- | --- |
| configuration and typed volume/point records | `config.py`, `contracts.py` | PyTorch, canonical geometry |
| MedicalNet ResNet10 and semantic prior | `medicalnet_resnet10.py`, `semantic_prior.py` | configuration |
| deterministic point placement and 3-D sampling | `points.py`, `sampling.py` | contracts |
| directional descriptor and bounded offset prediction | `directional.py`, `offset_predictor.py`, `refinement.py` | sampling |
| semantic/spatial affinity and sparse PoU | `semantic_affinity.py`, `spatial_affinity.py`, `pou.py` | contracts |
| public frontend composition | `model.py` | all locked components |
| future-only contracts | `interfaces.py` | typed frontend records only |

Dependency direction is inward: `model -> locked components -> contracts ->
canonical geometry`. The frontend may not import anchors, fields, memory,
routing, training, reconstruction, evaluation, or CLI packages.

## Existing code reused only by explicit boundary

`src/smagm/contracts/coordinates.py` remains the RAS-mm reference. The
existing BraTS21 implementation under `src/smagm/data/` is a sparse-plane
preparation path and stays outside the frontend until a separately tested
volume adapter exists. Existing Gaussian/anchor/training code is legacy
research code, not a dependency of the new point-guided model.

## Navigation

`CODEGRAPH.json` is the access-control map. Resolve one task before reading or
editing code with `python scripts/codegraph.py --task <name>`.
