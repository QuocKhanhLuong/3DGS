# Point-guided multi-modal brain MRI scaffold

This repository is being rebuilt as a research-grade PyTorch scaffold for:

```text
T1 + T2 + FLAIR full 3-D volumes -> full T1ce volume
```

Only the frontend through the frozen refined point field and semantic-aware
compact-support partition of unity is currently implemented. The trajectory
and decoder are deliberately interfaces, so this code does not yet produce a
T1ce reconstruction.

Start with [the frontend contract](docs/architecture/POINT_GUIDED_FRONTEND.md)
and use the task-scoped [codegraph](CODEGRAPH.json) rather than scanning the
whole repository.
