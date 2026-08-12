# Point-guided multi-modal brain MRI scaffold

This repository is being rebuilt as a research-grade PyTorch scaffold whose
future research objective is:

```text
T1 + T2 + FLAIR full 3-D volumes -> full T1ce volume
```

The implemented frontend ends after a single shared MedicalNet traversal,
coarse semantics, the frozen refined point field, semantic-aware compact-
support partition of unity, and static diagnostic base planes `Bxy/Bxz/Byz`.
`PLAN.md` Phases 1–5 are complete: the Phase-2 selected feature map feeds the
static projector while the same traversal continues to the semantic, point,
and PoU path. Wavelet anchor `A`, cross-plane fusion, dynamic trajectory, and
decoder remain research-gated interfaces, so this code does not produce a T1ce
reconstruction.

Start with [the frontend contract](docs/architecture/POINT_GUIDED_FRONTEND.md)
and use the task-scoped [codegraph](CODEGRAPH.json) rather than scanning the
whole repository.
