# Point-guided multi-modal brain MRI scaffold

This repository is being rebuilt as a research-grade PyTorch scaffold whose
future research objective is:

```text
T1 + T2 + FLAIR full 3-D volumes -> full T1ce volume
```

The implemented frontend ends after a single shared MedicalNet traversal,
coarse semantics, the frozen refined point field, semantic-aware compact-
support partition of unity, static diagnostic base planes `Bxy/Bxz/Byz`, the
static SWT-Haar spectral anchor `A`, geometry-aware point spectral evidence
`f_spec`, a bounded diagnostic Gate-C dynamic trajectory, and the explicit
Gate-D final-Z-only implicit decoder. `PLAN.md` Phases 1-7, Gate C C1-C7, and
Gate D D1 are complete:
the Phase-2 selected feature map feeds the static B/A branch while the same
traversal continues to the semantic, point, and PoU path.

Gate A, Gate B, Gate C C1-C7, Gate D D1, and Gate E E1-E9 are complete. Gate
E is target-after-inference supervision only: it adds no optimizer, training
loop, generic inference policy, or reconstruction-quality claim. Gate F is
next/inactive; Gate G remains inactive. The narrow `forward_reconstruction()`
endpoint still predicts an absolute scalar volume from final `Z` only.

Start with [the frontend contract](docs/architecture/POINT_GUIDED_FRONTEND.md)
and use the task-scoped [codegraph](CODEGRAPH.json) rather than scanning the
whole repository.

PFGR-Lite CLI/config integration and the Vietnamese R0–R10 execution boundary
are documented in [RUNBOOK_PFGR_LITE.md](RUNBOOK_PFGR_LITE.md). The runbook is
software evidence guidance only; real-data, CUDA, trained-checkpoint and
clinical claims remain review-gated.

Current software handoff (root-reported): the prior W5 service scope was 46
passed in 7.73 s and legacy point-guided was 336 passed/18 skipped in 71.08 s;
the final integrated root run is 977 passed, 18 skipped, 26 subtests, 1 warning
in 103.07 s, with targeted PFGR/acceptance 24 passed and all three
`POINT_GUIDED_FRONTEND` automated checks PASS (Human Gate pending). These are
software scopes only; MedicalNet pretrained,
GPU/AMP, patient-volume, trained-checkpoint, reconstruction, and scientific
headroom evidence remain pending and must not be inferred from synthetic CPU
smoke or historical runs.
