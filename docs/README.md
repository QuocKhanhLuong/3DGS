# Point-guided MRI frontend

This documentation set replaces the former sparse-Gaussian research direction.
The active work is a modular PyTorch frontend that accepts T1, T2, and FLAIR
volumes and prepares a point field for later T1ce reconstruction research.

- [Frontend contract](architecture/POINT_GUIDED_FRONTEND.md)
- [PFGR-Lite R0–R10 CLI runbook](../RUNBOOK_PFGR_LITE.md)
- [Point-guided server runbook](POINT_GUIDED_SERVER_RUN.md)
- [Trajectory collapse diagnostic runbook](TRAJECTORY_COLLAPSE_DEBUG_RUN.md)
- [Reward-logic smoke rerun playbook](POINT_GUIDED_REWARD_LOGIC_SMOKE_RUN.md)
- [Codegraph and access policy](../CODEGRAPH.json)
- [Software ownership](../CODEBASE.md)

The locked frontend is software infrastructure, not a validated reconstruction
method or a clinical claim.

PFGR-Lite handoff status is software-only: the prior W5 service scope was 46
passed in 7.73 s and legacy point-guided was 336 passed/18 skipped in 71.08 s;
the final integrated root run is 977 passed, 18 skipped, 26 subtests, 1 warning
in 103.07 s, with targeted PFGR/acceptance 24 passed and all three
`POINT_GUIDED_FRONTEND` automated checks PASS (Human Gate pending). CUDA/AMP,
pretrained MedicalNet, patient volumes, trained checkpoints, reconstruction
quality, and scientific headroom are not evidenced by local synthetic CPU
checks.
