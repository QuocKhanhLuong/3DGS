# Point-guided MRI frontend

This documentation set replaces the former sparse-Gaussian research direction.
The active work is a modular PyTorch frontend that accepts T1, T2, and FLAIR
volumes and prepares a point field for later T1ce reconstruction research.

- [Frontend contract](architecture/POINT_GUIDED_FRONTEND.md)
- [Point-guided server runbook](POINT_GUIDED_SERVER_RUN.md)
- [Trajectory collapse diagnostic runbook](TRAJECTORY_COLLAPSE_DEBUG_RUN.md)
- [Codegraph and access policy](../CODEGRAPH.json)
- [Software ownership](../CODEBASE.md)

The locked frontend is software infrastructure, not a validated reconstruction
method or a clinical claim.
