# S11 BraTS21 real-data smoke

This is a software and execution smoke for one retrospective sparse derivative
of one BraTS21 patient. It is not a permanently sparse acquisition study and
does not establish reconstruction quality, superiority, clinical validity, or
any Human Gate decision.

## Data boundary

`scripts/data/inspect_brats21.py` validates patient names, exact `t1`, `t1ce`,
`t2`, `flair`, and optional `seg` suffixes, three-dimensional finite data,
resolved qform/sform affine geometry, and BraTS labels. It fails closed on a
malformed patient. `scripts/data/prepare_brats21_smoke.py` reads only the
selected source planes, writes small rank-2 NumPy payloads, and records source
and derivative hashes. The dense NIfTI root is never copied into the prepared
bundle. Segmentation is evaluator-only.

The deterministic initial episode contains four context planes (`t1`, `t1ce`,
`t2`, and one FLAIR) and one disjoint held-out FLAIR target. The maintained
ledger decoder receives only manifest-bound payload bytes. Target bytes are
opened after prediction registration and receipt creation.

## Execution

The runner executes R0 interpolation and E2 + R4 + P0 for two to five small
optimizer steps, serializes a target-plane prediction package, evaluates it
with an explicitly unsealed diagnostic external target file, and audits the
package. W&B is optional and receives only sanitized scalar/hash metadata.

The config requests CUDA. If the host has no usable driver, CPU execution must
be requested explicitly with `--allow-cpu-fallback`; this is recorded in the
resolved run metadata. The run reports the requested device, actual device,
peak CUDA memory when applicable, explicit support masks, gradient norms, and
all artifact paths.

No T4 routing or adaptive acquisition is present. No scientific PASS is
generated automatically; T2/T3/T5 Human Gates remain separate.
