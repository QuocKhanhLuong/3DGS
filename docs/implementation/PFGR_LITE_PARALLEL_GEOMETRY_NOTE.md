# PFGR-Lite bounded geometry scheduling amendment

Principal decision, 2026-09-07: the accepted plan and interface companion remain
scientifically unchanged. W1 initial commit `461336e` is under a blocking fix
review. To make independent progress, W2a may implement only the canonical
query lattice and geometry tests while the Luna Max W1 worker fixes shared
declarations. This explicitly amends the original strictly sequential W1→W2
scheduling sentence; it does not accept W1 or authorize teacher/policy
integration before the shared contracts pass review.

W2a exclusively owns:

- `src/smagm/features/point_guided/pfgr_lite/footprint.py`
- `tests/features/point_guided/pfgr_lite/test_footprint.py`

The already accepted dependency-free interface is
`PFGRQueryLattice.build(output_geometry, feature_geometry, *, query_dtype,
build_chunk_size)` and `query(planes, voxel_ids_dhw, *, chunk_size) -> [Q,96]`
for an explicitly enforced single-subject plane batch. It consumes existing
`VolumeGeometry`, `FeatureGridGeometry`, `DynamicTriPlanes`, integer voxel
centres and plane tensors; it does not depend on the changing PFGR action,
trace, bank or calibration declarations. Candidate batching remains separate
from subject batching.

Implement the companion's exact four-neighbour bilinear stencil, geometry and
dtype identity, positive support membership, cached inverse node-to-voxel
index where the declared memory bound permits, and an explicitly counted
exact full-scan fallback otherwise. Test against the existing independent
query on rotated, sheared, translated, anisotropic and boundary fixtures.
Preserve FP64 atol=1e-10/rtol=1e-9 and FP32 atol=1e-6/rtol=1e-5. No numerical
tolerance relaxation or performance claim follows from this scheduling note.

Action-consuming footprint construction, sparse effect measurement and the
detached teacher wait for the corrected W1 handoff. W2a must not edit W1
files, declare competing shared types, or commit while another worker can
stage changes. The principal serializes explicit-path commits and reviews
both workstreams before integration. All implementation workers remain
`gpt-5.6-luna` with `max` reasoning; the principal remains Astra High.
