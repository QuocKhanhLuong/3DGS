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

## Independent W5a evidence packaging

W5a may also implement the standard-library evidence packaging helper while
W1/W2a work. Exclusive ownership is
`src/smagm/features/point_guided/pfgr_lite/artifacts.py` and
`tests/features/point_guided/pfgr_lite/test_artifacts.py`. Its stable public
interface is `package_evidence(run_dirs, destination) -> dict`, where input
directories are existing run artifacts and `destination` is a new directory
that must not already exist. The returned manifest describes a whitelist
archive, included-file hashes and exclusions. It does not consume PFGR model,
action, bank tensors or calibration types. The later W5 CLI wraps this helper
with the already frozen `package --run-dir ... --output-root ... --run-name`
command. Only reviewed metadata, metrics, tests and traceback evidence may be
included; patient volumes, predictions, checkpoints, secrets and raw target
banks remain excluded. Reject symlink escapes, oversized/unknown payloads and
existing destinations. No CLI/service completion is claimed by this helper.

W5a has no commit/staging permission until the principal grants a serialized
slot. W1, W2a and W5a have disjoint owned paths; action/teacher/policy/training
integration remains gated on corrected W1 acceptance.

## Accepted W1 and independent W3a cached fitting

W1 is accepted through `9f03660` after independent semantic checks, 38 final
owned tests and the recorded legacy golden comparison. W2 teacher and W4
policy may now proceed on their separate files. W3a may implement only
`value_bank.py`, `value_net.py`, `test_value_bank.py` and `test_value_fit.py`
against the frozen W1 descriptor, GainLabel, producer, manifest and value-fit
declarations. Its input is already measured, detached state/action metadata;
it must not import or invoke an MRI loader, updater, decoder or teacher.
Bank generation from MRI and S0/S1/S2 orchestration still wait for W2/W4.
This is an explicit scheduling amendment, not a change to the bank population,
scale, regression objective or staged scientific acceptance. All shared-file
changes require principal coordination; commits remain serialized.
