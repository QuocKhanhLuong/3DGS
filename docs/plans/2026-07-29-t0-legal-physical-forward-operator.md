# T0 implementation plan — Legal Physical Forward Operator

Date: 2026-07-29  
Status: implemented for QA contract coverage

## Scope and contracts

T0 establishes a CPU-first PyTorch reference package, `smagm`, for
retrospective budgeted progressive observations.  Its canonical patient frame
is RAS millimetres; homogeneous matrices multiply column vectors. Plane images
are `[H,W]`, where a pixel `[v,u]` is the physical centre
`origin + u * spacing_u * axis_u + v * spacing_v * axis_v`.

- `contracts.coordinates` freezes source-convention provenance, a source
  plane-index affine, physical planes, and target grids.  A DICOM LPS affine is
  canonicalized to RAS, while the independent affine slice axis validates the
  signed plane normal.
- `contracts.observation` freezes metadata/manifests, prevents a patient from
  spanning data splits within or across manifests, binds payloads by SHA-256,
  keeps raw file access and integrity digests private, permits context
  immediately, and requires commit then a single-use capability-based reveal
  for targets. Digests are excluded from public metadata and the canonical
  pre-commit manifest hash. Deterministic commit/reveal/open events and
  content-addressed reads feed the audit hash. Non-synthetic observations
  require source-affine provenance; budget arithmetic is exact in decimal
  representation.
- `GaussianBatch` accepts differentiable tensors directly: centers `[N,3]`,
  lower-triangular positive-diagonal factors `[N,3,3]`, log amplitudes `[N,1]`,
  and modality appearance/validity `[N,M]`.
- `render_plane` uses `solve_triangular`, never an explicit covariance inverse,
  and returns normalized additive intensity, support mass, and a separate
  unsupported mask plus the supported PSF mass. `SlabProfile.delta()` is the
  analytic thin reference; `box(n)` and `discrete(offsets, weights)` provide
  normalized slab quadrature.
  The slab operator evaluates the normalized latent intensity `N/S` at every
  depth, then applies the through-plane PSF weights; it does not density-weight
  depths by dividing integrated numerator by integrated support.

## Differentiability and discrete boundary

The renderer preserves gradients from the selected image through normalized
additive weights to Gaussian centers, covariance factors, log support
amplitudes, and appearance. Pixel/Gaussian chunking occurs only within these
tensor operations. Geometry validation, manifests, metadata, commit/reveal
capability issuance, file I/O, selection, and masks are intentionally discrete
boundaries. Gaussian tensors are revalidated before every render so an in-place
optimizer update cannot silently violate the public tensor contract.
Unsupported intensity is `NaN`, never a confident zero; consume the returned
mask in a loss or evaluation protocol. Slab support is classified using
weighted PSF coverage and a named minimum-coverage policy; supported samples
are renormalized for intensity. `support_mass` is an observability
diagnostic, not calibrated uncertainty, and must not be used to exclude errors
from headline evaluation.

## Exact out of scope

T0 does not implement a learned encoder, structural field/SDF, topology
operations, routing, full-volume export, analytic finite-slab convolution,
scanner-specific PSF estimation, medical-file adapters, motion/bias/outlier
models, spatial acceleration, custom CUDA, `gsplat`, camera alpha compositing,
or clinical/scanner-side acquisition claims.

## Verification handoff

QA covers analytic thin, delta/slab convergence, affine/signed-normal,
legality, chunking, and float64 gradcheck behavior. The verified CPU reference
environment is recorded in
`requirements/t0-cpu-macos-arm64-py310.lock`; future Linux/CUDA environments
must receive their own platform-specific lock and forward/backward parity gate.
