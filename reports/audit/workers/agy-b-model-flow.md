# AGY-B — Frontend, Trajectory, Decoder, and Target-Boundary Audit Report

## 1. Audit metadata

- **Lane:** AGY-B, model-flow and target-boundary correctness.
- **Repository:** `/Users/alvinluong/3DGS`.
- **Frozen HEAD:** `0efeb94af72ffa067769e19afcd19ad358feefd2` (`main`).
- **Declared authority:** `reports/audit/01-sol-plan.md:191-209`.
- **Read scope:** the `frontend`, `trajectory`, `decoder`, `supervision`, and
  `baseline_inference` CodeGraph routers, plus their declared focused tests.
- **Mutation scope:** this report only. No production code, tests, configs,
  plans, or existing reports were modified.
- **Pre-existing worktree state:** modified `.DS_Store` files and untracked
  audit reports, the architecture HTML artifact, and other `.DS_Store` files
  were preserved. The pre-existing point-guided pytest process was observed
  but not stopped or altered.

## 2. Executive summary

The normal locked path is coherent at the observation and target boundary:

```text
[B,3,D,H,W] observation
  -> one MedicalNet intermediate-feature traversal
  -> coarse semantics, deterministic points, bounded refinement, sparse PoU
  -> B -> fixed A -> q/reliability/f_spec
  -> bounded Gate-C route and final dynamic Z
  -> Gate-D decoder over final Z only
  -> Gate-E target-after-inference objective
```

The public frontend, trajectory, decoder, and Gate-G inference signatures do
not accept T1ce, and the generic `forward()` remains fail-closed. I found no
target-derived route, decoder-input, or frontend computation in the audited
normal path.

Four boundary defects remain:

| ID | Severity | Result | Evidence |
| --- | --- | --- | --- |
| `AGY-B-FIND-001` | P1 | A malformed strict checkpoint can partially mutate the live model before `load_state_dict` raises. | Direct temporary-checkpoint reproduction. |
| `AGY-B-FIND-002` | P2 | `compute_training_objective()` does not verify that the Gate-E context belongs to the receiving model. | Direct source-path proof; no ownership test exists. |
| `AGY-B-FIND-003` | P2 | Gate-G wrapper restores only the parent training flag and can change pre-existing child modes. | Direct synthetic mode-restoration reproduction. |
| `AGY-B-FIND-004` | P3 | A floating physical-point dtype mismatch reaches the decoder MLP as an untyped matmul error. | Direct synthetic decoder reproduction. |

These are software-contract findings only. No trained checkpoint, real GPU
run, reconstruction result, clinical claim, or held-out Gate-G result is
claimed.

## 3. Verified model-flow trace

### 3.1 Observation through Phase 1–7 frontend

`PointGuidedMRIModel._forward_frontend_with_gate_b_context()` validates the
rank-5 three-channel observation and constructs `VolumeGeometry` at
`src/smagm/features/point_guided/model.py:129-130`. It then calls
`SemanticPrior.extract_intermediate_features(x)` once at line 131. The same
returned `MedicalNetFeatures` object feeds the semantic head at lines 132-135
and the configured spectral tap at lines 136-142. The MedicalNet implementation
itself computes conv1, layer1, and the deeper maps in one
`forward_intermediate_features()` traversal
(`src/smagm/features/point_guided/medicalnet_resnet10.py:232-253`).

The rest of the frontend is a single ordered composition:

- Static feature-only base planes are built at `model.py:143`, then the fixed
  spectral anchor is built at line 144.
- Coarse semantics are checked for a per-voxel simplex at `model.py:145-147`.
- Initial points use only geometry, batch/device/dtype, and the optional mask
  (`model.py:148-154`); refinement receives the observation and coarse
  semantics but not a target (`model.py:155`).
- Sparse semantic-aware PoU support is built at `model.py:156-161`.
- Refined RAS-mm points query the three A planes and generate the typed
  `f_spec`/reliability evidence at `model.py:162-175`.
- `FrontendOutput` packages the geometry, B, A, point evidence, and point
  field at `model.py:176-187`.

The tensor contracts remain aligned with the locked authority: `VolumeGeometry`
keeps the full affine and `[d,h,w]` tensor convention
(`src/smagm/features/point_guided/contracts.py:80-125`), the point field keeps
the original-relative displacement and 2-mm bound
(`contracts.py:156-194`), and `PointSpectralEvidence` retains `[B,N,168]`
with three reliability weights (`contracts.py:292-329`). The source reads
showed no T1ce, target, ground-truth, or target-derived value entering this
frontend.

### 3.2 Gate-C trajectory

`forward_trajectory()` invokes the shared frontend helper once and passes only
typed B, refined points, point semantics, `f_spec`, reliability, Gate-B q
descriptors, feature geometry, and source geometry into the trajectory
(`src/smagm/features/point_guided/model.py:216-244`).

Within `AdaptiveRewardCostTrajectory._run()`:

- The input widths, batch/device/dtype alignment, and geometry provenance are
  checked at `src/smagm/features/point_guided/trajectory.py:242-258`.
- C1 creates the dynamic state from B at line 260.
- The parameter-free dynamic query and 126-wide RewardNet descriptor are built
  at lines 319-332; the descriptor source is the current dynamic state,
  point semantics, reliability-weighted q, and reliability
  (`src/smagm/features/point_guided/reward.py:130-174`).
- Travel, overlap, and step costs are combined with reward at
  `trajectory.py:333-342`. Availability is applied only after the dense
  candidate computation at lines 313-318 and 363-373.
- The selected update consumes dynamic features, the full 168-d `f_spec`, point
  semantics, and reliability at `trajectory.py:394-408`; the physical write
  back and compact overlap state follow at lines 434-467.
- The returned `TrajectoryResult` stores the final state and compact
  diagnostics at `trajectory.py:469-488`.

The route has no target parameter or target access. Gate-G supplies the
separate exact-no-revisit policy and evaluates the same route in eval/no-grad
mode (`src/smagm/features/point_guided/baseline_inference.py:202-231`).

### 3.3 Gate-D final-Z-only decoder

`forward_reconstruction()` runs the shared frontend and one trajectory, then
passes only `trajectory.final_state` to the decoder
(`src/smagm/features/point_guided/model.py:246-287`). The observation, B, A,
and spectral evidence are not decoder arguments.

The decoder validates the final dynamic plane grids and state/parameter
device/dtype at `src/smagm/features/point_guided/decoder.py:90-140`, performs
the shared XY/XZ/YZ point query, and applies the locked MLP at lines 140-144.
Dense decoding creates bounded voxel chunks and invokes that same point API
only after final Z exists (`decoder.py:168-202`). No target, reconstruction
loss, or observation bypass was found in the decoder API.

### 3.4 Gate-E target-after-inference supervision

`forward_training_context()` has no target parameter. It builds the shared
frontend, invokes the private target-free training trace, decodes the trace's
final state, and returns a typed context
(`src/smagm/features/point_guided/model.py:338-396`). The first target-bearing
model API is `compute_training_objective(context, target_t1ce, ...)` at
`model.py:419-438`.

The objective implementation type-checks the context before taking its private
trajectory/decoder at `src/smagm/features/point_guided/training_objective.py:291-309`.
Target validation and the reconstruction loss begin at lines 312-321. The
counterfactual reward supervision receives target data only after the existing
trace step has selected its candidate (`training_objective.py:342-374`); its
candidate descriptor and RewardNet prediction are target-free at
`src/smagm/features/point_guided/reward_supervision.py:641-669`, while measured
target transitions and local/spill decoding are under `torch.no_grad()` at
lines 677-735. This preserves the intended target-after-inference ordering.

### 3.5 Gate-F/G inference and checkpoint seams

The Gate-G model wrapper explicitly rejects construction without a trajectory
and documents a target-free eval/no-grad, one-traversal, one-final-Z decode at
`src/smagm/features/point_guided/model.py:298-310`. It enters `self.eval()` and
`torch.no_grad()` before the shared frontend and restores the parent mode in a
`finally` block at lines 311-336. The child-mode limitation is recorded as
`AGY-B-FIND-003` below.

`run_baseline_inference()` requires eval-mode trajectory/decoder and disabled
gradients, applies `ExactNoRevisitPolicy`, and invokes the decoder once on
`route.final_state` (`src/smagm/features/point_guided/baseline_inference.py:209-239`).
The result exposes route and prediction diagnostics without introducing a
target input (`baseline_inference.py:243-258`).

Checkpoint metadata records the schema, canonical model/trajectory configs,
decoder architecture, and Gate-E architecture at
`baseline_inference.py:269-282`. Exact metadata and tensor-valued state-dict
validation happen at lines 293-304. The final strict-load mutation behavior is
unsafe on failure and is recorded separately as `AGY-B-FIND-001`.

## 4. Focused verification

### 4.1 Completed router and test checks

The five declared read-only routers completed:

```text
python scripts/codegraph.py --task frontend
python scripts/codegraph.py --task trajectory
python scripts/codegraph.py --task decoder
python scripts/codegraph.py --task supervision
python scripts/codegraph.py --task baseline_inference
```

The smallest boundary suite was run after the source trace:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/features/point_guided/test_frontend_boundaries.py
15 passed, 2 skipped in 247.02s (0:04:07)
```

The prior broader point-guided pytest process launched before quota remained
untouched; this report does not treat its still-running state as a test result.

### 4.2 Read-only synthetic revalidations

1. **Checkpoint partial-load probe:** a temporary checkpoint copied the live
   state dict, changed one valid RewardNet bias to `123.0`, and truncated a
   later weight tensor. `load_validated_baseline_checkpoint()` raised a shape
   mismatch, but the live RewardNet bias was `123.0` afterward. This directly
   reproduces `AGY-B-FIND-001`; the temporary file was outside the repository
   and removed by the probe.
2. **Mode restoration probe:** with the parent model in training mode but both
   trajectory and decoder deliberately in eval mode, a monkeypatched frontend
   raised after the wrapper entered its `try` block. Modes changed from
   `(model=True, trajectory=False, decoder=False)` to
   `(True, True, True)`, reproducing `AGY-B-FIND-003` through the `finally`
   path.
3. **Decoder dtype probe:** a valid float32 `DynamicTriPlanes` state and
   float64 RAS-mm points reached `ImplicitTriPlaneDecoder.decode_points()`.
   The call raised the raw
   `RuntimeError: mat1 and mat2 must have the same dtype, but got Double and
   Float`, reproducing `AGY-B-FIND-004`.

`AGY-B-FIND-002` is a direct static ownership proof: the receiver checks only
its own module presence, while `_compute_training_objective()` consumes the
modules stored inside the supplied context. The existing focused tests cover
same-model contexts but do not construct a cross-model context.

## 5. Findings

### [AGY-B-FIND-001] (Severity: P1) Strict checkpoint failure can partially mutate the live model

- **Component:** `src/smagm/features/point_guided/baseline_inference.py:285-304`.
- **Description:** The loader validates path, top-level keys, exact metadata,
  and tensor-valued state entries, then calls `model.load_state_dict(state_dict,
  strict=True)` directly. Strictness causes an exception for a bad shape, but
  it does not make the mutation transactional.
- **Code evidence:** `baseline_inference.py:293-303` completes validation;
  line 304 directly loads into the supplied live module. There is no shape/dtype
  preflight against `model.state_dict()`, staging clone, or rollback path.
- **Reproduction:** The temporary malformed checkpoint probe raised the
  expected shape-mismatch `RuntimeError` while changing the valid
  `trajectory.reward_net.network.0.bias` to `123.0` before returning control to
  the caller.
- **Impact:** A caller can catch the loader error and accidentally continue
  with a hybrid of old and checkpoint parameters. The externally visible
  failure therefore does not leave the model in its pre-load state, violating
  the fail-closed Gate-F/G checkpoint boundary and potentially contaminating a
  later inference or training call.
- **Recommendation:** Preflight every expected name, shape, dtype, and device
  before copying, or load into a separately constructed compatible module and
  swap only after the complete load succeeds. A rollback snapshot is a weaker
  fallback. Keep the current strict metadata checks and do not label a failed
  load as a usable checkpoint.

### [AGY-B-FIND-002] (Severity: P2) Gate-E objective accepts a context from another model instance

- **Component:** `src/smagm/features/point_guided/model.py:419-438` and
  `src/smagm/features/point_guided/training_objective.py:131-165,291-309`.
- **Description:** `PointGuidedMRIModel.compute_training_objective()` checks
  only that the receiving model has its own trajectory and decoder, then passes
  the supplied context to `_compute_training_objective()`. The objective uses
  `context._trajectory` and `context._decoder`, not `self.trajectory` or
  `self.decoder`.
- **Code evidence:** The receiver guard is only `model.py:430-431`; the call
  forwards the context unchanged at lines 432-438. The context validates that
  its private trace, trajectory, and decoder are internally paired
  (`training_objective.py:149-168`), but it stores no producing-model identity.
  The objective takes those context-owned modules at lines 305-308.
- **Reproduction / evidence status:** Direct source-path proof. A valid context
  produced by model A can therefore pass the type checks when supplied to model
  B; the objective and its gradients continue through A's stored decoder and
  trajectory. The focused suite tests repeated same-model contexts but has no
  cross-instance ownership case.
- **Impact:** A training caller can invoke model B's public objective API while
  optimizing model A's context-owned modules. The loss may appear valid while
  model B receives no expected gradients, violating the Gate-E/F module and
  optimizer ownership boundary.
- **Recommendation:** Bind a context to an owner token/reference and require
  `context._trajectory is self.trajectory` and `context._decoder is self.decoder`
  before accepting the target. A typed context constructor check alone is not
  sufficient because two model instances use the same concrete classes.

### [AGY-B-FIND-003] (Severity: P2) Gate-G wrapper does not restore pre-existing child training modes

- **Component:** `src/smagm/features/point_guided/model.py:311-336`.
- **Description:** The wrapper snapshots only the aggregate `self.training`
  boolean. `self.eval()` recursively sets every child to eval mode, and the
  `finally` block calls `self.train(was_training)`, which recursively sets every
  child to the aggregate mode rather than restoring each child's prior flag.
- **Code evidence:** The single snapshot is at `model.py:311`, the recursive
  mode change is at lines 313-314, and recursive restoration is line 336.
  Trajectory behavior depends on its own `self.training` flag when selecting
  hard versus straight-through weights (`src/smagm/features/point_guided/trajectory.py:377-381`;
  `trajectory_solver.py:61-69`).
- **Reproduction:** The synthetic probe started with
  `(model=True, trajectory=False, decoder=False)`, forced the frontend to raise,
  and observed `(True, True, True)` after the `finally` block. The same
  restoration code executes after a successful inference call.
- **Impact:** A caller that intentionally keeps only selected modules in eval
  mode can have those semantics silently changed after Gate-G inference. In
  particular, a trajectory that was intentionally eval/hard can become
  training/straight-through on the next call while the parent remains in the
  same aggregate mode.
- **Recommendation:** Snapshot and restore every relevant submodule's training
  flag, or use a scoped eval helper that records module-local state. Preserve
  the existing no-grad and target-free behavior.

### [AGY-B-FIND-004] (Severity: P3) Decoder point dtype mismatch fails as a raw matmul error

- **Component:** `src/smagm/features/point_guided/reward.py:102-127` and
  `src/smagm/features/point_guided/decoder.py:121-144`.
- **Description:** `DynamicStatePointQuery` checks point floating-ness, batch,
  and device, but not point dtype against the dynamic state at
  `reward.py:115-117`. Its sampler intentionally retains the query dtype and
  converts the state to it at `reward.py:38-44`. `decode_points()` checks state
  versus MLP dtype at `decoder.py:136-140`, but does not check
  `points_ras_mm` before calling the query at line 140.
- **Reproduction:** Float32 state plus float64 physical points produced a
  float64 query and then raised
  `RuntimeError: mat1 and mat2 must have the same dtype, but got Double and
  Float` in the decoder MLP. This is a public explicit point-query seam, even
  though normal frontend-generated points share the observation dtype.
- **Impact:** A caller using otherwise valid floating RAS-mm coordinates gets an
  implementation-level matmul exception instead of a typed contract failure.
  The failure is especially easy to hit when geometry is supplied in default
  Python/float64 tensors while the learned state is float32.
- **Recommendation:** Enforce point/state dtype equality at the query boundary
  with a clear `ValueError`, or define and apply one explicit conversion policy
  before sampling and before the MLP. Do not let an implicit cast produce a
  later linear-layer error.

## 6. Verified invariants and remaining limits

Verified in the frozen source and focused CPU checks:

- one shared MedicalNet intermediate-feature traversal feeds semantics and the
  selected feature tap;
- B, fixed A, q/reliability, and 168-d `f_spec` remain distinct typed stages;
- Gate-C uses target-free reward/cost/routing and returns a final dynamic state;
- Gate-D receives final Z only and performs one bounded dense decode;
- Gate-E receives T1ce only through the later objective API;
- Gate-G uses eval/no-grad, exact no-revisit routing, and one final-Z decode;
- generic `forward()` remains fail-closed.

Not established by this audit: GPU behavior, distributed behavior, a trained
checkpoint, real-data reconstruction quality, held-out evaluation, clinical
validity, or Gate-H authorization. The findings above are software-contract
evidence at the frozen checkout.
