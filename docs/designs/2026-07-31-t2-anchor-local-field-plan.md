# T2 Pre-Authorization Design — Physical Anchors, Local Structural Field, and Seed Gaussians

Date: 2026-07-31  
Status: **PRE-AUTHORIZATION DESIGN — IMPLEMENTATION BLOCKED**  
Blocked by: T1-F/T1-R/T1-M evidence and explicit Human authorization
Stable owners: `src/smagm/anchors/`, `src/smagm/fields/`, `src/smagm/memory/initialize.py`, `src/smagm/state/builder.py`

## 1. Purpose

T2 is the first tranche that tests the project’s central support-anchor
representation hypothesis. It asks:

> Does lifting legal observed structural evidence into physical support
> anchors, aggregating multi-plane evidence, decoding one shared tiny
> anchor-local structural field, and initializing anchor-related seed Gaussians
> improve reconstruction over matched deterministic fixed supports and simpler
> free/global alternatives?

T2 must isolate this mechanism before propagation or adaptive topology is
introduced.

The governing references are:

- [`../reconstruction/phases/02_INITIAL_ANCHOR_BOOTSTRAP.md`](../reconstruction/phases/02_INITIAL_ANCHOR_BOOTSTRAP.md)
- [`../reconstruction/modules/ANCHOR_LOCAL_FIELD.md`](../reconstruction/modules/ANCHOR_LOCAL_FIELD.md)
- [`../../CODEBASE.md`](../../CODEBASE.md)
- [`../research/2026-07-29-isbi-full-mechanism-collision.md`](../research/2026-07-29-isbi-full-mechanism-collision.md)
- [`../../quality/checklists.json`](../../quality/checklists.json)

## 2. Authorization and prerequisites

This document freezes a candidate architecture and evaluation boundary. It does
not authorize implementation.

T2 remains blocked until all of the following are true:

1. T1-C implements a legal, reproducible sparse episodic trainer.
2. T1-F feature-validity evidence is accepted.
3. T1-R matched reconstruction attribution is accepted.
4. T1-M medical-fidelity development evidence is accepted.
5. The fixed-topology Gaussian baseline beats the declared interpolation floor.
6. An explicit Human decision authorizes T2.

Before that decision:

- no `src/smagm/anchors/`, `fields/`, or T2 memory implementation is added;
- no empty T2 package or placeholder API is created;
- no quality check is activated as though T2 software existed;
- `docs/codex/README.md` continues to report T2 as `BLOCKED`.

## 3. Scientific unit under test

The T2 causal unit is:

```text
legal cached context evidence
→ sparse physical candidate selection
→ RAS-mm provisional anchors
→ cross-plane consolidation
→ compact anchor evidence aggregation
→ one shared tiny anchor-local structural field
→ stable field blending
→ anchor-related structural and volumetric seed Gaussians
→ physical target-plane rendering
```

T2 does not include:

- anchor–Gaussian propagation;
- iterative support growth into unobserved regions;
- learned or residual-driven birth, split, merge, or prune;
- local assimilation after new observations;
- active trajectory selection;
- final full-volume reconstruction and export.

Those responsibilities belong to T3, T4, and T5.

## 4. T2 scope by stable package

### 4.1 Anchors

T2 may implement:

- typed anchor contracts;
- context-evidence candidate scoring;
- physical non-maximum suppression;
- lifting from feature coordinates to RAS millimetres;
- deterministic cross-plane consolidation;
- evidence aggregation;
- partial frame construction and validation;
- bounded spatial indexing;
- one initial static bootstrap.

`anchors/adaptation.py` remains unimplemented in T2. Patient-specific move,
birth, split, merge, and prune proposals are T3 responsibilities even though
the long-horizon theory discusses them.

### 4.2 Fields

T2 may implement:

- typed local-coordinate and field-query contracts;
- one shared low-capacity MLP;
- stable compact-support or partition-of-unity blending;
- batched nearby-anchor queries;
- overlap/value/gradient diagnostics;
- structural-field terminology and naming gates.

T2 must not call the field an SDF unless sign, distance calibration, Eikonal,
and gradient-norm evidence are separately established.

### 4.3 Seed Gaussian initialization

T2 may implement the smallest static initialization boundary needed to test the
anchor-field mechanism:

- field-aligned structural seed Gaussians;
- interior or local volumetric appearance seed Gaussians;
- explicit primitive kind and anchor provenance;
- modality appearance validity and observability summaries;
- canonical-RAS runtime Gaussian construction.

It must not implement propagation, assimilation, topology adaptation, or
unbounded patient-state growth.

### 4.4 State construction

A minimal `state/builder.py` may compose legal cached evidence, anchors, fields,
and seed Gaussian banks into one immutable initial patient state. Versioning
must bind every scientific input without target pixels.

## 5. Anchor contract

An anchor is a patient-specific geometric support. It is neither an observation
slice nor a Gaussian primitive.

A minimum typed contract is:

```python
@dataclass(frozen=True)
class AnchorBatch:
    anchor_ids: tuple[str, ...]
    centers_ras_mm: torch.Tensor          # [N, 3]
    frame_axes_ras: torch.Tensor          # [N, 3, 3]
    frame_validity: torch.Tensor          # [N, 3], bool
    support_scales_mm: torch.Tensor       # [N, 3]
    evidence: torch.Tensor                # [N, C]
    geometry_confidence: torch.Tensor     # [N, 1]
    observability: torch.Tensor           # [N, O]
    contributing_observation_ids: tuple[tuple[str, ...], ...]
    contributing_plane_hashes: tuple[tuple[str, ...], ...]
    provenance_hashes: tuple[str, ...]
```

Required semantics:

- centers and scales use canonical RAS millimetres;
- frames are right-handed and orthonormal where declared valid;
- initial anchors may have only a partial frame;
- confidence and observability are separate;
- modality-specific evidence is not silently averaged into one intensity;
- every anchor retains all contributing observation and plane provenance;
- no target-derived value enters anchor construction;
- anchor IDs and tie-breaking are deterministic under the same inputs/config.

A single observed plane does not justify a certain full 3D anatomical normal.
The source-plane basis may initialize a partial frame, while field-derived
normal refinement requires sufficient legal evidence.

## 6. Candidate generation

Candidate generation may use values from legal cached context features. This is
scientifically different from the value-independent fixed support baseline in
T1.

A candidate score may combine declared terms:

\[
s(p)=
\lambda_g s_{gradient}(p)
+
\lambda_c s_{contrast}(p)
+
\lambda_z s_{structural}(p)
+
\lambda_r r(p).
\]

Requirements:

- input is only cached context evidence and topology;
- reliability may score a legal candidate but cannot create validity outside
  the feature mask;
- every score component is separately logged;
- threshold and top-k policies are explicit;
- ties follow deterministic row-major or canonical physical ordering;
- candidate count has a hard budget;
- invalid and padded feature centres are excluded;
- the selection rule is separately ablated from the local field.

Candidate selection must not inspect target pixels, audit labels, hidden
planes, or future residuals.

## 7. Physical lifting and suppression

For a feature-grid candidate `(v_f, u_f)`, use its bound
`FeatureGridToPlaneTransform` to obtain the canonical RAS-mm centre. Do not
reconstruct geometry from shape, spacing, or affine fragments independently.

Physical non-maximum suppression must operate in RAS millimetres, not pixel
indices. The suppression policy must declare:

- Euclidean or anisotropic distance;
- modality and plane grouping;
- physical radius;
- deterministic tie handling;
- whether candidates from intersecting planes may consolidate;
- how disconnected anatomical regions retain coverage.

A lifted feature location is a provisional anchor, not a proven point on a
surface or zero level set.

## 8. Cross-plane consolidation

Consolidation receives provisional RAS-mm candidates and their provenance.
It may merge only when declared geometric and evidence criteria are met.

Required behavior:

- merge distances use physical units;
- compatible duplicate supports consolidate deterministically;
- conflicting observations are retained as disagreement/observability, not
  averaged away silently;
- modality evidence remains identifiable;
- disconnected regions are not bridged through transitive chains without a
  bounded policy;
- the output preserves contributing-plane references;
- no learned topology adaptation is hidden in consolidation.

A recommended reference policy is deterministic graph components under a
bounded physical radius with an explicit maximum component diameter and
confidence-aware medoid or weighted centre.

## 9. Evidence aggregation

For anchor `i` and legal cached observation `k`, construct a token such as:

\[
e_{ik}=
[
Z_k^{str}(\pi_k(a_i)),
Z_k^{app}(\pi_k(a_i)),
r_k(\pi_k(a_i)),
d(a_i,P_k),
n_k,
m_k,
c_{reg,k}
].
\]

Aggregation must:

- sample the exact feature cache;
- never rerun the encoder during patient inference;
- use differentiable physical projection where required;
- reject out-of-plane or invalid samples;
- require declared registration and confidence for cross-observation fusion;
- separate structural and modality-specific appearance evidence;
- expose geometry, distance, modality, registration, and reliability weights;
- return a typed empty-contributor failure;
- retain per-anchor observability and disagreement diagnostics.

The initial reference should use a transparent weighted mean plus dispersion
statistics. Attention is not required and should not be introduced until the
reference aggregation is tested.

## 10. Frame construction

Initial frame semantics:

```text
axis_u, axis_v from a contributing physical plane
signed plane normal as acquisition orientation evidence
frame_validity marks what is actually observed
```

Refined frame semantics:

- a field-derived normal may be used only when the blended field gradient is
  finite, sufficiently large, and supported by more than a declared minimum of
  legal evidence;
- tangent axes are constructed deterministically from the normal and a stable
  reference axis;
- sign ambiguity is recorded rather than hidden;
- uncertain frames retain partial validity and higher uncertainty;
- frame rotation tests must be equivariant in canonical RAS.

## 11. Shared tiny local structural field

For anchor `i` and physical query `x`:

\[
\xi_i(x)=S_i^{-1}R_i^T(x-a_i),
\qquad
f_i(x)=M_{tiny}([\xi_i(x),h_i]).
\]

Locked constraints:

- all anchors share one MLP;
- the MLP receives local coordinates and already aggregated compact evidence;
- it does not align modalities, select observations, create anchors, or plan
  routes;
- default output is one scalar structural-field value;
- width and depth remain intentionally small;
- smooth activations support field gradients;
- local coordinate scale and support bounds are explicit;
- empty or out-of-support queries return typed unsupported status.

Initial architecture envelope:

```text
input: 3 local coordinates + compact evidence
→ Linear
→ SiLU or Softplus
→ 2–4 hidden layers
→ width 16–64
→ one scalar structural-field output
```

The exact width, depth, and activation are ablation variables.

## 12. Field blending

The patient structural field is:

\[
F(x)=
\frac{\sum_i w_i(x)f_i(x)}{\sum_i w_i(x)+\epsilon}.
\]

Requirements:

- support weights are non-negative and finite;
- normalization is numerically stable;
- only bounded nearby anchors are evaluated;
- unsupported queries remain explicit when total legal weight is below a
  declared threshold;
- anchor-order permutation does not change the result beyond numerical
  tolerance;
- overlap disagreement is exposed;
- gradients remain finite through local coordinates, evidence, field weights,
  and MLP parameters;
- float32 and float64 references are tested.

Do not silently fill unsupported field regions with a confident scalar.

## 13. Seed Gaussian initialization

T2 should compare the anchor-field mechanism through a bounded seed Gaussian
state. It does not yet propagate support.

### Structural seed Gaussians

- centres are tied to or locally offset from valid anchors;
- covariance is predicted in the anchor-local frame and rotated into RAS;
- normal thickness is bounded and typically smaller than tangent scales;
- primitive provenance binds anchor ID, contributing evidence, and field query;
- field value/gradient may constrain initialization only through a declared,
  ablatable rule.

### Volumetric appearance seed Gaussians

- provide local interior intensity support;
- remain distinguishable from structural primitives;
- carry modality-specific appearance validity;
- use bounded covariance and explicit observability;
- do not hallucinate missing-modality appearance as observed evidence.

The runtime state must remain compatible with the pure existing renderer and
one-time amplitude-gauge policy.

## 14. Patient-state boundary

The T2 initial state should contain:

```text
InitialPatientState
├── exact legal context/cache provenance
├── AnchorBatch
├── field configuration and global model hash
├── structural seed Gaussian bank
├── volumetric appearance seed Gaussian bank
├── observability/disagreement summaries
├── explicit unsupported regions
└── immutable state version
```

Global trainable parameters include the shared MLP and any explicitly approved
aggregation/initialization weights. Anchors, evidence, fields evaluated for a
patient, Gaussian banks, and observability are patient state and must not become
persistent global model parameters.

## 15. Required baselines and ablations

T2 does not pass by outperforming T1 with more primitives or compute. At
minimum compare:

### Baselines

- T1 deterministic fixed supports + common fixed Gaussian head;
- matched free Gaussian initialization without anchor/field constraints;
- simple global coordinate MLP or global compact field under matched capacity;
- direct anchor-to-Gaussian decoder without the local structural field;
- interpolation floor retained from T1-R.

### Anchor ablations

- uniform physical candidates versus structural candidates;
- no consolidation versus physical consolidation;
- single-plane versus registered multi-plane aggregation;
- fixed plane frame versus supported field-derived frame;
- reliability-free versus reliability-weighted scoring.

### Field ablations

- no field;
- coordinate-only local field;
- coordinate + aggregated evidence;
- shared local tiny MLP versus capacity-matched global MLP;
- no overlap loss versus value/gradient consistency;
- fixed isotropic versus anisotropic support scale.

### Seed-Gaussian ablations

- structural-only;
- volumetric-only;
- distinguishable dual seed banks;
- field-aligned versus unconstrained covariance;
- equal primitive count and optimization opportunity.

Every comparison must lock legal observations, renderer, target planes,
primitive budget, optimizer opportunity, hardware, precision, and accounting.

## 16. Required automated tests

### Contracts and geometry

- anchor contract rejects non-finite centres/scales/evidence;
- RAS-mm lifting matches the bound feature transform;
- oblique and rotated planes produce equivariant anchors and frames;
- partial frames are represented honestly;
- anchor and Gaussian identities remain distinct;
- every anchor retains exact provenance.

### Candidates and consolidation

- invalid/padded centres are excluded;
- target or audit data cannot enter scoring;
- deterministic ties and budgets hold;
- physical NMS is invariant to pixel resolution changes representing the same
  physical geometry;
- duplicate merging is deterministic;
- conflicting evidence remains visible;
- disconnected regions are not merged accidentally.

### Aggregation and cache

- only exact cache keys are accepted;
- encoder is not rerun during patient inference;
- registration confidence gates cross-plane fusion;
- invalid projection and empty contributor sets fail clearly;
- modality appearance remains identifiable;
- gradients reach cached live evidence where training authorizes them.

### Field

- exact local-coordinate mapping;
- anchor-order permutation invariance;
- bounded support and explicit unsupported output;
- stable normalized blending;
- finite gradients in float32 and float64;
- overlap diagnostics activate on contradictory fields;
- one shared MLP is used for all anchors;
- no public `sdf` API appears without the naming gate.

### Seed Gaussians

- local covariance rotates correctly into RAS;
- structural and volumetric kinds are distinguishable;
- primitive and anchor provenance round-trip;
- missing modality slots remain invalid;
- amplitude gauge is applied exactly once;
- pure renderer receives a valid canonical state;
- primitive count and memory budget are enforced.

### Integration

A CPU synthetic path must execute:

```text
legal cached context features
→ candidates
→ RAS-mm provisional anchors
→ consolidation/aggregation
→ local field queries and blending
→ structural + volumetric seed Gaussians
→ target-plane render
→ legal reconstruction loss
→ backward
```

No propagation, routing, or adaptive topology may appear in this test.

## 17. Quality checklist activation

After explicit T2 authorization, convert the existing T2 `planned` checks into
exact evidence for:

- legal cached-evidence candidates;
- RAS-mm NMS and consolidation;
- typed anchor contract;
- cache-only aggregation;
- registration confidence;
- shared low-capacity local field;
- stable blending and explicit unsupported behavior;
- gradient reachability;
- structural-field naming discipline;
- anchor/field/primitive count, cache, runtime, and memory accounting.

Add a blocker verifying that T3 propagation, topology adaptation, T4 routing,
and T5 reconstruction packages were not introduced.

## 18. T2 decision gate

T2 may pass only when:

1. all geometry, legality, cache, field, and autograd contracts pass;
2. anchor count, primitive count, parameters, runtime, and memory are bounded;
3. the local field adds measurable reconstruction value over deterministic
   supports and direct anchor-to-Gaussian alternatives;
4. the complete T2 representation improves over a matched free/global
   alternative without hiding additional compute or primitive opportunity;
5. lesion/ROI and boundary development evidence does not regress meaningfully;
6. unsupported regions and disagreements remain explicit;
7. terminology remains `structural_field` unless SDF evidence exists.

Possible outcomes:

- `PASS`: the anchor-local field earns its complexity and T3 design may be
  considered separately.
- `PASS_WITH_CONDITIONS`: bounded issues remain, with tracked conditions before
  T3.
- `PARTIAL/REWORK`: software works but one mechanism lacks attribution.
- `FAIL`: the complex mechanism does not beat simpler matched alternatives;
  remove or simplify the failed component.

A successful synthetic demo, attractive anchor visualization, or lower training
loss alone is insufficient.

## 19. Stop and demotion rules

- If structural candidates do not beat uniform physical candidates, retain
  deterministic supports and remove candidate-scoring novelty.
- If consolidation adds no value, simplify to independent physical anchors.
- If coordinate + evidence does not beat coordinate-only, remove evidence
  conditioning from the field.
- If the local field does not beat a direct anchor-to-Gaussian decoder, remove
  or demote the field.
- If the local field does not beat a capacity-matched global/free alternative,
  remove it from the headline claim.
- If dual seed banks do not beat a simpler bank under matched primitives,
  simplify the memory initialization.
- If gains disappear under matched primitive count or compute, T2 fails.
- If pathology/ROI fidelity regresses meaningfully, T2 fails its medical gate.
- If sign, Eikonal, and distance evidence are absent, use
  `StructuralField`, never `SDF`.

## 20. Suggested implementation commits after authorization

1. `feat(anchors): add typed physical anchor contracts and candidates`
2. `feat(anchors): add lifting consolidation and evidence aggregation`
3. `feat(fields): add shared local structural field and blending`
4. `feat(memory): add bounded anchor-related seed Gaussian initialization`
5. `feat(state): add immutable initial patient-state builder`
6. `test(t2): add geometry cache field autograd and leakage gates`
7. `docs(codex): add T2 executable handoff`

## 21. Future Luna High prompt boundary

After T2 is explicitly authorized, the coding prompt must begin with:

```text
Implement only the approved T2 static bootstrap described by the active T2
design, CODEBASE.md, Phase-2 theory, Anchor-Local Field module, and T2 quality
checklist.

Use only legal cached context evidence. Implement physical anchors, transparent
aggregation, one shared tiny local StructuralField, stable blending, and bounded
structural/volumetric seed Gaussian initialization.

Do not implement propagation, residual assimilation, birth/split/merge/prune,
adaptive topology, routing, full-volume reconstruction, T3+, or placeholders.
Do not use SDF terminology unless the design gate is explicitly changed by a
Human decision.
```

Until the prerequisite T1 evidence and explicit T2 authorization exist, this is
only a future implementation boundary.
