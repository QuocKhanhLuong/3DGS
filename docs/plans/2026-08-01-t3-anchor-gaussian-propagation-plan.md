# T3 Implementation Plan — Bounded Anchor–Gaussian Propagation

Date: 2026-08-01  
Status: **IMPLEMENTATION AUTHORIZED — SCIENTIFIC VALIDATION DEFERRED**  
Depends on: executable T1-C contracts and software-valid T2 anchors, field, seed
Gaussians, and initial patient state

Product alignment (2026-08-02): the maintained P0/P1 path proposes bounded
offsets in each anchor's `(t1,t2,n)` RAS-mm frame. It enforces per-anchor,
per-round, per-bank, and patient primitive budgets, explicit StructuralField
support, uncertainty and duplicate rejection, and immutable transactions.
P2/P3 adaptive topology is optional; T4 routing is not in scope.

## 1. Purpose

T3 tests the second central representation mechanism:

> Can a patient-specific anchor and StructuralField scaffold propagate bounded
> Gaussian support from sparse observed planes into nearby unobserved 3D regions
> while preserving provenance, increasing uncertainty with propagation depth,
> and improving reconstruction beyond the static T2 seed bank under matched
> primitive and compute opportunity?

The T3 causal unit is:

```text
T2 anchors + StructuralField + seed Gaussian banks
→ identify supported propagation frontiers
→ propose local child support
→ inherit and update anchor/field evidence
→ initialize child Gaussian geometry and appearance
→ increase uncertainty with propagation depth
→ accept under bounded support/complexity rules
→ repeat for a fixed number of rounds
→ immutable propagated patient state
```

T3 is static. It does not choose new MRI observations and does not implement T4
routing.

## 2. Scientific variants

All T3 behavior must be switchable through one resolved config.

```text
P0 — no propagation: T2 seed Gaussian state
P1 — fixed bounded propagation without topology adaptation
P2 — propagation with conservative move/birth proposals
P3 — propagation with bounded move/birth/split/merge/prune proposals
```

The default main candidate is `P1`. `P2` and `P3` are optional complexity
ablations and must not be required for the basic end-to-end pipeline.

Every variant must use the same legal context, anchors, renderer, target planes,
optimizer opportunity, primitive opportunity or declared complexity penalty,
and evaluation masks.

## 3. Stable package ownership

Implement or complete:

```text
src/smagm/memory/
├── contracts.py
├── observability.py
├── propagation.py
├── topology.py
├── index.py
└── appearance.py

src/smagm/anchors/
└── adaptation.py

src/smagm/state/
├── patient.py
├── versioning.py
├── update.py
└── serialization.py

src/smagm/losses/
├── field.py
└── gaussian.py

src/smagm/cli/
└── t3.py
```

Do not place propagation logic in the renderer, encoder, trainer CLI, or anchor
contracts.

## 4. Patient-state contracts

A T3 patient state must be immutable between accepted update transactions and
contain at least:

- patient and manifest identity;
- observation and episode binding;
- evidence-cache references;
- anchor batch and field configuration;
- structural Gaussian bank;
- volumetric Gaussian bank;
- primitive-to-anchor bindings;
- propagation parent IDs;
- propagation depth and round;
- observability summaries;
- uncertainty diagnostics;
- residual-free static propagation history;
- state version and provenance hash.

Global model parameters are not patient state. New-patient inference uses frozen
model weights; only the patient-specific state changes.

## 5. Propagation frontier

A propagation frontier may be proposed only from currently supported anchors,
field neighborhoods, and Gaussian support. It may use:

- local StructuralField value and gradient;
- anchor frame and support scale;
- current Gaussian covariance and primitive kind;
- distance from legal observed planes;
- local observability and modality coverage;
- overlap with existing support;
- bounded deterministic directions in the anchor-local frame.

It must not use:

- hidden target pixels;
- dense audit volumes;
- target residuals before reveal;
- unqueried candidate images;
- T4 utility or routing logic.

The reference implementation uses deterministic local directions and a maximum
physical step in RAS millimetres.

The physical-volume check has two layers: an RAS AABB is used for a cheap early
reject, then (when source geometry is declared) the proposal is transformed by
the inverse source affine and checked against the oriented voxel-center extent.
This distinction is required for rotated or oblique affines; an AABB alone is
not a valid source-volume containment contract.

## 6. Child support proposal

For parent support `j` associated with anchor `i`, propose child center:

```text
x_child = x_parent + step_mm * normalized_direction
```

The direction may be:

- tangent to the local field for structural support extension;
- normal-limited for interface thickness refinement;
- local volumetric directions for interior appearance support.

Requirements:

- all steps are bounded in physical millimetres;
- proposed points remain inside declared patient bounds;
- each proposal has deterministic parent and direction provenance;
- no child is accepted outside field/anchor support without an explicit
  unsupported status;
- duplicate proposals are suppressed in RAS-mm space;
- proposal count per parent and per round is hard bounded.

## 7. Geometry and appearance inheritance

Structural child Gaussians inherit:

- parent primitive kind;
- field-aligned frame where valid;
- tangent-dominant covariance;
- conservative normal thickness;
- anchor and parent provenance;
- increased geometry uncertainty.

Volumetric child Gaussians inherit:

- modality appearance validity;
- broader covariance;
- appearance estimate from legal anchor/cache evidence or declared parent
  interpolation;
- missing-modality uncertainty.

No hidden target value may initialize child appearance.

## 8. Observability and uncertainty

Observability is descriptive; uncertainty is not automatically calibrated.

Track per primitive:

- legal contributing observation count;
- modality count and missing modalities;
- nearest observed-plane distance;
- local anchor confidence;
- field support weight;
- local disagreement;
- propagation depth;
- parent uncertainty;
- update round;
- overlap and coverage diagnostics.

A reference uncertainty score may increase monotonically with propagation depth
and distance from observed evidence. Until calibrated at T5, expose it as
`support_uncertainty` or `reliability_diagnostic`, not calibrated predictive
uncertainty.

## 9. Acceptance energy

Each proposed update must be evaluated by one shared declared energy:

```text
E = reconstruction_proxy
  + field_consistency
  + overlap_penalty
  + displacement_penalty
  + complexity_penalty
  + uncertainty_penalty
```

For static propagation without target residual, `reconstruction_proxy` may use
only legal context-plane reconstruction or leave-one-context-out prediction.
Target pixels remain closed until the legal prediction receipt barrier.

An update is accepted only when:

- all contracts validate;
- the energy improvement exceeds a declared threshold;
- primitive and compute budgets remain within bounds;
- coverage does not grow through unsupported regions without uncertainty growth;
- no legality or provenance check fails.

## 10. Topology operations

Topology is optional and config-gated.

### Move

Conservative center/frame adjustment inside anchor-local support.

### Birth

Create support at an accepted propagation frontier with insufficient current
coverage.

### Split

Replace one broad primitive with bounded children while preserving total support
and provenance.

### Merge

Combine close compatible primitives only when primitive kind, modality validity,
geometry, and provenance are compatible.

### Prune

Remove support that is persistently redundant, invalid, or unsupported under the
declared energy.

All operations must be deterministic under the same state/config/seed and must
produce an immutable transaction record. Default T3 runs keep topology disabled
until `P1` is stable.

## 11. Differentiability and training

Two modes are required:

### Offline training

Propagation parameters, field weights, and allowed initialization/update
parameters may receive gradients through target-plane reconstruction after the
legal reveal barrier.

### Frozen-weight patient inference

Global weights are frozen. Propagation updates only patient-specific state and
uses deterministic configured rules or frozen learned update modules.

Tests must prove gradients reach the intended trainable parameters without
making patient state persistent global parameters.

## 12. Required losses

Configurable components include:

- target-plane supported-mask reconstruction;
- context reconstruction consistency;
- field value and gradient overlap consistency;
- propagation displacement penalty;
- covariance and scale regularization;
- duplicate/overlap penalty;
- complexity and primitive-count penalty;
- uncertainty monotonicity diagnostic;
- optional topology acceptance margin.

Every component must return a typed active/inactive result and reason. No term
may silently become zero.

## 13. Required tests

### Contracts and state

- immutable state versions;
- parent and anchor provenance;
- structural/volumetric bank separation;
- patient and manifest binding;
- serialization round trip;
- no global parameter captured as patient state.

### Propagation

- deterministic proposals;
- physical step bound;
- patient-bound clipping/rejection;
- duplicate suppression;
- propagation-depth tracking;
- uncertainty non-decrease with depth;
- no hidden-target access;
- bounded primitive growth.

### Geometry

- rotated and anisotropic anchor frames;
- covariance positive definiteness;
- structural tangent/normal semantics;
- volumetric support distinction;
- renderer finite output.

### Topology

- move/birth/split/merge/prune config gates;
- conservation or declared change in support mass;
- deterministic transaction records;
- incompatible merge rejection;
- primitive budget enforcement.

### Autograd

- target reconstruction reaches propagation parameters;
- target receipt digest does not detach live tensors;
- frozen-weight inference changes no global parameters;
- all enabled losses remain finite.

### Scope

- no routing candidate or utility API;
- no unqueried pixel access;
- no full-volume audit loading;
- no T4 implementation.

## 14. CLI and artifacts

Required command:

```bash
python -m smagm.cli.t3 \
  --config configs/experiments/t3_synthetic.json \
  --variant p1 \
  --rounds 2 \
  --output-dir /tmp/smagm-t3
```

Artifacts:

```text
resolved_config.json
provenance.json
initial_state.json
propagation_transactions.jsonl
primitive_diagnostics.jsonl
render_metrics.json
summary.json
checkpoint.pt
```

The CLI must support `p0`, `p1`, and any implemented optional topology variants.
Generated artifacts are ignored by Git.

## 15. Software continuation gate

The fast-track sprint may continue to T5 when these software guards pass:

1. P0 reproduces the T2 seed state;
2. P1 produces a finite bounded propagated state;
3. propagation is deterministic under the same config and seed;
4. no target/audit leakage occurs;
5. state serialization is exact;
6. renderer output and backward pass are finite;
7. primitive and runtime budgets are reported;
8. propagation can be disabled for matched ablation;
9. T4 routing code is absent.

This continuation gate is not a scientific pass. The value of propagation is
decided only by the consolidated final evaluation.
