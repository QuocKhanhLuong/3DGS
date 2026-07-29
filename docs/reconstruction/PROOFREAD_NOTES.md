# Proof-Read Notes — Four-Phase Reconstruction Design

Date: 2026-07-29  
Status: design review before code

## 1. Review scope

The following documents were reviewed together:

1. `phases/01_DIRECT_SPARSE_TRAINING.md`
2. `phases/02_INITIAL_ANCHOR_BOOTSTRAP.md`
3. `phases/03_ACTIVE_TRAJECTORY_UPDATE.md`
4. `phases/04_FINAL_RECONSTRUCTION.md`
5. `modules/EVIDENCE_ENCODER.md`
6. `FULL_FLOW.md`

The downstream reconstruction flow remains locked:

```text
compact encoder evidence
→ physical support anchors
→ shared tiny local field
→ SDF/level-set constrained structural Gaussians
  + volumetric appearance Gaussians
→ anchor/Gaussian propagation and active updates
→ latent patient-specific 3D Gaussian state
→ full-volume reconstruction
```

The only major architectural revision in this review is Phase 1: teacher distillation is replaced by a teacher-free structural evidence encoder.

---

## 2. Cross-phase invariants

These rules must hold in every implementation:

### Data legality

- only legal context or committed observations may enter the patient state;
- target pixels are revealed only after prediction;
- non-manifest training slices must never be opened by the main loader;
- unqueried candidate pixels must never be used by routing;
- full audit volumes must be isolated from training.

### Parameter ownership

Global learned parameters:

```text
encoder weights
tiny local-field weights
optional aggregation/update weights
optional uncertainty calibration
optional learned routing utility
```

Patient-specific state:

```text
cached evidence maps
anchors
local field evidence
Gaussian attributes
observability and uncertainty
trajectory and residual history
```

Patient-specific state must never be registered as persistent global model parameters.

### Encoder usage

- one encoder execution per committed slice at inference;
- anchor movement, birth, split, and merge sample cached maps;
- no teacher in the main method;
- analytic preprocessing cost is included in runtime accounting.

### Field naming

Until exact signed-distance behavior is demonstrated, code and papers should use:

```text
structural field
local level-set field
```

rather than unconditionally claiming a true SDF.

### Gaussian roles

- structural Gaussians: thin, field-aligned, geometry-preserving;
- volumetric Gaussians: interior intensity and low-frequency appearance;
- a surface-only Gaussian bank is insufficient for full MRI reconstruction.

---

## 3. Phase 1 review

### Current status

**Revised and internally consistent.**

The main Phase-1 path is now:

```text
fixed sparse patient manifest
→ context/target split within that manifest
→ analytic differential scaffold
→ shared high-resolution micro-CNN
→ structural and appearance maps
→ locked downstream reconstruction path
→ sparse acquired target-plane loss
```

### Main decision

Teacher distillation is replaced by:

\[
\text{analytic scaffold}
+
\text{spatial equivariance}
+
\text{intensity invariance}
+
\text{cross-modality structural consistency}
+
\text{anti-collapse}
+
\text{sparse reconstruction supervision}.
\]

### Important implementation note

The structural warm-up losses are not sufficient evidence of a useful encoder by themselves. Phase 1 passes only when the teacher-free encoder improves downstream sparse target-plane reconstruction under matched FLOPs.

### Open implementation variables

- derivative and local-contrast channel definitions;
- output stride 1, 2, or 4;
- structural and appearance channel counts;
- modality conditioning method;
- registration-confidence weighting;
- exact warm-up duration and auxiliary-loss decay.

### Phase-1 code gate

Do not continue to full-system implementation until:

1. augmentation preserves feature alignment;
2. structural features do not collapse;
3. matched registered points are more similar than mismatched points;
4. local differential cues remain recoverable;
5. the E2 encoder beats analytic-only and raw-CNN baselines;
6. sparse-manifest leakage tests pass.

---

## 4. Phase 2 review

### Current status

**Conceptually consistent with the revised Phase 1.**

Phase 2 correctly starts from cached compact features and creates provisional anchors by physical lifting. The shared tiny MLP remains downstream of evidence aggregation and is not asked to perform global image understanding.

### Required clarification in code

Phase 2 appears in two execution modes:

#### Offline training unroll

Global weights are trainable and gradients pass through bootstrap.

#### New-patient inference

Global weights are frozen and only patient state is created or updated.

The same bootstrap function should support both modes without duplicating logic.

### Anchor/SDF caution

A lifted pixel is only a provisional 3D anchor. A single 2D slice cannot supply a complete 3D SDF normal. Therefore initial anchor state must store:

```text
local structural evidence
partial orientation evidence
plane normal
uncertainty
supporting-plane references
```

It must not store an unjustifiably certain full 3D normal.

### Propagation boundary

Phase 2 initializes the first anchor and Gaussian state. Large-scale anchor–Gaussian growth should be implemented behind a separate propagation/update interface rather than hidden inside the encoder.

### Open implementation variables

- structural candidate score;
- physical suppression radius;
- cross-plane merge rule;
- evidence aggregation rule;
- anchor refinement step;
- initial structural versus volumetric Gaussian density.

---

## 5. Phase 3 review

### Current status

**Conceptually consistent, with one major training constraint.**

The render-before-update protocol, commit barrier, cache append, local state update, and stopping statuses are consistent with the project formulation.

### Major constraint: router supervision

Under permanently sparse main training, the loader cannot reveal arbitrary unacquired candidate slices. Therefore a learned utility model cannot initially assume dense counterfactual rewards for every action.

Recommended implementation order:

```text
R0: metadata-only fixed routing
R1: analytic uncertainty and coverage utility
R2: single-wave analytic routing
R3: balanced multi-wave analytic routing
R4: learned utility only with a declared simulator or isolated audit protocol
```

The first scientific result should not depend on a learned router trained from hidden complete volumes.

### Legal candidate prediction

It is legal to render predicted intensity or uncertainty on an unqueried plane because these quantities are derived from the current patient state. It is not legal to inspect the true candidate image before commitment.

### Propagation and active acquisition

Anchor–Gaussian propagation can create provisional support in unobserved regions. Such support must carry uncertainty that grows with:

- distance from observed planes;
- propagation depth;
- weak modality support;
- parent-anchor uncertainty;
- disagreement among local fields.

The router should prioritize these regions for confirmation.

### Open implementation variables

- analytic utility normalization;
- wave source creation;
- graph granularity;
- birth/split/prune acceptance energy;
- uncertainty update rule;
- stopping thresholds and patience.

---

## 6. Phase 4 review

### Current status

**Consistent with the locked representation.**

The final volume is generated by evaluating the patient-specific continuous representation on a physical grid. It is not reconstructed by reloading a hidden target volume.

### Evaluation separation

There are two distinct objects:

#### Reconstruction output

Generated from sparse patient state only.

#### Audit ground truth

Loaded only by the evaluation process after reconstruction has been completed and serialized.

The evaluation process should run in a separate script or process with no access to mutable patient state.

### Output integrity

Every reconstruction must include:

- observation budget;
- status: `CONVERGED` or `INSUFFICIENTLY_OBSERVED`;
- uncertainty volume;
- queried trajectory;
- affine and physical metadata;
- unsupported-region diagnostics.

### Open implementation variables

- joint versus gated structural/volumetric Gaussian composition;
- chunking and neighborhood acceleration;
- uncertainty calibration method;
- output normalization and inverse normalization;
- modality-specific intensity range handling.

---

## 7. Cross-phase inconsistencies resolved in this review

### Resolved: teacher versus teacher-free encoder

Old ambiguity:

```text
compact encoder might require dense teacher supervision
```

Current decision:

```text
main method uses analytic scaffold + teacher-free micro-CNN
teacher/pretrained models are upper-bound ablations only
```

### Resolved: dense hidden targets versus permanently sparse supervision

Old ambiguity:

```text
different full-volume slices could become targets across epochs
```

Current decision:

```text
main training targets are acquired slices inside a fixed sparse patient manifest
full volumes are audit/evaluation data only
```

### Resolved: encoder role

The encoder extracts compact local evidence. It does not directly produce the final 3D Gaussian bank or complete anatomy.

### Resolved: tiny MLP role

The tiny MLP predicts a local structural field value from local coordinates and compact anchor evidence. It does not perform modality interpretation, global completion, routing, or final rendering.

---

## 8. Remaining scientific risks

### Risk A — Structural constraints learn generic edges

The analytic scaffold and auxiliary losses may preserve edges without preserving useful 3D anatomy.

Required test:

- matched downstream reconstruction under constant anchor/Gaussian logic.

### Risk B — Cross-modality alignment removes private pathology

Overly strong structural alignment may suppress modality-specific lesions.

Mitigation:

- separate structural and appearance maps;
- reliability-weighted alignment;
- do not align appearance features;
- evaluate lesion-sensitive reconstruction on the audit set.

### Risk C — Sparse targets do not constrain every hidden region

A plausible volume may still be incorrect between observed planes.

Mitigation:

- complementary sparse manifests across patients;
- uncertainty tied to evidence distance and propagation depth;
- active confirmation;
- fully sampled audit evaluation;
- no claim of unique recovery without observability assumptions.

### Risk D — Propagation amplifies an incorrect seed

Mitigation:

- confidence decay;
- bounded step size and depth;
- merge/prune checks;
- render residual validation;
- active query confirmation.

### Risk E — Encoder dominates the paper result

Mitigation:

- strict FLOP accounting;
- E0/E1/E2/E3/E4 ablations;
- matched downstream state and budget;
- report gain per FLOP and gain per queried slice.

---

## 9. Recommended implementation order after approval

```text
P1. sparse-only manifest loader and leakage tests
P2. analytic differential channel bank
P3. micro-CNN encoder with Z_str / Z_app outputs
P4. structural warm-up losses and anti-collapse diagnostics
P5. minimal physical-plane target prediction test
P6. Phase-2 anchor bootstrap
P7. tiny local field and field blending
P8. structural and volumetric Gaussian initialization
P9. physical-plane renderer
P10. static sparse reconstruction baseline
P11. analytic uncertainty and single-wave routing
P12. balanced multi-wave routing
P13. adaptive topology and propagation refinement
P14. final volume export and isolated audit evaluation
```

---

## 10. Final pre-code status

| Phase | Status | Blocking issue |
|---|---|---|
| Phase 1 | Revised; ready for prototype | Must pass teacher-free encoder code gate |
| Phase 2 | Design-ready | Initial 3D orientation and anchor uncertainty must be explicit |
| Phase 3 | Conditionally ready | Start with analytic routing; learned utility needs declared supervision source |
| Phase 4 | Design-ready | Audit data must remain isolated from reconstruction state |

The recommended next action is not full-system coding. It is a Phase-1 prototype that proves the teacher-free encoder can preserve useful structural evidence and improve sparse target-plane reconstruction without complete-volume access.
