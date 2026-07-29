# Phase 3 — Active Trajectory and Incremental Update

## 1. Objective

Select the next sequence–slice observations that most improve 3D reconstruction, then update only the affected portion of the patient-specific representation.

The trajectory is closed loop:

```text
current representation
→ predict where reconstruction is weak
→ select legal observation
→ render before update
→ observe residual
→ update local state
→ recompute utility
```

The route is not fixed once at initialization.

---

## 2. Inputs and outputs

### Inputs

```text
PatientState_t
├── current local/global structural field
├── structural and volumetric Gaussian banks
├── anchor evidence and support
├── compact observation cache
├── uncertainty and observability maps
├── previous residuals
├── trajectory history
└── remaining budget
```

### Outputs

```text
PatientState_t+1
├── appended legal observation
├── updated cache
├── locally repaired field
├── updated anchors and Gaussians
├── updated uncertainty
├── updated graph/frontiers
└── new stopping diagnostics
```

---

## 3. Candidate representation

Each unqueried candidate is a sequence–slice action

\[
a=(m,z).
\]

The router may attach region-level descriptors for planning, but the committed observation is still a whole physical slice unless the acquisition protocol explicitly supports region queries.

Legal candidate descriptors include:

- physical plane position and orientation;
- intersection with uncertain anchors or Gaussian support;
- current predicted intensity and uncertainty on the plane;
- modality evidence missing from intersected regions;
- distance and transition cost from previous queries;
- current coverage and redundancy.

Unqueried pixels remain inaccessible.

---

## 4. Reconstruction-driven utility

The task is 3D reconstruction, so utility should prioritize expected reduction of reconstruction error rather than pathology segmentation gain.

A candidate score may be decomposed as

\[
Q_t(a)=
\alpha_u U_t(a)
+\alpha_c C_t(a)
+\alpha_m M_t(a)
+\alpha_d D_t^{div}(a)
+\alpha_r \widehat{R}_t(a)
-\alpha_o O_t(a)
-\alpha_k K_t(a).
\]

Interpretation:

- \(U_t(a)\): uncertainty mass intersected by the plane;
- \(C_t(a)\): uncovered structural or volumetric support;
- \(M_t(a)\): missing modality evidence;
- \(D_t^{div}(a)\): diversity from previous queries;
- \(\widehat{R}_t(a)\): predicted reduction in reconstruction residual;
- \(O_t(a)\): overlap/redundancy;
- \(K_t(a)\): query or transition cost.

The utility can initially be analytic and later replaced or augmented by a learned predictor without changing the rest of the interface.

---

## 5. Balanced multi-wave routing

### 5.1 Motivation

A single greedy frontier can spend most of the budget near one difficult region. Multiple waves provide parallel exploration of disconnected or differently uncertain regions.

### 5.2 Wave sources

Wave sources may be initialized from:

- disconnected anchor clusters;
- high-uncertainty structural regions;
- modality-incomplete regions;
- superior, central, and inferior anatomical partitions;
- high-residual appearance regions.

### 5.3 Graph cost

For current fixed state, edge cost may be

\[
c_t(i,j)=
\frac{
\lambda_gd_{geo}(i,j)
+\lambda_fd_{feat}(i,j)
+\lambda_md_{mod}(i,j)
+\lambda_jd_{jump}(i,j)
}{
\epsilon+Q_t(j)
}
+\lambda_oO_t(j)
+\lambda_lL_t(j).
\]

A high-information candidate becomes easier to reach. Nonnegative edge costs permit exact shortest-path solutions for the current graph state.

### 5.4 Global scheduler

Each wave proposes candidates. A global scheduler selects one or a small complementary batch while penalizing overlap and load imbalance.

---

## 6. Commit barrier

The system must separate proposal from observation.

```text
score candidate
→ select action
→ commit action to ledger
→ only then load pixels
```

This barrier prevents accidental leakage from candidate images during routing.

---

## 7. Render-before-update

Before the new observation is assimilated, render it using the previous state:

\[
\hat I_t=R(\mathcal S_t,m_t,P_t).
\]

After loading the true slice:

\[
E_t=I_t-\hat I_t.
\]

The residual is an observable measure of what the old state failed to explain. Rendering after assimilation would destroy this diagnostic.

---

## 8. Encode once and append cache

The selected slice runs through the evidence encoder once:

\[
(Z_t,C_t)=E(I_t,m_t,P_t).
\]

The result is appended to the patient cache. Anchor movement or topology changes sample this cached map; they do not re-run the encoder.

---

## 9. Affected-region detection

Only state whose support intersects the new plane should receive large updates.

Define:

\[
\mathcal N_t=
\{i:\operatorname{Support}(g_i)\cap P_t\neq\emptyset\}.
\]

Additional affected support may be created in regions where residual magnitude is high but current Gaussian coverage is low.

---

## 10. Local evidence update

For each affected anchor, sample the new cached feature and form new evidence \(\tilde{\mathbf h}_{i,t}\).

A gated update is

\[
\mathbf h_i^{t+1}
=(1-\gamma_{i,t})\mathbf h_i^t
+\gamma_{i,t}\tilde{\mathbf h}_{i,t}.
\]

The gate may depend on:

- observation reliability;
- residual magnitude;
- plane distance;
- modality novelty;
- agreement with previous evidence;
- current uncertainty.

---

## 11. Local field update

The global tiny-MLP weights remain frozen during patient inference. The local field changes because:

- anchor evidence changes;
- anchor center/frame changes;
- local support scale changes;
- topology changes.

\[
f_i^{t+1}(\mathbf x)
=M_{tiny}(\boldsymbol\xi_i^{t+1}(\mathbf x),\mathbf h_i^{t+1}).
\]

This is state adaptation, not per-patient network fine-tuning.

---

## 12. Anchor and topology operations

### Move

Shift an anchor within its local tangent/normal constraints when persistent residual indicates geometric misalignment.

### Birth

Create support when a high-residual region is insufficiently covered.

### Split

Replace one broad primitive with smaller primitives when residual or evidence is heterogeneous within its support.

### Merge

Combine highly overlapping primitives with compatible evidence.

### Prune

Remove unsupported, persistently inconsistent, or negligible primitives.

Topology operations should be accepted only when they reduce a shared reconstruction/complexity energy by a sufficient margin.

---

## 13. Structural and appearance update rates

Geometry should update more conservatively than modality appearance:

\[
\eta_{structure}<\eta_{appearance}.
\]

Reason:

- a modality-specific intensity change should not immediately deform shared anatomy;
- persistent multi-plane residual is stronger evidence of geometric error;
- appearance slots can assimilate new modality evidence quickly.

---

## 14. Volumetric appearance update

For modality \(m_t\), intersecting volumetric Gaussians receive observed appearance evidence.

A simple update is

\[
\mathbf c_{i,m_t}^{t+1}
=(1-\eta_{i,t})\mathbf c_{i,m_t}^{t}
+\eta_{i,t}\tilde{\mathbf c}_{i,m_t}.
\]

Slots for unobserved modalities are not filled with false certainty. Their uncertainty remains high until supported by direct or well-calibrated cross-modal inference.

---

## 15. Uncertainty update

Uncertainty may decrease when:

- a plane directly observes the region;
- multiple modalities agree;
- residual decreases after update;
- neighboring local fields agree.

It may increase when:

- new evidence conflicts with the representation;
- topology changes substantially;
- an anchor moves far from its prior support;
- modalities disagree;
- Gaussian overlap creates ambiguous explanations.

---

## 16. Local graph repair

Let \(\Delta\mathcal R_t\) be the changed spatial region. Recompute candidate utilities and graph edges only for observations intersecting or depending on \(\Delta\mathcal R_t\) when possible.

A full graph rebuild is a valid reference implementation but should not be required by the final efficient system.

---

## 17. Stopping controller

Possible persistent stopping conditions:

- maximum predicted gain below \(\epsilon_Q\);
- observed residual below \(\epsilon_R\);
- relative structural field change below \(\epsilon_F\);
- anchor/Gaussian topology unchanged for \(P\) rounds;
- worst relevant uncertainty below \(\epsilon_U\);
- no legal candidate remains.

Status must distinguish:

```text
CONVERGED
INSUFFICIENTLY_OBSERVED
NO_CANDIDATES
INVALID_STATE
```

Budget exhaustion alone is not convergence.

---

## 18. Per-round diagnostics

Record:

- selected action and proposal scores;
- per-wave proposals;
- rendered prediction before update;
- residual map;
- affected anchors/Gaussians;
- topology operations;
- state-change magnitude;
- uncertainty before/after;
- reconstruction quality on legal diagnostics;
- encoder, renderer, update, and routing latency;
- cumulative observation budget.
