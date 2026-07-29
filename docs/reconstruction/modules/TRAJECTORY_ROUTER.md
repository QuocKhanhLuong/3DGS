# Module Architecture — Reconstruction Trajectory Router

## 1. Purpose

Choose the next legal sequence–slice observation that is expected to improve 3D reconstruction most efficiently.

The router operates on the current patient representation, not on unqueried image pixels.

---

## 2. Action space

A committed action is

\[
a=(m,z),
\]

where \(m\) is the MRI modality and \(z\) identifies a registered physical slice plane.

Internal planning nodes may include spatial regions or anchor clusters, but the external action contract remains a whole slice unless the dataset supports region-level acquisition.

---

## 3. Inputs

```text
RouterInput_t
├── legal unqueried candidate metadata
├── current structural field
├── structural and volumetric Gaussian banks
├── anchor clusters
├── uncertainty and coverage
├── modality observability
├── residual history
├── previous actions
└── remaining budget
```

No candidate image pixels may be loaded before action commitment.

---

## 4. Candidate descriptors

For candidate plane \(a\), derive descriptors from current state:

- uncertain structural mass intersected by the plane;
- uncertain appearance mass intersected by the plane;
- number of under-observed Gaussians;
- missing modality evidence;
- predicted current rendering variance;
- distance from observed planes;
- overlap with previous observations;
- physical jump or acquisition cost;
- intersection with high-residual neighborhoods;
- coverage of disconnected anchor clusters.

---

## 5. Utility

A reconstruction-oriented utility can be written as

\[
Q_t(a)=
\alpha_uU_t(a)
+\alpha_cC_t(a)
+\alpha_mM_t(a)
+\alpha_nN_t(a)
+\alpha_g\widehat{G}_t(a)
-\alpha_rR_t(a)
-\alpha_kK_t(a).
\]

- \(U_t\): uncertainty reduction opportunity;
- \(C_t\): uncovered volume or interface coverage;
- \(M_t\): missing modality evidence;
- \(N_t\): novelty relative to previous observations;
- \(\widehat{G}_t\): predicted reconstruction gain;
- \(R_t\): redundancy;
- \(K_t\): transition/acquisition cost.

The initial router may compute these terms analytically. A learned gain predictor is an optional later variant.

---

## 6. Observation graph

A graph node may represent a candidate sequence–slice region:

\[
v_i=(m_i,z_i,r_i,\mathbf x_i,\mathbf q_i).
\]

Edges may encode:

- adjacent physical slices;
- aligned cross-modality planes;
- shared anchor-cluster intersections;
- field-geodesic proximity;
- similar missing-evidence patterns.

The graph must be constructed without unqueried image features.

---

## 7. Multi-wave architecture

### Wave sources

Initialize parallel waves from complementary regions such as:

- high-uncertainty anchor clusters;
- disconnected structural components;
- under-covered volumetric regions;
- modality-incomplete regions;
- spatial partitions.

### Edge cost

\[
c_t(i,j)=
\frac{
\lambda_gd_{geo}
+\lambda_fd_{state}
+\lambda_md_{mod}
+\lambda_jd_{jump}
}{\epsilon+Q_t(j)}
+\lambda_oO_t(j)
+\lambda_lL_t(j).
\]

High-value candidates have lower effective traversal cost.

### Wave proposal

Each wave proposes frontier candidates based on accumulated path cost and current utility.

### Global scheduler

The scheduler chooses complementary proposals while penalizing:

- pairwise plane overlap;
- repeated modality evidence in the same region;
- wave load imbalance;
- excessive physical transitions;
- redundant coverage.

---

## 8. Router modes

### R0 — Metadata baseline

Uniform or balanced physical coverage using only modality and plane geometry.

### R1 — Uncertainty greedy

Select the plane intersecting the most uncertainty.

### R2 — Single-wave graph routing

Use one frontier and dynamic utility.

### R3 — Balanced multi-wave routing

Use parallel frontiers and a global scheduler.

### R4 — Learned expected reconstruction gain

Predict gain from legal state descriptors.

### R5 — Receding-horizon route optimization

Plan several steps, execute one or a small batch, then replan after assimilation.

The final contribution should compare these modes rather than assuming the most complex router is automatically best.

---

## 9. Commit protocol

```text
candidate scoring
→ action proposal
→ legality check
→ budget check
→ commit to observation ledger
→ load image pixels
```

The router receives no new image content until after the commit record exists.

---

## 10. Training options

### Analytic router

No router weights. Utility is computed from patient-state diagnostics.

### Supervised gain predictor

During training, estimate actual quality gain from querying candidate slices and regress the predicted gain.

### Ranking objective

Train the router to rank candidates by future reconstruction improvement rather than predict an absolute value.

### Reinforcement or policy optimization

Possible but higher variance and more difficult to attribute. It should not be the first learned-router experiment unless simpler objectives fail.

---

## 11. Oracle utility experiment

To determine whether routing is worth learning, compute an offline oracle on training/evaluation episodes:

1. temporarily query each legal candidate;
2. measure true reconstruction improvement;
3. rank candidates by gain;
4. compare analytic and learned scores to the oracle ranking.

Oracle information is never used in deployed inference. It establishes the headroom available to trajectory optimization.

---

## 12. Stopping interface

The router provides:

```text
RouterDecision
├── selected action or batch
├── predicted gain
├── proposal sources
├── redundancy diagnostics
├── no-candidate reason
└── confidence
```

If maximum legal predicted gain is below threshold for multiple rounds, the stopping controller may declare convergence only when representation-stability and uncertainty conditions also hold.

---

## 13. Complexity

Report:

- candidate count;
- graph nodes and edges;
- utility evaluation time;
- shortest-path or wave propagation time;
- scheduler time;
- graph-repair time;
- total router fraction of inference latency.

Routing should not become more expensive than encoding the selected observation.

---

## 14. Required ablations

- uniform versus random versus metadata-balanced;
- uncertainty greedy;
- single-wave versus multi-wave;
- no balancing versus load balancing;
- static graph versus dynamic local repair;
- analytic utility versus learned gain;
- one-step greedy versus receding horizon;
- fixed budget versus adaptive stopping.

Evaluate full quality–budget and quality–latency curves.

---

## 15. Failure modes

- accidental use of unqueried pixels;
- router repeatedly selecting adjacent redundant slices;
- one wave consuming most of the budget;
- uncertainty concentrated on representation artifacts rather than missing evidence;
- learned utility overfitting to dataset slice positions;
- graph rebuilding dominating runtime;
- reporting global optimality for a dynamically changing route when only each fixed-state shortest-path subproblem is exact.
