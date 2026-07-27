# Pipeline — From Sparse Multi-Sequence Queries to a Patient-Specific Gaussian Prior

## 1. Execution modes

The same architecture has two distinct flows:

- **training**: complete registered volumes exist as hidden supervision, but the agent only observes queried slices during an episode;
- **inference**: the model queries slices sequentially, updates the patient state, and stops at an observability fixed point or safety budget.

Candidate images must never be loaded before the routing decision. Otherwise the query-budget claim is invalid.

---

## 2. End-to-end inference flow

```text
[0] Patient slice pool remains on CPU/disk
        │
        ▼
[1] Select 3–5 anchor observations
        │
        ▼
[2] Encode structural evidence
        │
        ▼
[3] Build coarse SDF scaffold and uncertainty
        │
        ▼
[4] Sample SDF anchors and initialize low-DoF Gaussians
        │
        ▼
[5] Construct feature–geometry observation graph
        │
        ▼
[6] Start parallel information waves from support clusters
        │
        ▼
[7] Global scheduler selects complementary query batch
        │
        ▼
[8] Load only selected sequence–slice observations
        │
        ▼
[9] Structural + appearance encoding
        │
        ▼
[10] Render current prediction before update
        │
        ▼
[11] Compute residual and local affected primitives
        │
        ▼
[12] Assimilate evidence into Gaussian memory
        │
        ├── fast appearance update
        ├── constrained geometry update
        ├── slow SDF correction
        ├── uncertainty update
        └── birth / split / prune
        │
        ▼
[13] Repair local graph costs and wavefronts
        │
        ▼
[14] Evaluate convergence
        │
        ├── no: return to [6]
        └── yes: output segmentation/reconstruction/uncertainty
```

---

## 3. Stage 0 — Data preparation

### Required preprocessing

1. register all sequences to one patient coordinate system;
2. resample or preserve exact physical metadata;
3. normalize intensity per sequence;
4. crop to brain/ROI while retaining affine transforms;
5. index observations by `(patient, modality, slice, plane)`;
6. keep full volumes on CPU/NVMe during routing.

### Output schema

```python
ObservationMeta(
    patient_id,
    modality,
    slice_index,
    affine,
    spacing,
    plane_origin,
    plane_normal,
)
```

---

## 4. Stage 1 — Anchor scouting

### Input

Only metadata and the small anchor budget.

### MVP policy

Select spatially separated slices and complementary modalities. A practical start is a central structural slice plus superior/inferior anchors.

### Output

\[
\mathcal A_0=\{(m_k,z_k)\}_{k=1}^{K_0}.
\]

Load and encode only these observations.

---

## 5. Stage 2 — Structural SDF initialization

For each anchor observation:

1. compute sequence-invariant structural features;
2. lift the 2D features to the physical slice plane;
3. update a local implicit structural field;
4. estimate structural uncertainty in unsupported regions.

The SDF is initialized coarsely. It is not assumed correct after the anchor phase.

Outputs:

- field value \(F_0(\mathbf x)\);
- normals \(\nabla F_0\);
- optional curvature;
- uncertainty \(U_F\).

---

## 6. Stage 3 — Low-DoF Gaussian prior initialization

For each selected SDF support anchor \(\mathbf a_i\):

1. compute the normalized SDF gradient \(\mathbf n_i\);
2. construct tangent vectors \(\mathbf t_{1,i},\mathbf t_{2,i}\);
3. initialize center from the anchor and small local residuals;
4. derive orientation from the tangent-normal frame;
5. initialize tangent/normal scales from local spacing and uncertainty;
6. initialize opacity and compact tissue code;
7. initialize modality-specific observability.

The resulting Gaussian geometry is not free 3D position + quaternion + three scales. It is constrained by the SDF manifold.

---

## 7. Stage 4 — Candidate observation graph

### Node construction

A node represents a sequence–slice region rather than an already loaded image:

\[
v=(m,z,r).
\]

Its descriptor is predicted from:

- physical plane metadata;
- intersected Gaussian/SDF support;
- pooled feature state;
- uncertainty and coverage;
- trajectory history.

### Edges

Create spatial, cross-modality, SDF-geodesic, and feature-similarity edges.

### Constraint

No candidate descriptor may use the unseen image content.

---

## 8. Stage 5 — Multi-wave routing

### Initialization

Partition support anchors into \(K\) source clusters. Each source initializes an information front.

### Per-wave proposal

Each wave propagates over nonnegative dynamic costs and proposes its best frontier nodes. The cost decreases for high expected information and increases for distance, modality switches, redundancy, overlap, and load imbalance.

### Parallelism

Local wave expansions can run independently over shared read-only graph state, followed by synchronized scheduler arbitration.

### Global batch selection

Select a non-overlapping batch of proposals that maximizes total expected gain while maintaining balance across waves and anatomy.

For an MVP, batch size can be 1–4.

---

## 9. Stage 6 — Query and encode

After the scheduler commits to \(\mathcal B_t\):

1. load only the selected slices from CPU/disk;
2. transfer them to GPU;
3. run structural and appearance encoders;
4. release raw slice activations after assimilation when possible.

This streaming pattern avoids full-volume feature fusion and limits peak VRAM.

---

## 10. Stage 7 — Render-before-update

For each selected observation \(a=(m,z)\):

1. render the current predicted slice \(\hat I_a\);
2. optionally render predicted segmentation/logits;
3. compute image, structural, and task residuals;
4. estimate whether the observation is novel, redundant, or contradictory.

\[
E_a=I_a-R(\mathcal G_t,m,z).
\]

The residual is the message used by the assimilation block.

---

## 11. Stage 8 — Local evidence assimilation

### Affected set

Retrieve only Gaussians whose support intersects the queried plane/region.

### Fast Gaussian update

Update:

- compact appearance/tissue code;
- modality evidence;
- opacity;
- local tangent/normal scale;
- constrained tangent/normal offsets;
- uncertainty.

### Slow SDF update

Modify the structural field only when residuals are structural, persistent, and cross-sequence compatible.

### Manifold projection

After geometry updates, optionally project centers back toward the current SDF level set.

---

## 12. Stage 9 — Topology adaptation

Generate diagnostics from residuals, curvature, coverage, and support.

- **birth**: high residual with missing local support;
- **split**: large primitive with heterogeneous residual or high curvature;
- **prune**: unsupported, inconsistent, low-opacity, or off-manifold primitive.

Reject operations that do not decrease the common energy by a required margin.

Near convergence, freeze topology.

---

## 13. Stage 10 — Graph and wave repair

After a local memory update:

1. identify changed SDF/Gaussian regions;
2. invalidate only candidate descriptors intersecting those regions;
3. recompute local utilities and affected edges;
4. repair wave arrival costs/frontiers;
5. update shared coverage and load statistics.

This is cheaper than rebuilding every candidate score after every observation.

---

## 14. Stage 11 — Convergence and stopping

A round is considered stable when:

1. maximum marginal information gain is below \(\epsilon_I\);
2. relative SDF–Gaussian state change is below \(\epsilon_\theta\);
3. residual on observed slices is below \(\tau_D\);
4. worst-case ROI uncertainty is below \(\tau_U\);
5. topology has not changed;
6. the conditions persist for \(P\) consecutive rounds.

The safety budget \(B_{max}\) is an emergency stop, not the primary convergence definition.

Outputs include:

```text
status = converged | insufficiently_observed
queried_trajectory
patient_gaussian_state
segmentation / reconstructed volume
uncertainty map
convergence diagnostics
```

---

## 15. Training pipeline

### Phase 1 — Representation pretraining

Use random and uniformly spaced sparse trajectories to train:

- structural encoder/SDF;
- Gaussian initializer;
- slice renderer;
- local assimilation;
- auxiliary reconstruction.

Supervise queried slices plus a small random set of held-out slices. Do not render all slices at every step.

### Phase 2 — Utility/observability learning

For sampled states, evaluate a manageable candidate subset offline. Measure actual improvement after assimilating each candidate and train a utility/ranking predictor.

Possible target:

\[
\Delta Q(a)=Q(\mathcal G_{t+1}^{a})-Q(\mathcal G_t).
\]

### Phase 3 — Multi-wave routing training

Train or calibrate:

- dynamic edge costs;
- candidate utility;
- balancing/load penalty;
- overlap penalty;
- scheduler.

Start with greedy/imitation learning rather than reinforcement learning.

### Phase 4 — Joint fine-tuning

Unroll 4–6 routing/update steps, use truncated backpropagation, and detach persistent state between windows.

### Phase 5 — Convergence calibration

Select stopping thresholds on validation data to control:

- premature stopping rate;
- lesion miss rate;
- average queries;
- uncertainty calibration.

---

## 16. Loss flow

A practical total objective:

\[
\mathcal L=
\lambda_{obs}\mathcal L_{observed}
+\lambda_{held}\mathcal L_{heldout}
+\lambda_{seg}\mathcal L_{seg}
+\lambda_{sdf}\mathcal L_{SDF}
+\lambda_{man}\mathcal L_{manifold}
+\lambda_{cal}\mathcal L_{calibration}
+\lambda_{route}\mathcal L_{routing}
+\lambda_{topo}\mathcal L_{topology}.
\]

Recommended first-paper priority:

1. segmentation loss;
2. held-out reconstruction/structural consistency;
3. SDF/manifold regularization;
4. uncertainty calibration;
5. routing ranking loss.

---

## 17. Evaluation flow

Run each method under the same query budgets:

\[
B\in\{4,8,16,24,32,48\}.
\]

Baselines:

- random;
- uniform spacing;
- central-first;
- equal modality allocation;
- fixed learned trajectory;
- uncertainty-only selection;
- single-wave routing;
- proposed balanced multi-wave routing;
- full-volume upper bound.

Report:

- Dice, lesion-wise Dice, HD95, lesion recall;
- PSNR/SSIM/NMSE for auxiliary reconstruction;
- quality–budget AUC;
- queries to reach 90%/95% of full-input performance;
- peak VRAM, runtime, number of Gaussians;
- convergence success and premature-stop rate;
- predicted versus actual information-gain correlation.

---

## 18. Inference pseudocode

```python
@torch.no_grad()
def build_patient_state(provider, anchor_policy, router, scheduler, model, stop):
    observed = set()

    anchors = anchor_policy.select(provider.metadata)
    anchor_obs = [provider.load(a) for a in anchors]

    sdf = model.initialize_sdf(anchor_obs)
    memory = model.initialize_gaussians(sdf, anchor_obs)
    observed.update(anchors)

    graph = model.build_observation_graph(memory, provider.metadata, observed)
    router.initialize(graph, memory.support_clusters())

    while True:
        proposals = router.propose(memory, graph, observed)
        batch = scheduler.select(proposals, memory, graph)

        if not batch:
            break

        observations = [provider.load(a) for a in batch]

        diagnostics = []
        for a, obs in zip(batch, observations):
            prediction = memory.render_slice(a.modality, a.plane)
            residual = model.compute_residual(obs, prediction)
            features = model.encode_observation(obs)
            d = memory.assimilate(a, features, residual, sdf)
            diagnostics.append(d)
            observed.add(a)

        sdf.slow_update(diagnostics)
        memory.adapt_topology(diagnostics, sdf)

        changed = model.changed_regions(diagnostics)
        graph.repair(memory, changed, observed)
        router.repair(graph, changed)

        stop.update(memory, graph, diagnostics, proposals)
        if stop.should_stop() or stop.budget_exhausted():
            break

    output = model.decode_outputs(memory, sdf)
    output["trajectory"] = list(observed)
    output["status"] = stop.status
    return output
```

---

## 19. Initial repository implementation order

1. data provider and registered slice metadata;
2. plane-aware Gaussian slice renderer;
3. low-DoF SDF-constrained Gaussian data structure;
4. fixed-anchor initialization;
5. local assimilation and auxiliary reconstruction;
6. segmentation head;
7. uncertainty state;
8. static candidate graph;
9. single-wave greedy router;
10. multi-wave balancing and scheduler;
11. convergence controller;
12. topology adaptation and incremental graph repair.
