# Phase 2 — Initial Anchor Bootstrap

## 1. Objective

Create the first patient-specific structural scaffold and Gaussian memory from a small set of legally selected MRI slices.

This phase must solve a circular dependency:

```text
anchors are needed to define local fields,
but a field is needed to know where good anchors should lie.
```

The solution is a two-stage bootstrap:

1. generate **provisional anchors** directly from observed physical planes;
2. use shared local fields to refine those anchors into a coherent 3D scaffold.

---

## 2. Inputs and outputs

### Inputs

```text
InitialObservationSet
├── committed sequence–slice actions
├── observed pixels
├── physical plane geometry
├── modality identity
├── spacing and thickness
└── global frozen model parameters
```

### Outputs

```text
InitialPatientState
├── observation ledger
├── compact evidence cache
├── provisional/refined anchors
├── local field bundle
├── structural Gaussian bank
├── volumetric appearance Gaussian bank
├── observability state
└── uncertainty state
```

---

## 3. Initial observation selection

The initial selector operates before patient-specific image evidence exists. It therefore uses only legal metadata and population priors.

A practical objective is balanced physical coverage:

\[
\mathcal O_0=
\arg\max_{|\mathcal O|=K_0}
\left[
\operatorname{Coverage}(\mathcal O)
-\lambda_R\operatorname{Redundancy}(\mathcal O)
-\lambda_B\operatorname{Imbalance}(\mathcal O)
\right].
\]

Possible legal descriptors:

- physical slice position;
- modality;
- plane orientation;
- slice thickness;
- training-cohort region priors;
- acquisition cost.

No unqueried pixel may be inspected.

---

## 4. Evidence-cache creation

For each committed initial slice:

```text
slice pixels
→ lightweight evidence encoder
→ compact feature map Z
→ optional reliability map C
→ cache with physical metadata
```

Each slice is encoded once. Later anchor movement, birth, and split use interpolation from the cache rather than re-running the encoder.

The exact encoder architecture is deliberately not fixed here. The required invariant is:

\[
E(I_{m,z})\rightarrow
(Z_{m,z},C_{m,z}),
\]

where the map remains spatially aligned with the physical MRI plane.

---

## 5. Provisional structural candidates

For every observed plane, create a sparse candidate set from the compact evidence map.

Candidate score may combine:

- local structural response;
- multi-scale contrast;
- feature change magnitude;
- local reliability;
- non-maximum suppression;
- distance from already selected candidates;
- cross-modality agreement when registered planes overlap.

The bootstrap must not retain every pixel as an anchor. It should produce an adaptive sparse support set.

---

## 6. Physical lifting

A pixel location \((u,v)\) on plane \(P_k\) is mapped into physical space:

\[
\mathbf a=\mathbf o_k
+u\,\Delta_u\mathbf r_k
+v\,\Delta_v\mathbf c_k.
\]

Here:

- \(\mathbf o_k\) is the plane origin;
- \(\mathbf r_k,\mathbf c_k\) are physical row and column directions;
- \(\Delta_u,\Delta_v\) are pixel spacings.

The lifted point is a provisional representation anchor, not yet a guaranteed point on the final structural surface.

---

## 7. Cross-plane consolidation

Candidate anchors from different observed slices may describe the same 3D region. Consolidation should:

- merge physically close candidates;
- retain modality-specific evidence separately;
- prevent dense duplicate support near intersecting planes;
- preserve conflicting evidence as uncertainty rather than averaging it away;
- maintain coverage of disconnected regions.

A merged anchor stores references to every contributing cached plane.

---

## 8. Anchor evidence aggregation

For anchor \(i\) and observed plane \(k\), create an evidence token:

\[
\mathbf e_{ik}=
[
Z_k(\pi_k(\mathbf a_i)),
 d(\mathbf a_i,P_k),
 \mathbf n_k,
 \mathbf e_{m_k},
 C_k(\pi_k(\mathbf a_i))
].
\]

The anchor evidence is

\[
\mathbf h_i=
\operatorname{Aggregate}
\{\mathbf e_{ik}\}_{k\in\mathcal N_i}.
\]

The default aggregation should be geometry- and confidence-weighted. Attention may be evaluated later but is not required by the interface.

---

## 9. Local field decoding

Each provisional anchor defines a local coordinate system. For a query point \(\mathbf x\):

\[
\boldsymbol\xi_i(\mathbf x)=
\mathbf R_i^\top
(\mathbf x-\mathbf a_i)/\mathbf s_i.
\]

The one shared tiny MLP predicts a scalar local structural value:

\[
f_i(\mathbf x)=
M_{tiny}(
\boldsymbol\xi_i(\mathbf x),
\mathbf h_i
).
\]

The MLP should not perform modality alignment, global anatomical reasoning, or trajectory planning. Those responsibilities must be resolved before its input is formed.

---

## 10. Global structural scaffold

Blend local fields using compact support weights:

\[
F(\mathbf x)=
\frac{
\sum_i w_i(\mathbf x)f_i(\mathbf x)
}{
\sum_i w_i(\mathbf x)+\epsilon
}.
\]

The field may be interpreted as an SDF when Eikonal and sign constraints are enforced, or as a general structural level-set field when exact signed distance is not justified.

---

## 11. Anchor refinement

After a coarse global field exists, refine provisional anchors.

Potential operations:

1. project anchor toward a stable level set;
2. estimate local normal from the field gradient;
3. derive a tangent frame;
4. merge anchors with redundant support;
5. split support in regions of high curvature or heterogeneous evidence;
6. assign high uncertainty to anchors far from all observations;
7. reject anchors unsupported by reconstruction or structure evidence.

Anchor confidence is not simply the distance between a Gaussian and its anchor. It summarizes confidence in local geometry and evidence coverage.

---

## 12. Gaussian initialization

### Structural bank

Thin anisotropic Gaussians are initialized near important structural interfaces:

- center near refined anchor;
- orientation derived from the local field;
- large tangent support;
- small normal thickness;
- geometry confidence and modality observability state.

### Volumetric appearance bank

3D reconstruction also requires interior intensity support. Volumetric Gaussians are initialized:

- inside or around represented anatomical regions;
- with larger, less surface-constrained covariance;
- with modality-specific appearance slots;
- with confidence derived from observed-plane proximity and support.

A surface-only bank cannot faithfully reconstruct internal MRI intensity distributions.

---

## 13. Bootstrap uncertainty

Initial uncertainty should combine:

- distance to nearest observed plane;
- number and diversity of supporting modalities;
- disagreement between observed planes;
- local field disagreement;
- anchor density and coverage;
- provisional Gaussian support quality.

Unobserved regions may contain provisional support, but must remain explicitly uncertain so that the trajectory is encouraged to inspect them.

---

## 14. Bootstrap acceptance checks

Before entering the active loop, verify:

- at least one legal observation has been committed;
- every cached feature has matching physical metadata;
- anchors are finite and lie within the patient coordinate bounds;
- local field weights are nonnegative and numerically stable;
- no hidden target value entered the state;
- initial representation can render all committed slices;
- uncertainty is high outside observed support rather than falsely confident;
- structural and volumetric Gaussian banks are distinguishable in diagnostics.
