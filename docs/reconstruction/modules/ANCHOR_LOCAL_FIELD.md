# Module Architecture — Anchor-Local Structural Field

## 1. Purpose

Represent local 3D structure around adaptive anchors using one shared tiny MLP.

This is the currently locked decoder decision. The tiny MLP must have the smallest possible role:

> map an already organized local coordinate and compact anchor evidence to one scalar local structural-field value.

It must not absorb the responsibilities of the encoder, physical alignment, multi-plane aggregation, routing, or final appearance reconstruction.

---

## 2. Anchor definition

A representation anchor is a patient-specific geometric support point:

```text
Anchor_i
├── physical center a_i
├── local frame R_i
├── support scale s_i
├── compact evidence h_i
├── geometry confidence
├── modality observability
├── contributing-plane references
└── topology status
```

An anchor is not an observation slice and is not itself a Gaussian. It is the reference frame from which a local field and one or more Gaussian primitives are parameterized.

---

## 3. Local coordinates

For a physical query point \(\mathbf x\), define:

\[
\boldsymbol\xi_i(\mathbf x)
=\mathbf S_i^{-1}\mathbf R_i^\top
(\mathbf x-\mathbf a_i).
\]

- \(\mathbf a_i\): anchor center;
- \(\mathbf R_i\): local orientation;
- \(\mathbf S_i\): local support scaling.

This transforms a patient-space coordinate into a normalized anchor-local coordinate.

---

## 4. Compact anchor evidence

For every observed plane near anchor \(i\), create an evidence token:

\[
\mathbf e_{ik}=
[
Z_k(\pi_k(\mathbf a_i)),
 d(\mathbf a_i,P_k),
 \mathbf n_k,
 \mathbf e_{m_k},
 c_k
].
\]

Aggregate the set into a fixed-size vector:

\[
\mathbf h_i
=\operatorname{Aggregate}
\{\mathbf e_{ik}\}_{k\in\mathcal N_i}.
\]

The aggregation step must resolve variable numbers of planes before the MLP is called.

---

## 5. Tiny MLP contract

### Input

\[
[\boldsymbol\xi_i(\mathbf x),\mathbf h_i].
\]

### Output

\[
f_i(\mathbf x)\in\mathbb R.
\]

The default output is one scalar local structural-field value.

Optional experiments may add a second scalar for predictive variance, but the core method should first test the one-output formulation.

---

## 6. Suggested architecture envelope

The exact dimensions are experimental, but the design envelope is intentionally small:

```text
input dimension = local coordinate + compact evidence
→ linear layer
→ smooth activation
→ 2–4 hidden layers
→ width 16–64
→ scalar output
```

Candidate activations:

- Softplus;
- SiLU;
- sine activation if high-frequency behavior is justified;
- other smooth activations compatible with field gradients.

All anchors share the same MLP weights.

---

## 7. Why local and shared

A large global implicit network must encode the complete patient anatomy and every local detail in one function. The local shared formulation instead distributes complexity through:

- anchor positions;
- local frames;
- support scales;
- evidence vectors;
- adaptive anchor density.

The shared MLP learns a reusable rule for how local evidence maps to local field shape.

---

## 8. Field blending

Each anchor has a compact support weight, for example:

\[
w_i(\mathbf x)=
\exp(-\|\boldsymbol\xi_i(\mathbf x)\|^2).
\]

The global field is

\[
F(\mathbf x)=
\frac{\sum_iw_i(\mathbf x)f_i(\mathbf x)}
{\sum_iw_i(\mathbf x)+\epsilon}.
\]

Alternative partition-of-unity kernels may be evaluated. The blend must be numerically stable and differentiable.

---

## 9. SDF versus general structural field

The method may enforce SDF behavior through:

- sign supervision where available;
- Eikonal regularization;
- zero-level consistency;
- local gradient regularity.

If these conditions are not sufficiently met, the representation should be described as a local structural level-set field rather than a true SDF.

---

## 10. Frame construction

At the first bootstrap, a local frame may be initialized from:

- observed plane orientation;
- local evidence gradients;
- neighboring anchor geometry;
- population-level orientation prior.

After a stable field exists, the normal may be derived from

\[
\mathbf n_i=
\frac{\nabla F(\mathbf a_i)}
{\|\nabla F(\mathbf a_i)\|+\epsilon}.
\]

Two tangent directions span the plane orthogonal to \(\mathbf n_i\). They define how local coordinates move along the structural interface rather than through it.

---

## 11. Anchor adaptation

### Move

Update the anchor position when persistent residual indicates a displaced interface.

### Birth

Create a new anchor in a high-residual region without sufficient support.

### Split

Replace one anchor with multiple smaller supports when local evidence or residual is heterogeneous.

### Merge

Merge redundant anchors with compatible fields and evidence.

### Prune

Remove anchors with low support or persistent inconsistency.

The MLP weights do not change during these patient-specific operations.

---

## 12. Continuity constraints

Overlapping local fields should not contradict each other strongly. A consistency loss may be applied at overlap points:

\[
\mathcal L_{overlap}
=
\sum_{i,j}
\sum_{\mathbf x\in\mathcal X_{ij}}
|f_i(\mathbf x)-f_j(\mathbf x)|.
\]

A gradient-consistency variant may also be tested.

---

## 13. Computational strategy

For each query point:

1. find nearby anchors using a spatial index;
2. evaluate only anchors whose support contains the point;
3. batch tiny-MLP evaluations across anchors and query points;
4. blend local outputs;
5. avoid evaluating the MLP for distant support.

The representation is sparse in stored primitives but continuous in query space.

---

## 14. Required ablations

- global MLP versus shared local tiny MLP;
- MLP depth and width;
- local coordinate only versus coordinate + evidence;
- fixed versus adaptive anchor density;
- Euclidean versus field-derived frames;
- isotropic versus anisotropic local support;
- initialization-only versus persistent anchor constraints;
- no overlap loss versus value/gradient consistency.

---

## 15. Failure modes

- tiny MLP forced to infer modality alignment from raw features;
- anchor evidence too weak for the MLP to disambiguate local shape;
- field discontinuities between neighboring anchors;
- too few anchors in high-curvature regions;
- excessive anchors causing compute and overlap explosion;
- incorrect local frames creating distorted geometry;
- misleading use of the term SDF without distance-like behavior.
