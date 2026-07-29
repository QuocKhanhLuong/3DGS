# Module Architecture — SDF-Constrained Gaussian Memory

## 1. Purpose

Maintain a continuous, adaptive patient-specific 3D representation using Gaussian primitives whose geometry is organized by anchors and local structural fields.

A Gaussian is not merely a point. It is a continuous ellipsoidal basis function centered at a point and carrying geometry, appearance, and confidence.

---

## 2. Dual-bank representation

Because the primary task is full 3D MRI reconstruction, the representation uses two complementary banks.

```text
GaussianMemory
├── Structural surface bank G_surface
└── Volumetric appearance bank G_volume
```

### Structural surface bank

- thin anisotropic support;
- aligned with local structural interfaces;
- constrained by anchors and field normals;
- preserves edges, boundaries, and high-frequency geometry.

### Volumetric appearance bank

- thicker 3D support;
- represents internal tissue intensity;
- carries modality-specific appearance attributes;
- less tightly constrained to a surface.

A surface-only representation cannot faithfully fill the interior intensity field required for reconstruction.

---

## 3. Gaussian primitive

For primitive \(i\):

\[
g_i=(
\boldsymbol\mu_i,
\boldsymbol\Sigma_i,
\alpha_i,
\mathbf c_i,
\mathbf u_i,
\tau_i
).
\]

- \(\boldsymbol\mu_i\): center;
- \(\boldsymbol\Sigma_i\): 3D covariance;
- \(\alpha_i\): support strength or opacity-like weight;
- \(\mathbf c_i\): modality appearance values or compact codes;
- \(\mathbf u_i\): uncertainty and observability;
- \(\tau_i\): primitive type and topology metadata.

Its spatial influence is

\[
w_i(\mathbf x)=
\alpha_i
\exp\left[
-\frac12
(\mathbf x-\boldsymbol\mu_i)^\top
\boldsymbol\Sigma_i^{-1}
(\mathbf x-\boldsymbol\mu_i)
\right].
\]

---

## 4. Structural Gaussian parameterization

For anchor \(\mathbf a_i\), local frame

\[
\mathbf R_i=[\mathbf t_{1,i},\mathbf t_{2,i},\mathbf n_i],
\]

and tangent/normal scales, define

\[
\boldsymbol\Sigma_i=
\mathbf R_i
\operatorname{diag}(
\sigma_{t1,i}^2,
\sigma_{t2,i}^2,
\sigma_{n,i}^2
)
\mathbf R_i^\top.
\]

Typically:

\[
\sigma_{n,i}\ll\sigma_{t1,i},\sigma_{t2,i}.
\]

The center is constrained around the anchor:

\[
\boldsymbol\mu_i=
\mathbf a_i
+\delta u_i\mathbf t_{1,i}
+\delta v_i\mathbf t_{2,i}
+\delta n_i\mathbf n_i.
\]

Normal displacement should be regularized more strongly than tangent displacement.

---

## 5. Volumetric Gaussian parameterization

Volumetric primitives may use:

- isotropic or mildly anisotropic covariance;
- larger support radius;
- center initialized inside represented tissue regions;
- modality-specific appearance slots;
- optional relation to nearby structural anchors.

They should not be completely free without regularization. Useful constraints include:

- support within patient bounds;
- bounded scale;
- overlap control;
- coverage reward;
- consistency with observed slice intensities;
- uncertainty proportional to evidence distance.

---

## 6. Appearance state

For modalities \(m\in\mathcal M\), store either direct values

\[
\mathbf c_i=[c_{i,1},\ldots,c_{i,|\mathcal M|}]
\]

or compact modality-conditioned codes.

The low-complexity default is direct per-modality appearance slots so final reconstruction can use normalized Gaussian blending without a large decoder.

Unobserved slots retain uncertainty rather than being treated as known values.

---

## 7. Memory initialization

### Structural primitives

Initialized from refined anchors and local field geometry.

### Volumetric primitives

Possible initialization strategies:

- sparse sampling inside provisional structural regions;
- coverage-based placement between observed planes;
- residual-driven placement after initial slice rendering;
- coarse lattice followed by adaptive pruning;
- anchor-conditioned interior offsets.

The exact strategy remains an experimental choice, but initialization must not use hidden target intensities.

---

## 8. Memory assimilation

For a newly observed plane:

1. render the plane from the previous memory;
2. calculate residual;
3. identify intersecting primitive supports;
4. update modality appearance in affected volumetric primitives;
5. update structural primitives only when persistent evidence indicates geometric error;
6. update uncertainty and coverage;
7. consider topology operations.

Appearance should update faster than shared geometry.

---

## 9. Topology operations

### Birth

Create primitive support in unexplained high-residual regions.

### Split

Split when one primitive covers heterogeneous intensity or geometry.

### Merge

Merge compatible overlapping primitives.

### Prune

Remove primitives with low contribution, low support, or persistent inconsistency.

Operations should be evaluated against a common objective balancing reconstruction error and representation complexity.

---

## 10. Continuous reconstruction

For modality \(m\):

\[
\hat V^m(\mathbf x)=
\frac{\sum_iw_i(\mathbf x)c_{i,m}}
{\sum_iw_i(\mathbf x)+\epsilon}.
\]

Structural and volumetric banks may be jointly normalized or fused with a small analytic gate based on distance to the structural field.

---

## 11. Uncertainty and observability state

Each Gaussian may store:

```text
GaussianObservability
├── geometry confidence
├── support coverage
├── per-modality evidence count
├── per-modality uncertainty
├── disagreement score
├── residual history
└── last update round
```

This state feeds the trajectory utility and final uncertainty volume.

---

## 12. Complexity controls

- maximum structural primitive count;
- maximum volumetric primitive count;
- support-neighbor cap;
- minimum contribution threshold;
- merge radius;
- split acceptance margin;
- primitive-count regularizer;
- chunked spatial indexing.

The method must report primitive count and memory usage as a function of observation budget.

---

## 13. Required ablations

- free Gaussians;
- SDF/field initialization only;
- persistent structural constraint;
- surface bank only;
- volumetric bank only;
- dual-bank representation;
- fixed topology versus adaptive topology;
- direct appearance values versus decoded appearance codes;
- isotropic versus anisotropic structural support;
- shared versus separate modality geometry.

---

## 14. Core hypothesis

The geometry of a sparse reconstruction representation should be carried primarily by anchor positions, local frames, support scales, and adaptive primitive density, while the shared tiny MLP and appearance composition remain low-capacity. This shifts complexity from a large global neural decoder into an interpretable patient-specific spatial memory.
