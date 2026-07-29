# Module Architecture — Physical-Plane Renderer and 3D Reconstructor

## 1. Purpose

Render observed or hidden MRI slices from the current Gaussian representation and reconstruct complete 3D modality volumes at arbitrary physical coordinates.

MRI slices are physical cross-sections through a registered volume. They are not perspective-camera views. The renderer must therefore use plane geometry rather than camera projection.

---

## 2. Inputs

```text
RendererInput
├── target physical plane or 3D coordinate set
├── requested modality
├── structural Gaussian bank
├── volumetric appearance Gaussian bank
├── local structural field
├── uncertainty state
└── target output spacing and shape
```

---

## 3. Plane representation

A plane is defined by:

```text
PhysicalPlane
├── origin o
├── row direction r
├── column direction c
├── normal n
├── pixel spacing delta_u, delta_v
├── thickness delta_n
└── output shape H, W
```

Each pixel coordinate \((u,v)\) maps to

\[
\mathbf x(u,v)=
\mathbf o
+u\Delta_u\mathbf r
+v\Delta_v\mathbf c.
\]

---

## 4. Gaussian–plane interaction

For Gaussian \(i\), decompose its covariance relative to the plane into tangent and normal components.

For an infinitesimally thin slice, use the conditional in-plane covariance:

\[
\Sigma_{slice}
=
\Sigma_{tt}
-\Sigma_{tn}
\Sigma_{nn}^{-1}
\Sigma_{nt}.
\]

The normal-distance gate may be

\[
g_i(P)=
\exp\left(
-\frac{d(\boldsymbol\mu_i,P)^2}
{2\sigma_{n,i}^2}
\right).
\]

The in-plane footprint is a 2D Gaussian whose center is the projection of \(\boldsymbol\mu_i\) onto the plane.

---

## 5. Finite slice thickness

Real MRI slices have nonzero thickness. The renderer should support slab integration or an approximation that integrates Gaussian support along the plane normal.

A finite-thickness formulation is preferred over treating every slice as infinitely thin when the dataset has meaningful through-plane thickness.

The implementation must record whether it uses:

- conditional thin-plane rendering;
- finite-slab integration;
- sampled quadrature through thickness;
- another physically justified approximation.

---

## 6. Intensity composition

For modality \(m\), the default composition is normalized additive blending:

\[
\hat I_m(\mathbf x)=
\frac{
\sum_iw_i(\mathbf x)c_{i,m}
}{
\sum_iw_i(\mathbf x)+\epsilon
}.
\]

This is more appropriate than front-to-back alpha compositing when the goal is a physical MRI intensity field rather than an opaque camera scene.

Structural and volumetric banks may be jointly normalized or combined with a small field-dependent gate.

---

## 7. Render-before-update mode

For a newly selected observation, the renderer must predict the slice before the new pixels enter the state:

```text
old patient state
→ render selected physical plane
→ predicted slice
→ load true slice
→ residual
→ local update
```

This mode provides the residual used for assimilation, topology changes, and uncertainty update.

---

## 8. Hidden-target training mode

During direct sparse episodic training:

1. build representation only from observed set \(\mathcal O\);
2. receive hidden target coordinates;
3. render hidden slices or sampled 3D points;
4. only then read hidden target intensity values for loss computation.

Hidden target content must not affect neighbor search, Gaussian selection, or field construction before rendering.

---

## 9. Full-volume reconstruction mode

For target grid \(\mathcal X\):

```text
partition target grid into chunks
→ spatially query relevant Gaussians and anchors
→ evaluate local fields
→ evaluate Gaussian weights
→ compose each requested modality
→ compose uncertainty
→ write output chunk
```

Chunking prevents memory from scaling with the entire dense volume at once.

---

## 10. Spatial acceleration

Possible acceleration structures:

- uniform hash grid;
- voxel-to-Gaussian index;
- k-d tree;
- bounding-volume hierarchy;
- anchor-cluster index;
- tiled plane rasterization.

The final renderer should evaluate only primitives whose bounded support intersects the target plane or volume chunk.

---

## 11. Structural-field use

The structural field may influence reconstruction by:

- aligning thin structural Gaussians;
- modulating structural versus volumetric contributions;
- providing edge-aware interpolation;
- defining support regions;
- producing geometry and normal outputs;
- contributing to uncertainty near unstable interfaces.

The field should not replace volumetric appearance support for interior MRI reconstruction.

---

## 12. Uncertainty rendering

Possible pointwise components:

- distance to observed planes;
- denominator/support mass;
- modality evidence count;
- variance among contributing Gaussians;
- residual history of contributing primitives;
- disagreement among local fields;
- learned calibrated uncertainty.

The renderer should be able to return both reconstructed intensity and uncertainty for the same target coordinates.

---

## 13. Output contracts

### Slice output

```text
RenderedSlice
├── intensity [H, W]
├── uncertainty [H, W]
├── support mass [H, W]
├── physical plane metadata
└── contributing-primitive diagnostics
```

### Volume output

```text
RenderedVolume
├── intensity [D, H, W]
├── uncertainty [D, H, W]
├── structural field optional
├── affine / orientation
├── modality id
└── reconstruction status
```

---

## 14. Required ablations

- thin-plane versus finite-slab rendering;
- normalized additive versus alternative composition;
- structural bank only;
- volumetric bank only;
- dual-bank rendering;
- direct modality values versus tiny appearance decoder;
- full primitive evaluation versus spatial culling;
- fixed versus adaptive Gaussian support;
- renderer accuracy versus latency at multiple output resolutions.

---

## 15. Validation tests

- one isotropic Gaussian rendered on an aligned plane;
- one anisotropic Gaussian rendered on a rotated plane;
- analytic versus sampled slab integration;
- invariance under equivalent coordinate-frame transforms;
- stable output when no primitive contributes;
- exact re-rendering of synthetic known Gaussian fields;
- gradient checks for centers, covariance, and appearance;
- physical-affine round-trip test;
- deterministic render-before-update behavior;
- hidden-target leakage test.

---

## 16. Failure modes

- treating MRI slices as perspective views;
- using alpha compositing without physical justification;
- ignoring slice thickness;
- incorrect affine orientation;
- dense all-Gaussian evaluation dominating runtime;
- unsupported voxels receiving confident interpolated values;
- final reconstruction silently using hidden full-volume features;
- large appearance decoder becoming the true reconstruction bottleneck.
