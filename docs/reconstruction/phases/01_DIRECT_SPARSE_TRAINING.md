# Phase 1 — Teacher-Free Permanently Sparse Training

## 1. Objective

Train the evidence encoder and the downstream reconstruction system without teacher distillation and without complete-volume supervision in the main training path.

The role of Phase 1 is narrow and explicit:

> learn compact, spatially aligned structural evidence from permanently sparse multi-sequence MRI slices, then pass that evidence into the already-fixed anchor, local-field, Gaussian, propagation, and rendering pipeline.

The downstream architecture remains:

```text
teacher-free evidence encoder
→ compact structural and appearance maps
→ provisional physical anchors
→ shared tiny anchor-local field
→ SDF/level-set constrained Gaussians
→ anchor–Gaussian propagation
→ latent 3D Gaussian representation
→ sparse target-plane rendering
```

Phase 1 does not redesign anchor propagation, trajectory routing, Gaussian memory, or final reconstruction.

---

## 2. Main data regime

For training patient \(i\), only a fixed sparse acquisition set is legally available:

\[
\Omega_i^{sparse}
=
\{(a_{i,j},I_{i,j})\}_{j=1}^{K_i},
\]

where \(a_{i,j}\) contains modality and physical-plane metadata.

The main training loader must not access any patient slice outside \(\Omega_i^{sparse}\).

A training episode splits only the permanently available sparse set:

\[
\mathcal C_i\subset\Omega_i^{sparse},
\qquad
\mathcal Q_i\subset\Omega_i^{sparse}\setminus\mathcal C_i.
\]

- \(\mathcal C_i\): sparse context slices allowed to enter the encoder and patient representation;
- \(\mathcal Q_i\): acquired sparse target slices used only after state construction;
- slices outside \(\Omega_i^{sparse}\): unavailable to the main training pipeline.

```text
Permanently sparse patient set Ω_i^sparse
        ├── context C_i
        │     └── may enter encoder, anchors, fields, and Gaussians
        └── target Q_i
              └── pixel values revealed only for reconstruction loss
```

Fully sampled volumes may be retained in a physically separate audit split for final evaluation, leakage tests, and oracle ablations. They are not training targets for the main method.

---

## 3. Why teacher distillation is removed

The original motivation for distillation was valid: a very small encoder may fail to preserve enough structural evidence for the tiny local MLP to infer local 3D geometry.

However, a dense teacher introduces three problems:

1. it may process substantially more training data than the final sparse system;
2. it weakens the complete-volume-free training claim;
3. it can transfer semantic features that are not aligned with physical reconstruction geometry.

The replacement is not another large model. It is a teacher-free structural learning design:

\[
\boxed{
\text{analytic differential scaffold}
+
\text{high-resolution micro-CNN}
+
\text{direct structural constraints}
+
\text{sparse reconstruction supervision}
}
\]

---

## 4. Teacher-free structural evidence encoder

### 4.1 Analytic input scaffold

For each legally observed slice \(I_m\), construct fixed channels:

\[
\Phi(I_m)=
[
I_m,
\partial_x I_m,
\partial_y I_m,
\|\nabla I_m\|,
\Delta I_m,
C_{r_1}(I_m),
C_{r_2}(I_m)
].
\]

The exact bank remains configurable, but the baseline should include:

- normalized intensity;
- horizontal and vertical derivatives;
- gradient magnitude;
- Laplacian;
- local contrast at two or more scales;
- valid-content and optional artifact channels.

These channels supply transparent geometric bias and prevent the micro-CNN from relearning elementary differential operators from scratch.

### 4.2 Recommended core architecture

```text
analytic/raw channels
+ modality embedding or FiLM condition
        ↓
3×3 high-resolution stem
        ↓
small depthwise residual trunk
        ↓
limited downsampling: stride 1, 2, or 4
        ↓
compact projections
        ├── structural map Z_str
        ├── appearance map Z_app
        └── reliability map C, optional
```

Recommended initial capacity:

- shared trunk across modalities;
- small modality-specific normalization or FiLM parameters;
- 16 structural channels;
- 8 appearance channels;
- no large U-Net decoder;
- no pretrained teacher in the main path;
- one encoding pass per legally observed slice.

The exact channel counts and stride remain ablation variables, not theoretical claims.

### 4.3 Output contract

\[
(Z_k^{str},Z_k^{app},C_k)
=
E_\theta(\Phi(I_k),m_k,P_k).
\]

The maps must remain spatially aligned with the MRI plane so that later anchors can sample them at projected physical coordinates.

---

## 5. Structural and appearance separation

MRI modalities share anatomy but not intensity formation. The encoder therefore separates:

- \(Z^{str}\): modality-robust structural evidence used primarily by the local field and anchor logic;
- \(Z^{app}\): modality-specific appearance evidence used by Gaussian intensity slots;
- \(C\): optional reliability estimate.

A simplified observation model is

\[
I_m(x)=g_m(S(x),A_m(x))+\epsilon_m(x),
\]

where \(S\) denotes shared structure and \(A_m\) denotes modality-specific appearance.

The method should align \(Z^{str}\) across registered modalities without forcing \(Z^{app}\) to be identical.

---

## 6. Teacher-free structural supervision

The encoder receives the final sparse reconstruction gradient, but additional direct constraints are required to avoid weak or unstable structural features.

### 6.1 Sparse predictive reconstruction

Build the patient representation from context only:

\[
\mathcal S_i
=
\operatorname{BuildState}_\Theta(\mathcal C_i).
\]

For target plane \(q\in\mathcal Q_i\):

\[
\hat I_q=R(\mathcal S_i,a_q,m_q).
\]

The main loss is

\[
\mathcal L_{pred}
=
\sum_{q\in\mathcal Q_i}
\rho(\hat I_q-I_q).
\]

Target pixels must not affect encoder features, anchor creation, Gaussian state, or propagation before rendering.

### 6.2 Spatial equivariance

For a geometry-preserving image transform \(T\):

\[
E^{str}_\theta(TI)
\approx
T(E^{str}_\theta(I)).
\]

\[
\mathcal L_{eq}
=
\|E^{str}_\theta(TI)-T(E^{str}_\theta(I))\|_1.
\]

This preserves the physical meaning of feature coordinates.

### 6.3 Intensity invariance

For an intensity-only transformation \(g\) that does not alter geometry:

\[
E^{str}_\theta(g(I))
\approx
E^{str}_\theta(I).
\]

\[
\mathcal L_{inv}
=
\|Z^{str}(g(I))-Z^{str}(I)\|_1.
\]

The appearance branch is not forced to satisfy the same invariance.

### 6.4 Cross-modality structural consistency

At registered physical location \(x\), sample corresponding structural features from modalities \(m\) and \(n\):

\[
z_m^{str}(x)=Z_m^{str}(\pi_m(x)),
\qquad
z_n^{str}(x)=Z_n^{str}(\pi_n(x)).
\]

\[
\mathcal L_{xmod}
=
\sum_x w(x)
\|z_m^{str}(x)-z_n^{str}(x)\|_1.
\]

The weight \(w(x)\) should decrease under registration uncertainty, artifact, invalid support, or weak local evidence.

### 6.5 Anti-collapse constraints

Cross-modality consistency alone admits a constant representation. Use variance and covariance penalties:

\[
\mathcal L_{var}
=
\sum_c
\max(0,\gamma-\operatorname{Std}(Z_c^{str})),
\]

\[
\mathcal L_{cov}
=
\sum_{p\neq q}
\operatorname{Cov}(Z_p^{str},Z_q^{str})^2.
\]

### 6.6 Local differential preservation

A training-only head may reconstruct compact structural quantities:

```text
Z_str
→ tiny auxiliary head
→ predicted gradient / Laplacian / local contrast
```

\[
\mathcal L_{local}
=
\|\widehat{\nabla I}-\nabla I\|_1
+
\lambda_\Delta
\|\widehat{\Delta I}-\Delta I\|_1.
\]

The auxiliary head is removed at inference.

---

## 7. Total Phase-1 objective

\[
\boxed{
\begin{aligned}
\mathcal L_{P1}
={}&
\mathcal L_{pred}
+
\lambda_{eq}\mathcal L_{eq}
+
\lambda_{inv}\mathcal L_{inv}\\
&+
\lambda_{xmod}\mathcal L_{xmod}
+
\lambda_{var}\mathcal L_{var}
+
\lambda_{cov}\mathcal L_{cov}\\
&+
\lambda_{local}\mathcal L_{local}
+
\lambda_{field}\mathcal L_{field}
+
\lambda_G\mathcal L_{Gaussian}.
\end{aligned}
}
\]

The sparse predictive reconstruction term remains the final task objective. Structural auxiliary terms stabilize the encoder but must not dominate end-to-end reconstruction.

---

## 8. Episode flow

```text
Select permanently sparse training patient
        │
        ▼
Read fixed sparse manifest Ω_i^sparse
        │
        ▼
Split available slices into context C_i and sparse targets Q_i
        │
        ▼
For context slices only:
raw image + analytic channels
→ teacher-free micro-CNN
→ cache Z_str, Z_app, C
        │
        ▼
Locked downstream path:
anchors
→ tiny local field
→ anchored Gaussians
→ propagation
→ latent 3D Gaussian state
        │
        ▼
Render only target planes Q_i
        │
        ▼
Reveal sparse target pixels
        │
        ▼
Reconstruction + structural auxiliary losses
        │
        ▼
Backpropagate global parameters
```

The preferred training unit is one patient episode. Different context/target roles may be sampled across epochs, but only within the patient's fixed sparse acquisition set.

---

## 9. Training schedule

### Stage E0 — Teacher-free structural warm-up

Use only legally available sparse slices. Optimize:

\[
\mathcal L_{eq}
+
\mathcal L_{inv}
+
\mathcal L_{xmod}
+
\mathcal L_{var}
+
\mathcal L_{cov}
+
\mathcal L_{local}.
\]

Purpose:

- prevent collapse;
- preserve spatial detail;
- learn modality-robust structural evidence;
- initialize the micro-CNN without a teacher.

### Stage E1 — Joint sparse reconstruction

Connect the complete locked reconstruction path and optimize:

\[
\mathcal L_{pred}
+
\lambda_{aux}\mathcal L_{structural}
+
\lambda_{field}\mathcal L_{field}
+
\lambda_G\mathcal L_{Gaussian}.
\]

### Stage E2 — End-to-end refinement

Reduce auxiliary weights and let sparse predictive reconstruction dominate. Learned routing is not required in this stage; use fixed or analytic observation selection to isolate representation quality.

---

## 10. Gradient path

```text
sparse target-plane error
→ physical renderer
→ propagated Gaussian state
→ anchors and local field
→ cached Z_str / Z_app
→ teacher-free evidence encoder
```

Because the reconstruction gradient is long and attribution is ambiguous, the direct structural constraints are part of the core Phase-1 optimization, not optional decoration.

---

## 11. Theoretical interpretation

In a simplified local linear model:

\[
i_m=A_ms+B_ma_m+\epsilon_m,
\]

where \(s\) is shared structure and \(a_m\) is modality-specific appearance.

If modality-specific components are not stably correlated across modalities, then cross-modality covariance is dominated by the shared structural component:

\[
\operatorname{Cov}(i_m,i_n)
\approx
A_m\operatorname{Cov}(s)A_n^\top.
\]

Cross-modality agreement therefore favors a shared structural subspace. Spatial equivariance preserves geometric correspondence; intensity invariance suppresses scanner- and contrast-specific nuisance; anti-collapse terms prevent constant features; reconstruction loss selects the structural information that is actually useful for 3D prediction.

This argument supports recovery of a useful shared subspace up to representation transforms. It does not claim that individual feature channels equal a unique anatomical quantity.

---

## 12. Required ablations

Hold the downstream pipeline fixed and compare:

```text
E0 — analytic differential features only
E1 — raw-image shallow CNN
E2 — analytic scaffold + teacher-free micro-CNN, main method
E3 — frozen pretrained encoder, compute upper bound
E4 — dense teacher-distilled student, privileged-training upper bound only
```

For every encoder, hold constant:

- permanently sparse context/target manifests;
- tiny local MLP;
- anchor bootstrap;
- Gaussian memory and propagation;
- renderer;
- observation budget;
- optimization schedule after encoder warm-up.

Report reconstruction gain per encoder FLOP in addition to raw quality.

---

## 13. Required logs

Per patient episode record:

- sparse manifest ID and hash;
- context and target slice IDs;
- confirmation that no non-manifest slice was opened;
- modality distribution;
- encoder FLOPs, latency, memory, and cache size;
- structural loss terms;
- target-plane reconstruction losses;
- number of anchors and Gaussians;
- downstream state statistics;
- random seed and patient-level split.

---

## 14. Failure conditions

Phase 1 is invalid if:

- a teacher is required for the main result;
- complete-volume features or targets enter main training;
- a slice outside the fixed sparse patient manifest is opened by the training loader;
- target pixels influence state construction before rendering;
- cross-modality alignment forces modality-specific appearance features to become identical;
- structural features collapse to constants;
- output stride removes detail that the tiny downstream field cannot recover;
- reported compute excludes analytic preprocessing or hidden dense operations;
- patient splitting is performed at slice level.

---

## 15. Code-entry gate

Do not implement the full downstream system until the following Phase-1 checks pass on a small sparse-only prototype:

1. feature maps remain correctly aligned after augmentation;
2. structural features do not collapse;
3. registered modalities show higher structural agreement than mismatched locations;
4. local differential information can be decoded from the compact structural map;
5. E2 improves sparse target-plane prediction over E0 and E1 at acceptable FLOPs;
6. the loader audit proves that no complete-volume data are accessed.
