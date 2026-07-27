# Architecture — SDF-Manifold Active Gaussian Memory

## 1. Research objective

The system builds a patient-specific 3D representation from a small, adaptively selected subset of registered multi-sequence MRI slices. It avoids explicit fusion of complete T1/T2/FLAIR/T1ce volumes. Instead, each queried observation is assimilated asynchronously into a persistent SDF-constrained Gaussian memory.

The architecture is organized around two linked contributions:

1. **SDF/3D-SLNR-adaptive low-DoF Gaussian prior** — an anatomical SDF scaffold derives Gaussian orientation and constrains position/covariance, reducing the free optimization space.
2. **Balanced multi-wave observability routing** — multiple information fronts propagate from support anchors over an evolving feature–geometry graph and select complementary sequence–slice observations until an observability fixed point is reached.

The dependency between the two is deliberate:

\[
\text{SDF scaffold}
\rightarrow
\text{low-DoF Gaussian state}
\rightarrow
\text{tractable observability}
\rightarrow
\text{multi-wave routing}.
\]

---

## 2. Top-level module graph

```text
Registered multi-sequence slice pool (CPU/disk)
                       │
                       ▼
                 Anchor Scout
                       │
                       ▼
        Sequence-invariant Structural Encoder
                       │
                       ▼
             Local SDF Scaffold / Bundle
                       │
                       ▼
       Low-DoF SDF-Constrained Gaussian Memory
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
   Observation Graph       Slice Renderer / Heads
             │                   │
             ▼                   │
  Balanced Multi-Wave Router    │
             │                   │
             ▼                   │
      Global Batch Scheduler     │
             │                   │
             ▼                   │
     Query selected slices       │
             │                   │
             ▼                   │
 Structural + Appearance Encoders
             │
             ▼
       Render-Before-Update
             │
             ▼
   Local Evidence Assimilation ──┘
             │
             ├── fast Gaussian update
             ├── slow SDF correction
             ├── uncertainty update
             ├── birth / split / prune
             └── local graph repair
```

---

## 3. Data model

For one patient, the hidden full dataset is

\[
\mathcal V=\{V^m\mid m\in\mathcal M\},
\qquad
\mathcal M=\{T1,T2,FLAIR,T1ce\}.
\]

A query action is

\[
a_t=(m_t,z_t)
\]

or, for region-level routing,

\[
a_t=(m_t,z_t,r_t),
\]

where \(r_t\) denotes a support cluster or spatial region on the slice. Candidate images are not loaded before selection. Only the chosen observation is transferred to the GPU.

Each slice carries physical metadata:

```text
SliceObservation
├── image: [1, H, W]
├── modality_id
├── slice_index
├── origin_xyz
├── spacing_xyz
├── plane_normal
└── affine / registration transform
```

---

## 4. Block A — Anchor Scout

### Purpose

Produce a small set of initial observations that provides coarse anatomical coverage and enough evidence to initialize the SDF scaffold.

### Inputs

- sequence and slice metadata;
- maximum anchor budget \(K_0\);
- optional low-cost scout features.

### Outputs

\[
\mathcal A_0=\{a_1,\ldots,a_{K_0}\}.
\]

### MVP

Use 3–5 fixed, spatially separated anchors with cross-sequence diversity.

### Advanced version

Learn an anchor proposal function that maximizes population-level expected coverage while penalizing redundancy:

\[
\max_{\mathcal A_0}
C(\mathcal A_0)-\lambda R(\mathcal A_0).
\]

---

## 5. Block B — Structural SDF Scaffold

### Purpose

Estimate a slow, sequence-invariant anatomical field used to initialize and constrain Gaussian geometry.

### Components

```text
queried slice
    │
    ▼
structural encoder E_str
    │
    ▼
plane-conditioned local field updates
    │
    ▼
SDF decoder F_psi(x)
```

A single outer-brain SDF is insufficient for volumetric MRI. The implementation should support one of:

1. a multi-channel SDF bundle;
2. boundary SDF primitives plus interior volumetric Gaussians;
3. a multi-level structural field.

The recommended MVP is **boundary SDF + interior Gaussians**.

### Field outputs

For a point \(\mathbf x\):

- signed field value \(F_\psi(\mathbf x)\);
- normal \(\mathbf n(\mathbf x)\);
- optional curvature;
- structural uncertainty.

The normal is

\[
\mathbf n_i=
\frac{\nabla F_\psi(\mathbf a_i)}
{\|\nabla F_\psi(\mathbf a_i)\|+\epsilon}.
\]

---

## 6. Block C — Low-DoF Gaussian Memory

### 6.1 State representation

For Gaussian \(i\):

\[
g_i=(
\mathbf a_i,
\delta u_i,\delta v_i,\delta n_i,
\sigma_{t,i},\sigma_{n,i},
\alpha_i,
\mathbf z_i,
\mathbf u_i
).
\]

- \(\mathbf a_i\): SDF anchor;
- \(\delta u_i,\delta v_i\): tangent-plane offsets;
- \(\delta n_i\): optional normal offset, strongly regularized;
- \(\sigma_t,\sigma_n\): tangent and normal scales;
- \(\alpha_i\): occupancy/opacity;
- \(\mathbf z_i\): compact tissue/sequence state;
- \(\mathbf u_i\): geometry and modality observability.

### 6.2 Derived orientation

Build the local frame

\[
\mathbf R_i=[\mathbf t_{1,i},\mathbf t_{2,i},\mathbf n_i].
\]

Rotation is derived from the SDF and is not stored as a free quaternion.

### 6.3 Derived covariance

For tangent-plane isotropy:

\[
\Sigma_i=
\sigma_{t,i}^2(\mathbf I-\mathbf n_i\mathbf n_i^\top)
+
\sigma_{n,i}^2\mathbf n_i\mathbf n_i^\top.
\]

This replaces free quaternion + three independent scales with two learned scale variables. A curvature-aligned tangent angle may be added later if tangent-plane isotropy is too restrictive.

### 6.4 Center parameterization

\[
\mu_i=
\mathbf a_i+
\delta u_i\mathbf t_{1,i}+
\delta v_i\mathbf t_{2,i}+
\delta n_i\mathbf n_i.
\]

Optional projection back to the manifold:

\[
\Pi_F(\mathbf x)=
\mathbf x-
\frac{F_\psi(\mathbf x)}
{\|\nabla F_\psi(\mathbf x)\|^2+\epsilon}
\nabla F_\psi(\mathbf x).
\]

### 6.5 Slow–fast memory

- Gaussian evidence and appearance update after every queried slice.
- SDF geometry updates only from persistent structural residuals.

\[
\eta_{SDF}\ll\eta_G.
\]

This prevents modality-specific intensity changes from deforming anatomy.

---

## 7. Block D — Slice Renderer and Task Heads

MRI slices are cross-sections, not perspective camera images. The renderer therefore computes the intersection/contribution of 3D Gaussians with a physical slice plane.

For plane \(P_z\), a Gaussian contribution along the plane normal can be gated by

\[
w_{i,z}=\exp\left(
-\frac{d(\mu_i,P_z)^2}{2\sigma_{n,i}^2}
\right).
\]

The in-plane footprint is rendered as a 2D Gaussian. Sequence-conditioned intensity can be decoded by

\[
c_{i,m}=D_{app}(\mathbf z_i,\mathbf e_m).
\]

Heads:

- optional slice/volume reconstruction head;
- 3D segmentation head;
- uncertainty/calibration head;
- observability descriptor head.

For the first ISBI implementation, segmentation can be primary and held-out slice reconstruction auxiliary.

---

## 8. Block E — Structural and Appearance Encoders

Each queried slice is processed by two pathways:

```text
I_(m,z)
├── E_str → anatomy/boundary evidence
└── E_app(·, modality embedding) → sequence-specific evidence
```

### Structural branch updates

- SDF;
- anchor residual geometry;
- tangent/normal scales;
- topology operations.

### Appearance branch updates

- tissue code;
- sequence-conditioned intensity/class logits;
- modality uncertainty.

The branches may share an early stem but must have separate update gates.

---

## 9. Block F — Render-Before-Update Assimilation

Before assimilating an observation, render the current prediction:

\[
\hat I_t=R(\mathcal G_t,m_t,z_t).
\]

Residual:

\[
E_t=I_t-\hat I_t.
\]

Only evidence not already explained by the memory should drive large updates.

Local affected primitive set:

\[
\mathcal N_t=
\{i\mid d(\mu_i,P_{z_t})<k\sigma_{n,i}\}.
\]

A gated local update is applied:

\[
\mathbf h_i^{t+1}=
(1-\gamma_{i,t})\mathbf h_i^t+
\gamma_{i,t}\tilde{\mathbf h}_{i,t}.
\]

The gate depends on residual magnitude, structural consistency, modality identity, current uncertainty, and observation quality.

---

## 10. Block G — Adaptive Topology

### Birth

Create a primitive when a high residual region has insufficient Gaussian/SDF support.

### Split

Split when a primitive is too large for local curvature or has heterogeneous residuals within its footprint.

### Prune

Remove primitives with low support, low opacity, persistent inconsistency, or excessive distance from the structural field.

A topology operation is accepted only if it decreases the common energy by a minimum margin:

\[
\mathcal E_{after}<\mathcal E_{before}-\delta_{topo}.
\]

Training phases:

1. burn-in: topology changes allowed frequently;
2. stabilization: decreasing operation frequency;
3. freeze: fixed topology near convergence.

---

## 11. Block H — Joint Feature–Geometry Observation Graph

A node represents a candidate sequence–slice region:

\[
v_i=(m_i,z_i,r_i,\mathbf x_i,\mathbf h_i,\mathbf u_i).
\]

Edge types:

- within-sequence spatial adjacency;
- cross-sequence aligned correspondence;
- SDF-geodesic neighborhood;
- learned feature similarity.

Dynamic edge cost:

\[
c_t(i,j)=
\frac{
\lambda_gd_{geo}+
\lambda_fd_{feat}+
\lambda_md_{mod}+
\lambda_jd_{jump}
}{\epsilon+Q_t(j)}
+
\lambda_rR_t(j).
\]

\(Q_t\) combines expected uncertainty reduction, uncovered mass, novelty, and pathology risk. \(R_t\) penalizes redundancy and overlap.

---

## 12. Block I — Balanced Multi-Wave Router

### Sources

Support clusters emit parallel fronts. Each front maintains a local frontier/priority structure and shared access to coverage and uncertainty maps.

### Propagation

For a fixed graph, each front minimizes accumulated information-weighted geodesic cost. Multi-source shortest paths are exact for the current nonnegative edge costs.

### Balancing

Balance is based on remaining uncertainty mass, not raw node count:

\[
M_k=\sum_{v\in\mathcal C_k}u(v).
\]

A load penalty discourages one wave from consuming most of the query budget.

### Scheduler

Each wave proposes top candidates. The global scheduler selects a complementary batch:

\[
\max_{\mathcal B_t}
\sum_{v\in\mathcal B_t}Q_t(v)
-
\lambda_o\sum_{i\neq j}Overlap(v_i,v_j)
-
\lambda_bBalancePenalty.
\]

### Dynamic repair

After assimilation, only graph nodes and edge costs affected by changed Gaussian/SDF neighborhoods are recomputed. Arrival fields are repaired incrementally rather than rebuilt globally.

---

## 13. Block J — Convergence Controller

The main stopping condition is an observability fixed point, not a fixed number of updates.

Stop when all conditions hold for \(P\) consecutive rounds:

\[
\max_a\Delta I(a)<\epsilon_I,
\]

\[
\frac{\|\theta_{t+1}-\theta_t\|}
{\|\theta_t\|+\epsilon}<\epsilon_\theta,
\]

\[
\mathcal L_{observed}<\tau_D,
\]

\[
U_{worst}<\tau_U,
\]

and topology is stable.

A maximum query budget remains a safety fallback. Reaching the budget without convergence returns an `insufficiently_observed` status and an uncertainty map.

---

## 14. Recommended software interfaces

```python
class StructuralSDF(nn.Module):
    def query(self, xyz): ...
    def normal(self, xyz): ...
    def update_from_slice(self, observation, structural_features): ...

class GaussianMemory(nn.Module):
    def initialize_from_sdf(self, sdf, anchors): ...
    def render_slice(self, modality, plane): ...
    def assimilate(self, observation, structural, appearance): ...
    def adapt_topology(self, diagnostics): ...

class ObservationGraph:
    def build(self, memory, candidate_metadata): ...
    def repair(self, changed_regions): ...

class MultiWaveRouter:
    def initialize(self, graph, support_clusters): ...
    def propose(self, memory, graph): ...

class GlobalScheduler:
    def select_batch(self, proposals, budget): ...

class ConvergenceController:
    def update(self, diagnostics): ...
    def should_stop(self) -> bool: ...
```

---

## 15. MVP versus full architecture

### MVP

- registered axial volumes;
- fixed anchor scout;
- one structural SDF plus interior Gaussians;
- 2D encoders;
- diagonal uncertainty;
- region-level graph;
- greedy balanced multi-wave scheduler;
- segmentation primary, reconstruction auxiliary;
- no beam search or RL.

### Full version

- learned anchors;
- SDF bundle;
- curvature-aligned covariance;
- low-rank posterior precision;
- incremental shortest-path repair;
- short-horizon beam search;
- adaptive patient-specific stopping;
- multi-dataset validation.
