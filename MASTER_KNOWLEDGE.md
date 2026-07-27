# MASTER KNOWLEDGE

## 1. Kết luận cốt lõi

Hướng nghiên cứu không nên được mô tả đơn giản là “dùng 3D Gaussian Splatting cho MRI đa sequence”. Câu chuyện thống nhất hơn là:

> Một bệnh nhân có nhiều MRI sequence đã đăng ký cùng không gian. Thay vì đọc và fusion toàn bộ các volume, hệ thống chỉ truy vấn tuần tự một số ít sequence–slice. Mỗi quan sát cập nhật một patient-specific Gaussian memory. Gaussian không được tối ưu tự do hoàn toàn mà được sinh và ràng buộc bởi một structural SDF scaffold. Trạng thái hiện tại của SDF–Gaussian space quyết định trajectory quan sát tiếp theo. Quá trình dừng khi không còn quan sát nào làm giảm đáng kể uncertainty, representation gần như không đổi và vùng nguy cơ đã được phủ đủ.

Hai novelty liên kết thành một chuỗi logic:

\[
\boxed{
\text{SDF scaffold}
\rightarrow
\text{low-DoF Gaussian state}
\rightarrow
\text{tractable observability}
\rightarrow
\text{multi-wave trajectory}
}
\]

---

## 2. Novelty 1 — 3D-SLNR-adaptive SDF initialization and constraint for Gaussian priors

### 2.1. Insight lấy từ 3D-SLNR

3D-SLNR biểu diễn global SDF bằng nhiều local SDF band-limited đặt tại support points. Tất cả local SDF dùng chung một tiny MLP; từng local element chỉ khác position, rotation và scale. Expressiveness đến từ adaptive geometry thay vì latent feature lớn.

Điểm cần kế thừa:

- local support primitives;
- shared basis field;
- adaptive point allocation;
- efficient local detection;
- prune–clone–split;
- geometric state nhỏ hơn feature-heavy neural points.

Điểm không bê nguyên:

- 3D-SLNR giả định point cloud geometry đã có;
- output của nó là SDF/mesh, chưa có multi-sequence intensity;
- nó không tối ưu observation trajectory;
- position–rotation–scale vẫn là biến học độc lập.

### 2.2. Tuning đề xuất

Dùng SDF không chỉ để khởi tạo, mà để định nghĩa manifold mà Gaussian được phép tồn tại và cập nhật.

Structural field:

\[
F_\psi(\mathbf x):\mathbb R^3\rightarrow\mathbb R
\]

Level set cấu trúc:

\[
\mathcal M=\{\mathbf x\mid F_\psi(\mathbf x)=0\}
\]

Normal tại anchor \(\mathbf a_i\):

\[
\mathbf n_i=\frac{\nabla F_\psi(\mathbf a_i)}{\|\nabla F_\psi(\mathbf a_i)\|+\varepsilon}
\]

Tangent frame:

\[
\mathbf R_i=[\mathbf t_{1,i},\mathbf t_{2,i},\mathbf n_i]
\]

Rotation của Gaussian được **suy ra** từ SDF, không còn là quaternion độc lập.

Covariance low-DoF:

\[
\Sigma_i=
\sigma_{t,i}^2(\mathbf I-\mathbf n_i\mathbf n_i^\top)
+
\sigma_{n,i}^2\mathbf n_i\mathbf n_i^\top
\]

Như vậy rotation + anisotropic scale có thể giảm từ 7 giá trị lưu trữ xuống 2 scale thực sự học.

Position có thể được parameterize trong local manifold coordinates:

\[
\mu_i=\mathbf a_i+\delta u_i\mathbf t_{1,i}+\delta v_i\mathbf t_{2,i}+\delta n_i\mathbf n_i
\]

với \(\delta n_i\) bị regularize mạnh hoặc chiếu lại lên manifold.

### 2.3. Gaussian state khuyến nghị

\[
g_i=(\mathbf a_i,\delta u_i,\delta v_i,\delta n_i,\sigma_{t,i},\sigma_{n,i},\alpha_i,\mathbf c_i,\mathbf u_i)
\]

Trong đó:

- \(\mathbf a_i\): anchor SDF, có thể không học hoặc cập nhật chậm;
- \((\delta u_i,\delta v_i,\delta n_i)\): residual geometry;
- \((\sigma_t,\sigma_n)\): tangent/normal extent;
- \(\alpha_i\): occupancy/opacity;
- \(\mathbf c_i\): tissue hoặc sequence-conditioned code nhỏ;
- \(\mathbf u_i\): uncertainty/observability state.

Đối với bốn MRI sequence, có thể dùng:

\[
\mathbf c_i=[c_i^{T1},c_i^{T2},c_i^{FLAIR},c_i^{T1ce}]
\]

hoặc compact tissue code \(\mathbf z_i\) cùng sequence decoder:

\[
c_{i,m}=D(\mathbf z_i,\mathbf e_m)
\]

### 2.4. Slow–fast memory

SDF là **slow structural memory**; Gaussian là **fast appearance/evidence memory**.

\[
\eta_{SDF}\ll\eta_G
\]

- Gaussian cập nhật sau mỗi queried slice.
- SDF chỉ cập nhật khi có structural residual đủ mạnh và nhất quán.
- Appearance residual không được phép làm biến dạng geometry.

### 2.5. Structural và appearance separation

Với observation \(I_{m,z}\):

\[
F^{str}=E_{str}(I_{m,z})
\]

\[
F^{app}=E_{app}(I_{m,z},m)
\]

Structural branch cập nhật:

- SDF;
- Gaussian anchors/residual geometry;
- scale;
- birth/split/prune.

Appearance branch cập nhật:

- sequence evidence;
- tissue code;
- class logits;
- uncertainty theo sequence.

### 2.6. Topology adaptation

**Birth** khi residual cao nhưng không có primitive hỗ trợ.

**Split** khi footprint lớn, curvature cao hoặc residual nội vùng không đồng nhất.

**Prune** khi primitive không còn evidence, opacity thấp, xa SDF level set hoặc liên tục gây lỗi.

Topology change chỉ được chấp nhận nếu làm giảm một energy chung ít nhất một ngưỡng.

---

## 3. Novelty 2 — Balanced multi-wave trajectory optimization

### 3.1. Observation space

Action cơ bản:

\[
a=(m,z)
\]

hoặc bản chi tiết:

\[
a=(m,z,r)
\]

với \(r\) là region/support cluster.

Trajectory:

\[
\tau=(a_1,a_2,\ldots,a_T)
\]

Không gian quan sát đầy đủ có thể là \(4\times Z\), nhưng model chỉ mở một tập rất nhỏ.

### 3.2. Không gian mà wave di chuyển

Không phải chỉ là trục slice. Ta xây joint feature–geometry graph:

\[
\mathcal H=(\mathcal V,\mathcal E)
\]

Node:

\[
v_i=(m_i,z_i,r_i,\mathbf x_i,\mathbf h_i,\mathbf u_i)
\]

Cạnh gồm:

- spatial adjacency;
- cross-sequence correspondence;
- feature-geodesic relation;
- SDF-level-set neighborhood.

### 3.3. Edge cost

\[
c_t(i,j)=
\frac{
\lambda_g d_{geo}(i,j)
+\lambda_f d_{feat}(i,j)
+\lambda_m d_{mod}(i,j)
+\lambda_j d_{jump}(i,j)
}{
\varepsilon+Q_t(j)
}
+\lambda_rR_t(j)
\]

Trong đó \(Q_t(j)\) gồm:

- uncertainty reduction;
- uncovered mass;
- pathology risk;
- novelty;
- expected topology correction.

Redundancy tăng nếu vùng đã được wave khác phủ hoặc quan sát gần đây.

### 3.4. Multi-source propagation

Các support anchors \(\{a_k\}_{k=1}^K\) phát ra nhiều wavefront song song.

Arrival cost:

\[
T_t(v)=\min_k\min_{P:a_k\rightsquigarrow v}\sum_{(i,j)\in P}c_t(i,j)
\]

Ở continuous form:

\[
\|\nabla T_t(\mathbf x)\|V_t(\mathbf x)=1
\]

với propagation speed:

\[
V_t(\mathbf x)=
\frac{\varepsilon+\operatorname{Information}_t(\mathbf x)}
{\operatorname{Difficulty}_t(\mathbf x)}
\]

### 3.5. Balanced parallel frontiers

Mỗi wave đề xuất candidate tốt nhất của mình. Global scheduler chọn một batch bổ sung:

\[
\mathcal B_t^*=\arg\max_{\mathcal B}
\sum_{v\in\mathcal B}Q_t(v)
-\lambda_o\sum_{i\ne j}\operatorname{Overlap}(v_i,v_j)
-\lambda_b\operatorname{Imbalance}(\mathcal B)
\]

Các wave dùng shared coverage map nên không chạy trùng nhau.

Có thể thêm capacity penalty cho wave đã sử dụng quá nhiều budget:

\[
c_t^{(k)}(i,j)=c_t(i,j)+\lambda_b\frac{n_k}{B_k}
\]

### 3.6. Dynamic replanning

Sau mỗi batch observation:

- Gaussian state thay đổi;
- uncertainty thay đổi;
- graph cost thay đổi;
- topology có thể thay đổi.

Không cần chạy lại toàn bộ search từ đầu. Dùng local graph repair theo tinh thần incremental shortest path/D*-style replanning cho vùng bị ảnh hưởng.

### 3.7. Vai trò của multi-wave

Multi-wave không chỉ để chạy nhanh. Nó giúp:

- tránh một trajectory đơn bị mắc kẹt trong local region;
- phủ nhiều anatomical modes;
- tăng diversity;
- cho phép batch querying/parallel encoding;
- tạo geodesic partition theo uncertainty;
- cân bằng exploration và refinement.

---

## 4. Pipeline end-to-end

### Stage 0 — Preprocessing

- Register các sequence về cùng physical coordinate system.
- Normalize intensity theo sequence.
- Resample/crop thống nhất.
- Lưu metadata plane, spacing và sequence ID.
- Full volume nằm trên CPU/disk; GPU chỉ nhận queried slices.

### Stage 1 — Anchor scout

Chọn 3–5 anchor observations có coverage lớn, hoặc học anchor policy từ training cohort.

Output:

\[
\mathcal O_0=\{(m_k,z_k)\}_{k=1}^{K_0}
\]

### Stage 2 — Structural SDF scaffold

Các anchor đi qua structural encoder để dựng coarse SDF hoặc SDF bundle.

Output:

\[
F_0(\mathbf x),\quad U_F(\mathbf x)
\]

### Stage 3 — SDF-guided Gaussian prior

- Sample anchors trên level sets/regions.
- Derive normals/tangent frames từ SDF gradient.
- Initialize covariance từ tangent/normal scales.
- Initialize appearance/tissue code từ anchor features.
- Initialize uncertainty cao ở vùng suy diễn từ prior.

Output:

\[
\mathcal G_0
\]

### Stage 4 — Build observation graph

Tạo candidate nodes \((m,z,r)\), descriptors và edge costs từ state hiện tại.

### Stage 5 — Multi-wave propagation

Các wave xuất phát từ support anchors, lan truyền theo information-weighted geodesic cost.

### Stage 6 — Global arbitration

Chọn candidate/batch có gain cao, overlap thấp và phân bổ cân bằng.

### Stage 7 — Query and encode

Chỉ lúc này mới load ảnh thật:

\[
I_t=I_{m_t,z_t}
\]

Encoder 2D shared + sequence embedding tạo structural và appearance features.

### Stage 8 — Render-before-update

Render prediction của observation từ current Gaussian state:

\[
\hat I_t=R(\mathcal G_t,m_t,z_t)
\]

Residual:

\[
E_t=I_t-\hat I_t
\]

Tách structural residual và appearance residual.

### Stage 9 — Local assimilation

Chỉ update primitives giao với slice/region:

- fast Gaussian evidence update;
- slow SDF correction;
- uncertainty update;
- birth/split/prune;
- optional segmentation logits update.

### Stage 10 — Local graph repair

Update coverage, candidate utility và edge costs chỉ ở neighborhood bị ảnh hưởng.

### Stage 11 — Convergence test

Dừng khi representation đạt fixed point của observability, hoặc dừng bất thường khi hết safety budget.

### Stage 12 — Output

Primary output khuyến nghị:

- 3D segmentation;
- uncertainty map;
- query trajectory;
- final SDF–Gaussian representation.

Auxiliary output:

- held-out slice reconstruction;
- optional full multi-sequence volume rendering.

---

## 5. Hội tụ và điều kiện dừng

### 5.1. Energy chung

\[
\mathcal E_t=
\lambda_U\mathcal U_t
+\lambda_C\mathcal C_t
+\lambda_R\mathcal R_t
+\lambda_D\mathcal D_t
+\lambda_S\mathcal S_t
\]

- \(\mathcal U_t\): uncertainty mass;
- \(\mathcal C_t\): uncovered risk;
- \(\mathcal R_t\): wave overlap/redundancy;
- \(\mathcal D_t\): disagreement với observations thật;
- \(\mathcal S_t\): representation complexity.

Mục tiêu implementation là làm:

\[
\mathcal E_{t+1}\leq\mathcal E_t
\]

sau mỗi complete iteration.

### 5.2. Stop rule

Dừng khi đồng thời thỏa:

\[
\max_a\Delta I(a)<\varepsilon_I
\]

\[
\frac{\|\theta_{t+1}-\theta_t\|}{\|\theta_t\|+\varepsilon}<\varepsilon_\theta
\]

\[
\mathcal L_{obs}<\tau_D
\]

\[
U_{worst}<\tau_U
\]

\[
N_G^{t+1}=N_G^t
\]

trong \(P\) vòng liên tiếp.

Budget \(B_{max}\) chỉ là fallback:

\[
\operatorname{Stop}=\operatorname{Converged}\lor t=B_{max}
\]

Nếu hết budget nhưng chưa hội tụ, trả trạng thái `insufficiently observed`.

### 5.3. Claim hội tụ hợp lý

Không claim global optimum.

Claim nên là:

> The alternating multi-wave and SDF–Gaussian update process converges to an observation-stable fixed point under monotone energy descent and stabilized topology.

---

## 6. Task khuyến nghị

### Primary task

**Budgeted active multi-sequence 3D tumor segmentation from sparse slice queries.**

Lý do:

- SDF gắn trực tiếp với boundaries;
- không cần tái tạo mọi intensity detail;
- trajectory có objective lâm sàng rõ;
- giảm memory và scope;
- dễ đánh giá pathology preservation.

### Auxiliary task

Held-out slice reconstruction để kiểm chứng representation không chỉ ghi nhớ mask.

### Alternative full task

Sparse multi-sequence 3D reconstruction, nhưng scope và clinical validation khó hơn.

---

## 7. Training strategy

### Phase A — Representation pretraining

- random/uniform sparse observations;
- train structural encoder, SDF scaffold, Gaussian renderer/updater;
- observed and held-out slice consistency.

### Phase B — Utility learning

Offline thử candidate queries và đo actual improvement:

\[
\Delta Q(a)=Q(\mathcal G_{t+1}^a)-Q(\mathcal G_t)
\]

Train utility predictor/ranker.

### Phase C — Multi-wave policy training

- initialize anchors;
- run wave propagation;
- train global scheduler bằng oracle ranking, contrastive/ranking loss hoặc differentiable surrogate;
- chưa cần RL ở paper đầu tiên.

### Phase D — Joint fine-tuning

- unroll ngắn 3–6 steps;
- truncated BPTT;
- detach memory định kỳ;
- topology frozen ở giai đoạn cuối.

---

## 8. Losses

\[
\mathcal L=
\lambda_{seg}\mathcal L_{seg}
+\lambda_{rec}\mathcal L_{heldout}
+\lambda_{sdf}\mathcal L_{SDF}
+\lambda_{man}\mathcal L_{manifold}
+\lambda_{cal}\mathcal L_{uncertainty}
+\lambda_{traj}\mathcal L_{trajectory}
+\lambda_{cmp}\mathcal L_{complexity}
\]

Segmentation:

\[
\mathcal L_{seg}=\mathcal L_{Dice}+\mathcal L_{CE}+\lambda_b\mathcal L_{boundary}
\]

Manifold:

\[
\mathcal L_{manifold}=\sum_iF_\psi(\mu_i)^2+\lambda_n\delta n_i^2
\]

Uncertainty calibration:

\[
\mathcal L_{cal}=|U(a)-D(\hat y_a,y_a)|
\]

Trajectory ranking:

\[
\mathcal L_{rank}=\max(0,\delta-\hat\Delta Q(a^+)+\hat\Delta Q(a^-))
\]

---

## 9. Evaluation

### Segmentation

- Dice: WT, TC, ET;
- HD95;
- lesion-wise Dice/HD95;
- lesion recall and false-negative lesion count;
- surface Dice;
- ASSD.

### Trajectory

- quality–budget curve;
- area under quality–budget curve;
- \(B_{90}\), \(B_{95}\): queries cần để đạt 90%/95% full-input quality;
- actual vs predicted gain correlation;
- redundancy rate;
- sequence allocation distribution;
- convergence steps.

### Representation efficiency

- peak VRAM;
- number of Gaussians;
- bytes per primitive;
- runtime per query/update;
- total queried slices;
- energy/latency if available.

### Auxiliary reconstruction

- PSNR;
- SSIM;
- NMSE;
- edge/gradient error;
- downstream segmentation on reconstructed volumes.

---

## 10. Critical ablations

1. Vanilla/free Gaussian vs SDF-guided init.
2. SDF init only vs SDF constraint throughout optimization.
3. Free quaternion vs SDF-derived rotation.
4. Full anisotropic scale vs tangent/normal low-DoF covariance.
5. Single wave vs multi-wave.
6. Multi-wave without global balance.
7. No redundancy penalty.
8. Full recomputation vs incremental graph repair.
9. Fixed-step stopping vs convergence stopping.
10. No topology adaptation.
11. Shared state without sequence-conditioned evidence.
12. Full-volume fusion baseline vs asynchronous assimilation.
13. Segmentation-only vs segmentation + held-out reconstruction.

---

## 11. Key claims to preserve

### Strong claim 1

SDF does not merely initialize Gaussians; it defines the low-dimensional manifold on which Gaussian geometry is optimized.

### Strong claim 2

Trajectory determines the observability of the Gaussian space.

### Strong claim 3

Multiple information waves enable complementary, balanced and parallel exploration from sparse anchors.

### Strong claim 4

Stopping is based on observability convergence, representation stability and worst-case safety, not an arbitrary number of updates.

### Avoid

- “globally optimal trajectory”;
- “fusion-free” tuyệt đối;
- “400 time points” nếu thực tế là 400 sequence–slice images;
- “3DGS” nếu implementation không còn dùng differentiable Gaussian splatting/rendering;
- reconstruction fidelity chỉ dựa trên PSNR/SSIM;
- fixed 5–6 slices per sequence như một giả định cứng.
