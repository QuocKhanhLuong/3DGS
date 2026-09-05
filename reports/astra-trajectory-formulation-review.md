# Astra trajectory formulation and systems review

Date: 2026-09-05. Repository: `QuocKhanhLuong/3DGS`, branch `main`.

Reviewed source: `ad560413eb3fbcc02fa633891642a7a3a08e7933` (also `origin/main` after fetch). This is an analysis-only review of Gates C/E and their F/G execution boundaries. Proposed changes below are **not implemented or authorized by this report**. MedicalNet remains frozen. No training, checkpoint modification, or production-code edit was performed.

## 1. Executive verdict

**The patch repairs two concrete defects but does not establish coherent adaptive route learning. Do not start a longer training run.** Bounded travel removes an unbounded ranking term, and terminal counterfactuals restore direct RewardNet supervision at K=0. Neither establishes calibrated gain, action-consistent stopping, useful UpdateNet learning, or validation/deployment parity.

The strongest findings are:

1. **Ranking and continuation can refer to different candidates.** The solver selects `argmax(reward - costs)` but continues on `max(reward) > 0.025`. An unselected candidate can keep the route alive while the selected candidate has insufficient gain. Compact writes need not change the unselected candidate's descriptor. This permits persistent continuation to Kmax.
2. **The latest aggregates support a K=0/K=64 split, not healthy adaptation.** Under the reviewed aggregation and Kmax, conditional mean K among nonzero routes is exactly 64 within floating-point error. For 121 validation subjects, this is 119 stopped immediately and two executed 64 updates. Those active subjects' mean selected reward is only **0.0199015**, below the halt threshold. Their candidate utility is positive only in the initial scoring round, to the precision of the exported metrics.
3. **The supervised score is not expected net reconstruction gain.** Its target clips all harmful outcomes to zero, divides local gain by local pre-error, and subtracts a sampled collateral penalty. A perfectly fitted score can still favor an action with negative expected signed gain. Low regression loss cannot validate stopping or argmax quality.
4. **The training floor explains train K0=0 mechanically.** It forces continuation, not diverse actions, and has no gradient through the halt comparison. Terminal reward loss trains RewardNet only; it does not bootstrap UpdateNet at a stopped state.
5. **The server validation policy differs from public Gate-G inference.** The server wrapper supplies no availability policy. Gate-G supplies exact no-revisit but constructs a new legacy `TrajectoryConfig`, losing the corrected flags. A W&B validation result is therefore not evidence for the deployed Gate-G route.
6. **Gate-E consumes 84.70% of the last epoch's training time.** Its nested state/candidate loop includes repeated full-volume validation and mask conversion, full-plane cloning, many small GPU calls and synchronizations, and repeated decoder queries. The code does not have purely local computational complexity just because the decoded query count is local.

Recommendation: a versioned, short **fixed-budget UpdateNet bootstrap followed by separate signed-gain fitting and calibrated, action-consistent stopping**, capped at four updates for the next diagnostic. Start with zero locality/overlap weights in this diagnostic, retain physical diagnostics and exact no-revisit, freeze the reconstructor while fitting RewardNet, and validate decisions against post-inference counterfactual measurements. Increasing K is not itself a success criterion; correctly rejecting useless updates is a valid outcome.

## 2. Current system reconstruction

### Actual call graph

```text
BraTS21PointGuidedDataset / collate
  observations T1,T2,FLAIR; observation-derived mask; observation affine
  -> PointGuidedTrainer._forward_objective
     -> _TrainingContextModule.forward
        -> model.forward_training_context
           -> _forward_frontend_with_gate_b_context
              one shared frozen MedicalNet traversal
              semantic head; deterministic initial points; bounded refinement
              B projection -> fixed Haar A -> fixed point spectral context
           -> trajectory._forward_with_training_trace -> _run
              B -> Z0
              repeat: query Z -> RewardNet -> costs -> solver
                      -> UpdateNet -> compact plane write -> Znext
           -> shared Gate-D decoder(final Z) -> prediction
     -> fetch target T1ce and segmentation from batch
     -> compute_training_objective(context, T1ce, observation mask)
        final reconstruction loss
        per executed pre-state: counterfactual reward + local/monotonic/delta
        final reached state: terminal counterfactual reward
     -> semantic auxiliary loss, after context creation
     -> backward / optimizer (training only)
     -> detached statistics -> subject-weighted epoch reduction -> W&B
```

The model's public `forward_baseline_inference` instead enters eval/no-grad and calls `run_baseline_inference`, which passes `ExactNoRevisitPolicy()` and a newly constructed Gate-G route config. Generic `forward` is not the training endpoint.

### Implemented equations

Let (p_i) be a fixed refined RAS-mm point within this call, (Z_t) the three dynamic 32-channel planes, and (d_i^t) its RewardNet descriptor:

\[
d_i^t=[z_{xy}(p_i),z_{xz}(p_i),z_{yz}(p_i),\pi_i,\bar q_i,\alpha_i]\in\mathbb R^{126},
\quad r_i^t=\sigma(W_2\operatorname{SiLU}(W_1d_i^t+b_1)+b_2).
\]

The descriptor contains 96 dynamic channels, three semantics, a reliability-weighted 24-channel descriptor, and three reliability values. It has no route index, budget, history, or direct error observation. UpdateNet sees `[z(96), f_spec(168), semantics(3), reliability(3)]`, hence 270 channels. It emits 96 corrections through `270 -> 128 -> 96`, bounded channelwise by `0.1*tanh` in this profile. The decoder is `96 -> 64 -> 32 -> 1`, with SiLU hidden layers and final-Z-only input.

With (R=4\) mm, current corrected costs and decisions are:

\[
T_i^0=0,\qquad T_i^t=\frac{\|p_i-p_{a_{t-1}}\|}{R+\|p_i-p_{a_{t-1}}\|}\quad(t>0),
\]
\[
O_i^t=\max_{j<t}\left[1-\frac{\|p_i-p_{a_j}\|}{2R}\right]_+^2,
\quad U_i^t=r_i^t-0.05T_i^t-0.20O_i^t,
\]
\[
a_t=\arg\max_{i\in\mathcal A_t}U_i^t,
\qquad \text{continue}=\big[\max_{i\in\mathcal A_t}r_i^t>0.025\big],\quad t<64.
\]

The overlap expression is a compact redundancy surrogate, not the intersection volume of two spheres. Travel is computed after reward scoring, before selection. The overlap map is initialized to zero and updated after each executed write by a pointwise maximum. Neither uses targets.

During training, selected weights have hard forward values and a straight-through backward surrogate:

\[
w=\operatorname{onehot}(a)+\operatorname{softmax}(U/1.0)
-\operatorname{stopgrad}(\operatorname{softmax}(U/1.0)).
\]

Only active rows receive these weights. The comparison that creates `active` is boolean and has no useful derivative. In the corrected profile, the first training halt-score tensor is replaced by `threshold + 1`; ranking itself remains greedy. This is **forced continuation**, not epsilon-greedy exploration.

### Source map

Line numbers refer to the reviewed commit; links open the relevant files.

| Evidence | Location |
|---|---|
| Shared frontend/context ordering | [model.py](../src/smagm/features/point_guided/model.py), lines 120–198, 353–408 |
| Reward and update descriptors | [reward.py](../src/smagm/features/point_guided/reward.py), 149–192; [updater.py](../src/smagm/features/point_guided/updater.py) |
| Route loop, telemetry before masking, training floor | [trajectory.py](../src/smagm/features/point_guided/trajectory.py), 296–399 |
| Selection, differentiable update/write, overlap accumulation | trajectory.py, 405–478 |
| Bounded costs and legacy/config semantics | [trajectory_cost.py](../src/smagm/features/point_guided/trajectory_cost.py), 23–69, 79–154 |
| Separate maxima and straight-through weights | [trajectory_solver.py](../src/smagm/features/point_guided/trajectory_solver.py), 77–94 |
| Counterfactual mixture, target, detach boundaries | [reward_supervision.py](../src/smagm/features/point_guided/reward_supervision.py), 316–381, 547–573, 665–749 |
| Actual-step and terminal objectives | [training_objective.py](../src/smagm/features/point_guided/training_objective.py), 345–499 |
| Server wrapper and statistics | [training/point_guided.py](../src/smagm/training/point_guided.py), 299–323, 410–619, 687–852 |
| Epoch timers and W&B serialization | training/point_guided.py, 1514–1656 |
| Gate-G overrides | [baseline_inference.py](../src/smagm/features/point_guided/baseline_inference.py), 43–77, 189–239 |
| Observation/target preprocessing separation | [brats21_point_guided.py](../src/smagm/data/brats21_point_guided.py), 1300–1420 |

## 3. Evidence from latest run

### Provenance boundary

The supplied files are in `03-09-2026-reports/`, not at repository root. They describe run `trajectory-logic-fix-smoke-e3-gpu1-20260903-152425`, started `2026-09-03T08:45:45.532293Z`, Python 3.10.20, RTX A4000, FP32 (`--no-amp`), three epochs, batch size one. The exported metadata lists two installed GPUs; that does not establish two-GPU training. `epoch=3`, `global_step=2895`; the epoch timing/step timing ratio is exactly 965 training examples.

**The reported run commit is `d02e50b57d5d82165641f1f39a16b83a9d6e431b`, not the audited HEAD.** It is absent locally. Fetching `origin main` succeeded without changing the reviewed HEAD; fetching that exact object was rejected with `upload-pack: not our ref`. The W&B config includes training/supervision but omits trajectory/model/data configuration, matching the current `wandb.init` implementation. The runtime source tree and effective corrected flags are therefore not cryptographically established by these exports. The arithmetic is strongly consistent with the reviewed corrected profile, but this is not proof of source parity.

Missing: resolved server `config.json`, `split.json`, source diff/clean-tree evidence, per-subject routes, `metrics.csv`/`train.jsonl` history, local `summary.json`, checkpoint identity, and a W&B run URL. The writer identifier in metadata is not a verified W&B run ID. I did not access a live W&B dashboard or server. A/B/C figures below are supplied historical evidence, not independently reloaded run histories.

Evidence fingerprints (SHA-256; the original untracked files were left untouched):

| File | SHA-256 |
|---|---|
| config.yaml | `bb4854bbf7ebc7fd186c57f8c1cd0bcf351d1495ab7602613a53d61d5051d37c` |
| wandb-metadata.json | `6fe7df81220db74d0e500fa91c889bf7ad5fbae14f0255d4eb2c754e08e38c1c` |
| wandb-summary.json | `09cbe609c787b18f7c519c13cc897efee9115237a661dad4e51854e95fb65983` |

### What the summary means

| Exported last-epoch statistic | Value | Code-supported interpretation |
|---|---:|---|
| Validation K0 fraction | 0.9834710743801653 | Subject fraction, including subjects with no selected step |
| Validation mean K | 1.0578512396694215 | Subject mean number of executed updates |
| Positive-utility fraction | 0.9837293388429752 | At B=1, mean of per-subject fractions over **all dense candidate scoring rounds**, before availability masking; not fraction of subjects permitted to continue |
| Candidate reward mean / max | 0.0162059684 / 0.0233039322 | Subject mean of within-route candidate mean / subject maximum; “max” is not cohort maximum |
| Utility after costs mean / max | +0.0153830533 / +0.0231913604 | Same aggregation, before availability masking; threshold is not subtracted |
| Travel / overlap means | 0.0154311317 / 0.0002567930 | Unweighted cost components; most subjects contribute first-round zeros |
| R-star mean / positive fraction | 0.0319713732 / 0.5095994916 | Sampled, valid counterfactual targets, including terminal state, then subject averaging |
| Train K0 / reward loss | 0 / 0.0006955971 | K0 is mechanically prevented; loss is a small-scale supervised regression objective |
| PSNR / SSIM / MAE | 14.9437 / 0.329820 / 0.144253 | Final prediction quality averaged over subjects; does not isolate update benefit |
| Train gradient norm | 1.07863 | Global optimizer-set norm returned **before** clipping at 1.0; not RewardNet/UpdateNet-specific |
| Train Gate-E / train total | 34,116.999 / 40,278.321 seconds | Last epoch, not the entire three-epoch run |

The actual three-epoch W&B elapsed runtime is approximately 129,644 seconds (36.0 hours). The epoch-three train time alone is 11.19 hours; Gate-E is 9.48 hours. These timings cannot establish full-run per-stage totals without history.

### Recovering the route distribution

For subject-weighted statistics, write (f_0=P(K=0)). Then:

\[
f_0=119/121,\qquad \bar K=128/121,
\qquad E[K\mid K>0]=\frac{\bar K}{1-f_0}=64.00000000000021.
\]

Because (0\le K\le64), equality at the upper bound means **all nonzero routes have K=64**, up to export precision. If the validation cohort has 121 subjects, the exact counts are 119 and two. These rational values also permit multiples of 121; the missing split prevents independently asserting the cohort size. The distribution conclusion does not need the exact cohort count, provided the reviewed aggregation and cap apply.

There is a further consistency check. With sigmoid rewards and zero initial costs, every initial candidate has positive utility at ordinary finite precision. For a K=64 subject, at least (1/64) of its scoring-round candidate evaluations must therefore be positive. The exported fraction is exactly:

\[
\frac{119}{121}\cdot1+\frac{2}{121}\cdot\frac1{64}
=0.9837293388429752.
\]

Thus the aggregates are consistent with—and under these assumptions force, within rounding—**all positive utilities occurring in the initial round, none in rounds 2–64**. A large positive-utility fraction is mostly the population that immediately halts. It is not evidence for continued useful updates.

Removing K0 zero sentinels from selected-step statistics yields:

| Quantity conditional on K>0 | Derived value |
|---|---:|
| Mean selected reward | 0.01990149 |
| Mean selected utility | −0.01527364 |
| Mean path length | 743.7705 mm |
| Mean update norm | 0.3702056 |
| Mean dense-candidate travel component | 0.9335835 |
| Mean dense-candidate overlap component | 0.0155360 |

The active routes select below-threshold rewards on average while some candidate must exceed the threshold at every executed state. This supports the different-candidate continuation mechanism; individual identities and score persistence still require traces. The small all-subject path/update statistics were diluted by 119 zero routes.

The cost identity also checks numerically:

\[
0.0162059684-0.05(0.0154311317)-0.2(0.0002567930)
=0.0153830532.
\]

Travel is bounded, but the **active-subject** weighted travel term is about 0.04668, still larger than their typical predicted rewards. Boundedness does not imply calibrated weighting.

## 4. Mathematical failure analysis

### Reconciling the earlier A/B/C sequence

The [earlier remediation report](trajectory-reward-logic-remediation-2026-09-01.md) and supplied history describe the **legacy** solver, where the same utility ranked and halted:

\[
U_i=r_i-\lambda_T\|p_i-p_{previous}\|/4-\lambda_O O_i-\lambda_{step}.
\]

At Z0, travel and overlap are zero, so continuation is exactly `max(r)>lambda_step`. Run A's 0.05 barrier and reported reward maximum around 0.0433 are consistent with near-universal K0. Lower barriers in B (0.025) and C (0.01) permit entry. After one write, the physical-distance/radius ratio appears for the first time: a 50-mm jump costs `0.05*(50/4)=0.625`, readily dominating rewards around 0.07–0.08. Hence positive first-round utility and K=1 can coexist with negative route-averaged utilities afterward. These are mechanisms consistent with the historical aggregates; per-round traces would be required to prove every individual stop's cause.

The new patch removes cost terms from the halt decision but retains a raw threshold of 0.025. It therefore removes the old post-first-step economic barrier while retaining a new, uncalibrated entry barrier. The reward network has also been trained under changed state sampling and terminal labels, so B's earlier success at 0.025 does not calibrate 0.025 for the new model. The travel ratio was already dimensionless; the old problem was its unbounded magnitude and unjustified tradeoff scale, not a literal units error. Reusing the smallest historical successful threshold does not establish a decision-theoretic margin.

### Is gain minus travel minus overlap plus separate halt coherent?

It can be coherent as an explicitly constrained heuristic: first identify actions with sufficient calibrated gain, then choose among them using locality preferences. It is also coherent to maximize expected gain minus a real execution cost and compare that same maximum against the stop action's value of zero. Neither requires conflating unscaled millimetres with reconstruction error.

The current implementation instead asks whether **any** action clears a raw threshold and then executes a potentially different low-gain action. This is not the one-step optimizer of net reconstruction benefit with a stop action, and it does not guarantee the selected action satisfies the stated improvement requirement.

A concrete counterexample, executed against the current solver:

```text
raw reward       [0.030, 0.024]
weighted costs   [0.020, 0.000]
ranking utility  [0.010, 0.024]
halt threshold   0.025
result           active=True, selected index=1, selected reward=0.024
```

The rejected-for-ranking candidate is the reason for continuing. If repeated local updates leave its three sampled plane footprints unchanged, its score remains above threshold. No step/budget feature in RewardNet decreases that score automatically. Even no-revisit would not remove this unselected “witness” candidate. No-revisit helps repeated actions, but does not by itself fix the logical mismatch.

### Is travel scientifically justified?

There is no scanner movement, robot motion, or new physical measurement between these points. All observation features are already computed; every round densely queries and scores all N candidates. Millimetre separation is not demonstrated to predict FLOPs, memory traffic, reconstruction improvement, or risk. Treat it as an **optional locality prior**, not an acquisition expense or physical necessity.

The 4-mm spatial write contract has a legitimate geometric meaning. A preference for short routes needs separate evidence. It can conflict with diversity: travel prefers nearby points; overlap discourages nearby points. Neither term measures the actual overlap of all affected output voxels, because a plane write influences extended 3-D fibres through the decoder. Points far apart in 3-D can share an affected XY, XZ, or YZ footprint.

For a first route-learning experiment, set these preference weights to zero and measure coverage/redundancy. This is removal of an unvalidated objective term to identify gain learning, not a sweep for a favorable K histogram.

### What should determine halting?

The defensible primary signal is **calibrated expected signed marginal improvement for the action that will actually execute**, compared with an explicit improvement/compute margin, plus a hard budget. A scalar reward sigmoid is neither a calibrated probability of benefit nor calibrated uncertainty.

| Candidate halt criterion | Judgment |
|---|---|
| Current raw reward | Reject as the sole criterion: clipped target, changing model, and unmatched calibration population |
| Calibrated signed gain | Best near-term foundation; state its loss/support/normalization units explicitly |
| Uncertainty alone | Insufficient: T1ce ambiguity need not be reducible by another latent update without new observations |
| Improvement margin | Necessary to distinguish tiny/noisy improvements from useful computation; derive and freeze it on a training calibration subset |
| Learned stop head | Plausible later; needs trustworthy stop/continue targets and state coverage, neither provided by the present loss |
| Cumulative budget | Necessary safety/compute cap, but a budget alone says nothing about usefulness |
| State/update convergence | Useful secondary telemetry; a small update does not prove small error, and a large update does not prove improvement |

A calibrated lower bound is a conservative operational rule, not a clinical confidence guarantee. Maximization across 2,048 candidates introduces selection bias: candidate-wide calibration is not automatically calibration of the winning candidate. Calibration must measure winners on fresh subjects and the relevant state-depth distribution.

### Greedy, bandit, or sequential control?

This task is best described as **budgeted adaptive latent computation**. Each action changes the next context, may damage distant output regions, and may interact with later actions. It is sequential decision making; a contextual-bandit model is a myopic training approximation at sampled states, not a complete description of its transitions. It is only analogous to active acquisition: it acquires no new MRI measurement.

Greedy one-step routing is an appropriate low-variance baseline while the gain target and UpdateNet are being validated. It has no demonstrated optimality here. Learned nonlinear writes can be noncommutative, nonmonotone and complementary; adaptive submodularity/diminishing returns have not been shown. A long-horizon RL policy would currently learn against a moving, partly mismeasured reward and greatly increase experimental cost. Measure one-step regret and then bounded two-step interaction before considering that escalation.

### Is RewardNet predicting the right target?

The actual detached label is:

\[
R_i^*=\operatorname{clip}_{[0,1]}
\frac{(\ell_{L,b}-\ell_{L,a})-\beta[\ell_{S,a}-\ell_{S,b}]_+}
{\ell_{L,b}+\epsilon},\qquad\beta=1.
\]

Local and spill losses are masked mean Charbonnier errors. The local domain is a 4-mm sphere; spill is 12 samples on three centre fibres. The label is **positive relative local improvement with a collateral-harm deduction**. It is not whole-volume Gate-E improvement, signed gain, a return, a probability, or uncertainty.

Consequences:

- Harmful and neutral actions both receive zero. For equal-probability gains of +0.04 and −0.10 in comparable relative units, the target's conditional mean is +0.02 while expected signed gain is −0.03. Perfect positive-part regression does not fix this.
- Per-candidate division can prioritize a 50% reduction from 0.001 over a 10% reduction from 0.1 even though the latter removes twenty times more absolute local error. The denominator is unavailable to inference and varies by region, subject and model age.
- Local and spill **means** get equal beta weighting without affected-volume weighting. Twelve centre-fibre samples do not cover all output fibres affected by the finite plane patch and bilinear interpolation. The label is a sampled proxy, not an unbiased global-loss estimator.
- SSIM, full-volume gradient loss and semantic loss are outside this label. Their gradients may change the reconstructor while RewardNet fits a different quantity.
- RewardNet's compressed `q_bar` input does not contain all raw `f_spec` information seen by UpdateNet. Even with fixed parameters, exact individual gain may be unidentifiable from the 126-dimensional descriptor. Conditional expectation is still meaningful, but residual calibration/within-state ranking must test whether this representation is adequate.

For the next bounded diagnosis, use **signed absolute local-minus-spill gain**, retain its proxy label explicitly, and test its agreement with actual whole-volume post-inference changes. If that agreement fails, replace the measurement with an affected-footprint, volume-weighted estimator before extending adaptive routing. Do not simply rename the current label “expected reconstruction gain.”

## 5. Code-level findings

### Decisions, availability, and gradients

| Question | Verified implementation |
|---|---|
| Ranking score | `route_utility` in corrected mode is `r - lambda_travel*T - lambda_overlap*O`; no step subtraction |
| Halt quantity | Maximum eligible raw reward, independently of selected index; strict `>` threshold |
| Availability | If a policy is supplied, exhausted rows stop before scoring; unavailable ranking entries receive a finite minimum sentinel and halt scores become zero. With nonnegative threshold, unavailable high rewards cannot keep a row alive. Dense scoring and telemetry still include them before masking |
| Server policy | `_TrainingContextModule.forward` passes no policy; model default is `None`. Both train and validation can revisit |
| Training floor | Only effective with separate-halt mode; first configured training rounds overwrite halt scores, but do not change candidate ranking. It cannot override exhaustion |
| RewardNet gradients | Direct SmoothL1/ranking loss on `descriptor.detach()`; additionally reconstruction/local/monotonic/delta can reach RewardNet through the active-route straight-through weights. No differentiable halt objective |
| UpdateNet gradients | Final reconstruction and actual local/monotonic/update-regularization paths through executed writes, including indirect effects on subsequent route states. No direct gradient from measured counterfactual targets |
| Terminal supervision | Requeries all candidate descriptors; chooses raw-reward argmax without availability; then recomputes all rewards for the selected/high/random mixture. Transition/decoder measurement is under no-grad and descriptors are detached. Direct reward gradient only |
| Frozen prior | Backbone parameters excluded from optimizer; `SemanticPrior.train` restores backbone eval, preventing BatchNorm-statistic training. No second MedicalNet pass in the traced frontend |

The pairwise implementation already exists but its coefficient is zero in the source config and supplied W&B config. For valid within-subject pairs separated by at least 0.001, it computes

\[
\left[|R_i^*-R_j^*|-\operatorname{sign}(R_i^*-R_j^*)(r_i-r_j)\right]_+.
\]

It is a target-gap margin hinge, not just an inversion indicator. Its reported violation fraction can be positive for correctly ordered but insufficiently separated predictions. Pair statistics are computed even when the weight is zero; they are not forwarded into the current epoch W&B row.

### Train/validation versus public inference is materially different

`GateGInferenceConfig.route_config` copies only travel/overlap/step weights, Kmax, temperature and write scale. It leaves `bounded_travel_cost=False`, `separate_halt_from_utility=False`, and exploration zero by default. Its default step value is 0.05. `run_baseline_inference` explicitly supplies this as an override to `_run`, even if the loaded model's trajectory config had the corrected flags. This was reproduced by constructing the effective config locally.

This may preserve historical Gate-G behavior intentionally, but it is a **high-severity evaluation-contract discrepancy** for any claim that the smoke run validates current end-user inference. Version the policy and compare identical effective configs before making such a claim. Do not silently migrate old checkpoints.

Terminal and ordinary counterfactual sampling also ignore availability. That is consistent with the current server route, which has no mask. After unifying on no-revisit, terminal “best candidate” and high/random probes must be selected from legal remaining candidates; otherwise the stop learner is trained on actions the inference policy cannot take. A terminal state at budget exhaustion is not evidence that all remaining gains are zero: retain its measured positive labels, and record a budget stop separately.

### Target leakage review

No target-to-inference **value path** was found in the inspected chain:

- The loader derives the brain mask and normalizes each observation from T1/T2/FLAIR before reading target data; the reference affine comes from the observations. Target normalization uses its own statistics and the observation mask. Segmentation is separate.
- The trainer passes only observations, mask, spacing and affine to the context wrapper. Target and segmentation are transferred to the model device and consumed after that context and prediction exist.
- Counterfactual candidate identities depend on prediction, fixed selected indices and RNG, not measured target error. Their measured results never replace the route state or resume the route.
- Target-derived training gradients into allowed trainable components are supervised learning, not inference leakage. Semantic labels influence the semantic head through the explicit post-context auxiliary loss, not via inference-time label inputs.

Limits: the data loader can read/normalize targets on CPU before inference; “target-after-inference” accurately describes the computation boundary, not file-I/O chronology. Target/segmentation validity can determine whether a sample is admitted, which is a cohort-selection issue, not hidden target-pixel routing. These checks do not audit the upstream provenance of server-preprocessed images, registration or skull stripping. The caller-supplied mask has no runtime proof of provenance. Server source parity also remains unverified.

### Logging correctness and omissions

At B=1, K metrics are correctly subject weighted. Candidate means include K0 scoring rounds and all scored candidates. Epoch “max” metrics are means of per-batch maxima, hence means of per-subject maxima here. At B>1, some quantities pool candidates/pairs within a batch and then weight by subjects, while candidate means use per-subject normalization; this produces batch-size-dependent meanings. Use explicit numerator/denominator reductions instead.

R-star and candidate reward are **not paired populations**: R-star covers up to 32 selected/high/random candidates per supervised state, includes terminal states, and discards invalid local support; candidate reward covers all 2,048 candidates in route scoring rounds. A Kmax route has an additional terminal supervised state absent from route telemetry. This prohibits interpreting their mean difference as a measured calibration bias.

The JSON stop histogram is put in the local epoch row, then removed from W&B by the numeric-only `wandb_run.log` filter. Train mean K and train stop counts are reduced internally but not included in the W&B row. `step_cost` is actually a halt threshold in corrected mode; `nonpositive_utility` actually denotes insufficient raw gain. These labels obstruct diagnosis.

### Executed local checks

Environment: local Python/PyTorch 2.13.0, pytest 9.1.1, CPU synthetic data only; no pretrained-weight claim.

- `test_reward_route_logic_fix.py`: **3 passed**. Its terminal test asserts a positive finite reward loss, but never calls backward; it does not establish useful gradient magnitude or improved routing.
- `test_trajectory_cost.py`, `test_trajectory_solver.py`, `test_reward_supervision.py`, `test_training_objective.py`, `test_frontend_forward.py`: **60 passed** together, including frontend smoke/invariants.
- Independent inline solver probe reproduced the different-candidate continuation counterexample above.
- Independent inline K0 reward-only backward: nonzero gradients in two RewardNet parameter tensors, none in UpdateNet, decoder or state initializer; context prediction and final planes remained unchanged. The synthetic loss was only approximately `1.03e-9`, illustrating why “nonzero” is weaker than “useful.”

Passing tests demonstrate implemented semantics, including the problematic separate-maxima behavior. They do not certify the formulation, server execution or reconstruction value.

## 6. Training/inference mismatch analysis

**Terminal supervision solves the zero-direct-RewardNet-gradient branch, not the full self-locking system.** At K0, UpdateNet has no executed transition to receive reconstruction/local gradients. A reward target built with a weak updater can legitimately be near zero; fitting it better can reinforce stopping. Forced steps provide updater examples, but do not establish that those examples improve reconstruction or generalize to later states.

`training_exploration_steps=1` directly explains train K0=0 versus validation K0≈0.983 without invoking overfitting. The first training step is guaranteed whenever candidates remain, whereas eval allows immediate halt. Training statistics also average checkpoints evolving throughout the epoch; validation uses the final epoch checkpoint on different subjects. Compare identical subjects and weights under both modes with forcing disabled to isolate the policy mismatch. The frozen backbone's BatchNorm is not a supported explanation for this discrepancy.

UpdateNet bootstrap remains coupled in three ways:

1. The current updater/decoder define labels anew every batch, while the same optimizer updates them and RewardNet. This is a moving supervised target, not stationary gain regression.
2. Forced actions remain greedy under RewardNet/locality costs. Selected/high-reward counterfactual examples are preferentially sampled; random **counterfactuals** train RewardNet but do not give UpdateNet gradient-bearing random actions.
3. Reconstruction gradients use a biased straight-through selection surrogate, potentially pulling RewardNet scores away from absolute calibration. At reward differences around 0.01 and temperature 1 across 2,048 candidates, the soft backward distribution can be much flatter than the hard argmax forward action. The logged global gradient norm does not resolve this.

The target is zero-inflated, moving, clipped and locally normalized. With ranking disabled and rewards in [0,1], default SmoothL1 lies in its quadratic region: (L=\tfrac12(r-R^*)^2). A small value around 0.000696 corresponds to roughly 0.0373 RMS residual on that training loss population, not evidence of accurate top-1 decisions. Near r=0.016, sigmoid derivative is only about 0.0157 before the residual/averaging factors.

`R_star_mean≈0.032` versus reward mean≈0.016 is a **calibration warning**, not proof of 2× underestimation. The sampler, validity mask, state inclusion and train/eval times differ. Log paired predicted/target values on identical valid candidates, separately for selected/top/random, Z0/intermediate/terminal, and new subjects. Compare to a constant-mean predictor and report top-1 regret. Do not double rewards based on two unpaired means.

Pairwise ranking deserves a bounded auxiliary objective because routing uses argmax. It cannot replace absolute signed calibration for stopping, recover signs erased by clipping, or repair an incorrect target. Freeze the reconstructor first; use informative within-state pairs and report inversion accuracy separately from margin violations. Warm up UpdateNet before separately fitting RewardNet; warming RewardNet against a useless cold updater alone is not a solution.

## 7. Runtime analysis

### Multiplicative structure

For subject s with executed length (K_s), the objective supervises (K_s+1) states when terminal supervision is enabled. With M=32 probes and nonzero spill sampling, per state it performs:

- one dense dynamic query and RewardNet pass over N=2,048 candidates (terminal setup adds another pass);
- M hypothetical UpdateNet transitions and tri-plane writes;
- **4M `decoder.decode_points` calls**: before/after local and before/after spill;
- **4M target/mask volume sampling calls**, and 2M conversions of the full boolean mask to floating point;
- candidate/spill sampling loops, affine transformations, scalar validity checks and host/device synchronization.

Actual selected-step supervision additionally performs **three local decoder calls per executed step**: before and after on that step's sphere, plus after on the first-step fixed monotonic sphere. The first local `before` result is computed but unused by the objective. At t=0 the monotonic support also repeats the just-decoded local-after support. Full final-volume Charbonnier, six SSIM convolutions (mask plus five moments), finite differences, and semantic loss are also within the Gate-E timer.

Thus, excluding the one final dense decode and shared frontend:

\[
N_{CF}=32\sum_s(K_s+1),\qquad
N_{decode}=128\sum_s(K_s+1)+3\sum_sK_s.
\]

For the conditional 121-subject validation reconstruction: 249 supervised states, 7,968 hypothetical transitions, **32,256 point-decoder calls**. There are 247 inference scoring rounds (119 immediate stops plus 128 executed rounds), hence 505,856 dense candidate evaluations at N=2,048. Counterfactual and terminal scoring are additional work outside that count.

For 965 training subjects, train K is missing. With the forced floor and cap, counterfactual transitions range from **61,760 to 2,007,200 per epoch**, and point-decoder calls from **249,935 to 8,214,080**. Do not substitute validation mean K into training's runtime model. The high train travel mean (0.88954) is also inconsistent with assuming that every subject merely contributes one first-round score and one halt round with modest travel; train routes need direct logging.

At 1-mm isotropic spacing, `build_local_support_samples` constructs a padded `11^3=1,331` voxel-centre cube for each local query. About 257 centres satisfy the radius for an interior grid-aligned point, but **all 1,331 are decoded before masking**. Per candidate this is 2×1,331 + 2×12 = 2,686 decoder point evaluations, or 85,952 per supervised state before actual-step terms. For other spacings/shears, the cube width is

\[
2(\lceil4/\sigma_{\min}(A)\rceil+1)+1,
\]

using the affine's minimum singular value. The supplied exports do not prove subject geometry, so 1-mm counts are an illustrative concrete case, not an inferred server invariant.

### Hidden full-array work

`_sample_target_support` calls `sample_volume_ras_mm` for target and float mask. That function invokes `_require_finite_float_tensor` over the **entire volume** each time. With local and spill supports, this is 128 full-volume finite scans and 64 full-mask float conversions per supervised state at M=32. At an illustrative `155×240×240` grid (8,928,000 voxels), that is roughly 1.14 billion voxel finite checks and 0.57 billion mask conversions per state, before other validation. Even the mask's already-established validity is rescanned.

`_validate_target_and_mask` additionally validates and constructs a safe whole target per counterfactual state call. `_write_plane` clones each entire latent plane although it alters a tiny patch, followed by stacking and `DynamicTriPlanes` full finite validation. It recomputes CPU float64 singular values six times per hypothetical transition—two identical calls per plane—and calls `.item()` to form windows. Spill sampling uses `.item()`/`.tolist()`, many tensor validation predicates use `bool(tensor)`, and sampled errors repeatedly copy counts to CPU. Small decoder calls therefore carry substantial launch/synchronization overhead.

These are exact expensive operations visible in code. Their individual share of 34,117 seconds is **not profiled**. `_StageTiming` uses CUDA event elapsed times on GPU and includes stream idle intervals between recorded events, so it is not a pure kernel-FLOP attribution. Gate-E also includes semantic loss. A profiler trace and substage counters are needed before claiming which operation dominates within Gate-E.

### Preserve supervision while cutting cost

1. Validate/detach/sanitize the target once per subject context; cache a float support mask, physical lattice transforms, singular-value bounds and candidate support indices. Preserve public input validation; use a private validated sampling context inside the repeated loop. Invalidate caches when geometry, mask, refined points, dtype or state/parameter identity changes.
2. Gather only actual valid local voxel centres; local centres are integer voxel positions, so exact target/mask indexing can replace trilinear target sampling there. Keep trilinear interpolation and original validity semantics for fractional spill points.
3. Concatenate local/spill point queries to share decoder calls; batch candidate probes in chunks. Reuse before-state point samples/predictions within the same frozen state. Do not cache after-state labels across updater/decoder changes.
4. For detached counterfactuals, query the original planes plus **exact sparse patch contributions** at query locations, avoiding full-plane clones. Bilinear sampling is linear in plane values, so the correction to the 96-vector can be computed before the unchanged nonlinear decoder MLP. Match the actual discrete write weights and four interpolation neighbors, including boundaries/affines; a continuous radial shortcut would not be equivalent. Keep the actual inference writer as reference and test equality.
5. Cap reward-supervised states at three per subject (Z0, terminal, one seeded interior), with eight probes (one proposed/selected, three high, four random). Increase spill samples to 48 to assess collateral noise while still reducing the expensive local-probe factor. Keep actual-step updater gradients on the short fixed route. Skip unneeded losses before computation when a training stage has frozen their recipients.

Against K=64/M=32, 3×8 rather than 65×32 is an **86.7× reduction in counterfactual transitions**, not an end-to-end speedup promise. At K=1 the factor is only 64/16=4 when the two unique states are used. Larger spill budgets and batching overhead reduce the decoder-query speedup. Removing repeated full-volume scans/casts is valuable independently of these reductions. Require measured ≥4× reduction of the counterfactual substage on identical saved states/queries; do not claim that batching alone achieves it.

## 8. Literature comparison

These are primary papers or author-maintained code/docs, consulted during this review. The mechanisms support design choices; none validates this repository's latent-update formulation or clinical utility.

| Reference | Exact relevant mechanism | Borrow / do not borrow |
|---|---|---|
| Zhang et al., CVPR 2019, [Reducing Uncertainty in Undersampled MRI Reconstruction With Active Acquisition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.html), [paper](https://arxiv.org/pdf/1902.03051) | Reconstructor produces an image and uncertainty estimate; an evaluator rates prospective k-space measurements and iteratively guides acquisition. Estimated uncertainty is compared with actual error and motivates an acquisition halt signal. | Borrow the separation of measurement selection, reconstruction, and empirical uncertainty/error validation. Do not equate cross-plane agreement or a reward sigmoid with calibrated uncertainty. Their next action obtains new measurements; ours cannot reduce missing-contrast ambiguity by obtaining new T1ce evidence. |
| Pineda et al., MICCAI 2020, [Active MR k-space Sampling with Reinforcement Learning](https://arxiv.org/pdf/2007.10469) | Fixed reconstructor; partially observed acquisition state, invalid previously acquired columns, finite horizon; Double-DQN uses replay, exploration and temporal-difference targets for future return. | Borrow a stationary reconstructor during policy learning, explicit legality/budget, and measured metric differences. Do not label the current one-step SmoothL1 target a Q value or import full RL before validating marginal rewards. |
| Author code: [active-mri-acquisition](https://github.com/facebookresearch/active-mri-acquisition), [environment API](https://facebookresearch.github.io/active-mri-acquisition/envs.html) | `step` returns new observation, measured score improvement and done; `budget` sets episode length; `try_action` measures a hypothetical action without changing environment state. | Borrow immutable counterfactual evaluation and separate budget/score accounting. Their ground-truth-backed simulator metadata must never become this model's inference inputs. |
| Bakker, van Hoof & Welling, NeurIPS 2020, [Experimental design for MRI by greedy policy search](https://proceedings.neurips.cc/paper/2020/file/daed210307f1dbc6f1dd9551408d999f-Paper.pdf) | Fixed pretrained reconstruction model; policy-gradient acquisition under a budget; compares greedy γ=0 with longer-horizon objectives and analyzes gradient signal-to-noise. Greedy was competitive in their experiments. | Borrow the fixed-model controlled comparison and start with immediate-gain routing. Do not infer a theorem that greedy is optimal for nonlinear latent updates or that longer-horizon learning never helps. |
| Isler et al., ICRA 2016, [An Information Gain Formulation for Active Volumetric 3D Reconstruction](https://rpg.ifi.uzh.ch/docs/ICRA16_Isler.pdf), [author code](https://github.com/uzh-rpg/rpg_ig_active_reconstruction) | Eqs. 11–13 rank by normalized information gain minus normalized robot movement cost and stop when every view's information gain is below a threshold. Robot interface supplies feasibility and movement costs. | This is a real precedent for separate ranking and stopping, so separation itself is not theoretically absurd. Borrow explicit units/constraints. Do not claim their normalization is `d/(4+d)`, or that robot energy/registration cost justifies latent-point travel. Their equations do not prove selected-action gain adequacy in this different task. |
| Singh et al., IJCAI 2007, [Efficient Planning of Informative Paths for Multiple Robots](https://www.cs.cmu.edu/~guestrin/Publications/IJCAI2007/ijcai-2007.pdf) | Maximizes an information objective subject to travel/measurement budgets; spatial decomposition and branch-and-bound make informative-path planning practical. | Borrow explicit budgets and marginal information accounting if a real acquisition resource exists. Do not import robot path length as a compute cost or call our learned objective submodular. |
| Golovin & Krause, [Adaptive Submodularity: Theory and Applications](https://arxiv.org/abs/1003.3967) | Adaptive diminishing returns yields guarantees for adaptive greedy policies under the paper's assumptions. | Borrow the requirement to test/establish diminishing returns before invoking greedy guarantees. Nonlinear, harmful, order-dependent writes here violate any automatic appeal to those results. |
| Graves, 2016, [Adaptive Computation Time](https://arxiv.org/pdf/1603.08983) | Sigmoid halt units accumulate to a threshold; a remainder allocates probability mass; weighted intermediate outputs and a ponder penalty permit differentiable learning of computation. | Borrow explicit computation penalties and a real halt-learning objective. Do not copy weighted intermediate output mixing into the locked final-Z decoder contract. |
| Banino, Balaguer & Blundell, 2021, [PonderNet](https://arxiv.org/pdf/2107.05407) | Hazard (h_t) defines (p_t=h_t\prod_{j<t}(1-h_j)); expected prediction loss is regularized toward a geometric halt prior. Training explores possible computation depths. | Borrow state-depth coverage and explicit stop distributions if a later stop-head study is justified. It adds unrolling/loss costs and a prior, not free calibration or guaranteed useful updates. Deterministic thresholding would be a new deployment choice to validate. |
| Burges, 2010, [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) | Within-query score differences train pairwise preferences; LambdaRank weights pair changes by ranking-metric consequences. | Treat one subject/state as a query; use measured gain gaps and top-1 regret. Ranking is invariant to score offset and cannot alone support an absolute halt threshold. The current hinge is not RankNet's logistic pair loss. |
| Kuleshov, Fenner & Ermon, ICML 2018, [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://proceedings.mlr.press/v80/kuleshov18a.html) | Held-out recalibration of predicted cumulative probabilities improves regression uncertainty calibration. | Borrow a separate calibration dataset and coverage checks. A point-estimate affine calibrator is not their full predictive-distribution method, and marginal calibration does not guarantee selected-action or shifted-state calibration. |
| [Vowpal Wabbit contextual-bandit documentation](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html) | Context → action → selected-action feedback, with explicit exploration and regression/importance-weighted learning reductions. | Borrow logged action probabilities for random behavior data. Here counterfactual labels offer richer feedback and actions alter future contexts; ordinary contextual-bandit guarantees do not cover the entire route. |

## 9. Redesign candidates

Notation: (g_i^t) is a measured **signed** post-inference one-step gain in a declared objective; (\mu_i^t) is its calibrated prediction; (m_i^t) a validated error allowance; (c_i^t) a justified cost; B an explicit update budget. The near-term proxy and calibration are specified in section 10.

| Rank / design | Equation and training target | Inference rule | Advantages | Risks / compute / compatibility |
|---|---|---|---|---|
| **1. Fixed-budget bootstrap → calibrated gain with action-consistent halt** | Bootstrap UpdateNet with target-free fixed-budget actions. Freeze reconstructor, fit signed (g_i^t) by squared regression plus a bounded within-state ranking term. Calibrate on different training subjects. | Legal profitable set (F_t=\{i:\mu_i^t-m_i^t>\delta\}); stop if empty or budget exhausted; otherwise select max gain (or documented locality ranking only within F). | Breaks updater starvation and moving-target confounding; gives stopping an interpretable quantity and matches the executed action. | Several explicit training stages; small proxy can miss collateral harm. Eight probes at at most three states; cheap cached fitting. Reuses most code; requires versioned head/target/config and F/G policy unification. |
| **2. Calibrated signed gain + separate halt immediately** | (L=E[(\hat g-g)^2]), optionally ranking; calibrate (\mu=a\hat g+b). | Same profitable-set rule, (K\le B). | Smallest adaptive algorithm; no stop network. | Without a trained updater and broad state data, accurately predicts poor updates and stops. Joint fitting keeps target drift. Similar online inference cost; existing counterfactual cost unless optimized. |
| **3. Ranking-only RewardNet + external fixed budget** | Within-state pair loss (\log(1+e^{-s_{ij}(v_i-v_j)})) for reliably different signed gains. | `argmax(v)` among available points, exactly (\min(B,N)\) updates; no score-based halt. | Clear top-1 objective; stable compute; excellent diagnostic/control arm. | May force harmful updates, no individualized stopping, ranking scores have no gain units. Cheapest migration; no absolute threshold should be applied to v. |
| **4. Learned stop head** | Observation-only state summary h predicts stop. For example train BCE against whether a fixed, post-inference probe set has any reliable gain above δ, or train hazard probabilities with expected loss plus a computation prior. | Stop on calibrated stop probability or consume remaining budget; route gain scorer chooses actions otherwise. | Learns state-level stopping independently of reward offset. | Probe-set “no good action” is not exhaustive; noisy maximum labels can reproduce collapse. Needs separate calibration, stop/continue balance and train-time unrolling. Moderate new parameters, larger contract change; not preferred before gain is measurable. |
| **5. Budgeted sequential value policy** | (Q(s,i,b)=E[g_i+\gamma\max(0,\max_jQ(s',j,b-1))]); STOP has value zero. Train on bounded rollout/TD targets with a fixed transition model. | Choose the maximal value among legal actions and STOP, decrement b. | Can represent delayed benefits and action interactions. | Bootstrapping bias, offline coverage and reward error become compounded; 126-d local descriptor may not be Markov. Replay/target network or lookahead adds scope and high compute. Lowest near-term compatibility; defer until two-step interactions demonstrate need. |

Options 1 and 2 are the same eventual inference architecture but different learning protocols. Rank 1 is the recommended **next implementation** because the evidence specifically implicates training dynamics as well as the decision rule. Option 3 is a required matched-budget control, not an alternative success criterion. None permits targets to enter inference or MedicalNet unfreezing.

## 10. Recommended implementation

### Versioned signed-gain diagnostic

Implement a new explicit experimental protocol, e.g. `signed-gain-budget-v2`; preserve old checkpoint/config semantics with explicit version checks. This changes locked C/E/F/G contracts and needs a separately named implementation task/plan update. It is not Gate H or an architecture-wide redesign.

Use the existing local and spill measurements initially, but replace the reward label with:

\[
g_i^t=(\ell_{L,b}-\ell_{L,a})-\beta[\ell_{S,a}-\ell_{S,b}]_+.
\]

**No positive clipping and no division by target-derived local pre-error.** This is an absolute-intensity local/spill proxy in the existing robust-[0,1] normalization space, not global-volume gain. Keep `local_gain`, signed `spill_change` and the old R-star as separate audit fields. Even this conservative proxy can be biased by rectifying a noisy spill estimate; measure that noise with independent samples.

Keep the `126 -> 64 -> 1` RewardNet MLP and remove the final sigmoid only in v2. Reinitialize its final linear layer for the new units; do not interpret a legacy sigmoid checkpoint as a signed head. Normalize the regression numerically with a fixed training-only scale (s=\max(\operatorname{RMS}_{train}(g),10^{-4})), stored in metadata. Predict (\hat g=s f_\theta(d)). Train on detached descriptors/labels with:

\[
L_R=\operatorname{mean}(f_\theta(d)-g/s)^2
+0.1\operatorname{mean}_{|g_i-g_j|>\eta}
\left[\frac{|g_i-g_j|}{s}-\operatorname{sign}(g_i-g_j)(f_i-f_j)\right]_+^2.
\]

Here (\eta=\max(10^{-4},2q_{0.9}(|g^{seed1}-g^{seed2}|))), estimated on training-only bank data and frozen. The squared, normalized hinge keeps units compatible with regression; it is not simply enabling the existing unnormalized hinge at weight 0.1. Record the base regression and ranking gradient norms. A paired MSE-only shadow fit on the same cached examples isolates whether the ranking auxiliary helps; no additional volume inference is needed.

Fit a nonnegative-slope affine calibrator (\mu=a\hat g+b) by least squares on a separate training calibration-fit cohort. A nonpositive covariance yields a=0, which is a diagnostic failure of discrimination, not a reason to invent a threshold. On another training calibration-margin cohort, run the frozen greedy policy for a target-free diagnostic budget of four and measure the proposed winner at every state afterward. Let (e_s=\max_t(\mu_{a_t}^t-g_{a_t}^t)); set (m=\max(0,q_{0.9}(e_s))). This is a subject-level empirical error allowance over a bounded path, **not a guaranteed conditional confidence interval**. Freeze a, b and m before held-out validation. Log leave-one-subject-out sensitivity because the calibration cohort is small.

Set (\delta=\max(10^{-5},q_{0.99}(|g_{noop}|))) from exact no-op measurements in the training bank. Continue only if the selected legal candidate has (\mu_i-m>\delta); otherwise stop. A hard cap B=4 remains independent. In this first diagnostic, travel and overlap weights are zero, so the maximum calibrated gain and selected candidate coincide. If priors are reintroduced later, filter by that profitable set **before** ranking, then assert that the chosen action clears the margin.

No-op differences test numerical consistency, not the real compute price of an update. The experiment asks whether positive useful updates can be identified; a deployment latency/quality tradeoff is a later explicit decision.

### Exact proposed code changes

| File / responsibility | Required change |
|---|---|
| `trajectory_cost.py` | Add versioned policy and explicit `halt_gain_threshold`/budget semantics; allow nonnegative locality coefficients including zero. Reject conflicting legacy/v2 fields. Do not reuse `lambda_step` under a new unit system silently |
| `trajectory_solver.py` | Explicit legal mask, profitable mask and proposed index; select only profitable legal candidates; stop on empty set. Expose proposed gain, margin, reason and whether continuation was forced by a declared diagnostic budget. Use hard selection without RewardNet straight-through reconstruction gradients during isolated gain fitting |
| `trajectory.py` | Thread policy/availability consistently; use fixed-budget random actions only in the named bootstrap protocol. Remove permanent train-mode-only forcing from v2. Log per-state selected versus maximum-gain candidate and eligibility. Retain target-free shared traversal, bounded update and geometry contracts |
| `reward.py` | Versioned signed output (same MLP width), stored fixed target scale; original sigmoid mode retained for legacy checkpoints. No T1ce input or extra backbone |
| `reward_supervision.py` | Signed proxy, paired diagnostics, legal selected/high/random sampling, independent RNG streams, cached validated target sampler, batched queries and exact sparse detached counterfactual evaluation. Keep measured targets/no-grad transitions isolated from RewardNet's gradient |
| `training_objective.py` | Explicit phase ownership; skip expensive reward branch during updater-only bootstrap; at most three distinct supervised states with Z0/terminal retained. Preserve live executed-step updater gradients. Budget terminal labels remain measured, never fabricated zero stop labels |
| `baseline_training.py` and `training/point_guided.py` | Named updater-only and reward-only optimizer sets, snapshot/hash assertions for frozen modules, fixed reconstructor during reward fitting, cached-bank versioning. Shared effective policy for train evaluation, validation and Gate G; explicit calibration partitions and deterministic behavior seeds |
| `baseline_inference.py`, config/checkpoint metadata, CLI loading | Carry the complete v2 policy, scale and frozen calibrator from checkpoint; no legacy rebuild dropping flags. Exact no-revisit at all matched inference boundaries. Refuse a missing/mismatched calibration artifact rather than silently falling back |
| New diagnostic config/runbook | Separate file `configs/training/point_guided_route_diagnostic_v2.json`; do not overwrite the 4070 production baseline while testing |
| Focused tests | Different-candidate counterexample; unavailable high reward; equality threshold; exhaustion; K0 reward-only gradients; signed harmful labels; no-op; target substitution invariance; policy parity; frozen parameter hashes; scalar versus batched/sparse counterfactual equality with rotation/shear/anisotropy; aggregation invariance |

Proposed config values below are **new schema fields, not executable with current main**:

```json
{
  "protocol": "signed-gain-budget-v2",
  "trajectory": {
    "k_max": 4,
    "lambda_travel": 0.0,
    "lambda_overlap": 0.0,
    "training_exploration_steps": 0,
    "availability_policy": "exact_no_revisit",
    "write_scale": 0.1,
    "support_radius_mm": 4.0,
    "selection_gradient": "hard_detached"
  },
  "supervision": {
    "reward_target": "signed_absolute_local_spill_v2",
    "counterfactual_candidates": 8,
    "high_candidate_count": 3,
    "random_candidate_count": 4,
    "max_reward_states": 3,
    "retain_initial_and_terminal": true,
    "spill_sample_count": 48,
    "spill_weight_beta": 1.0,
    "reward_loss": "scaled_mse_plus_squared_gap_hinge",
    "reward_ranking_weight": 0.1,
    "reward_scale": "training_bank_rms_floor_1e-4",
    "pair_gap": "max_1e-4_and_twice_training_repeat_q90"
  },
  "calibration": {
    "fit": "nonnegative_slope_affine",
    "error_allowance": "positive_part_subject_max_overestimate_q90",
    "improvement_margin": "max_1e-5_and_training_noop_abs_q99",
    "frozen_before_validation": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 1,
    "amp": false,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "decoder_chunk_size": 16384,
    "seed": 20260813,
    "bootstrap_fixed_k": 4,
    "bootstrap_action_policy": "seeded_uniform_without_replacement",
    "freeze_reconstructor_during_reward_fit": true
  }
}
```

Keep normalization, observation channels, MedicalNet hash, geometry, N=2,048 points, refinement bound and update architecture unchanged. Epoch-one updater loss retains existing reconstruction/local/monotonic/delta weights; semantic/RewardNet training is inactive in that stage. Later stages optimize the explicit reward objective only. Zero coefficients must skip construction of unused expensive terms, not merely multiply them by zero afterward.

### Required new metrics

Every record includes subject ID, split role, epoch, checkpoint hash, policy version, candidate seed and state index; target-bearing measurements live in a separate post-context diagnostic record.

| Group | Exact fields and denominators |
|---|---|
| Route distribution | `K_histogram/0..4`, `fraction_K0`, `fraction_Kmax`, `K_mean`, `K_given_active_mean`, `K_median`, `K_p90`, numeric `stop_count/{low_gain,budget,exhausted}` for both train and val |
| Each decision | `legal_count`, `profitable_count`, `proposed_index`, `selected_index`, `argmax_gain_index`, `max_legal_gain`, `selected_gain`, `calibrated_gain`, `error_allowance`, `halt_margin`, `forced_budget_step`, `witness_mismatch`, `revisit_count` |
| Score populations | Separate all-scored / eligible / supervised / selected counts and sums; global max reduced with MAX; subject-mean-of-max explicitly named; positive raw utility and positive halt margin separately |
| Paired calibration | `gain_pred`, `gain_target_signed`, `gain_legacy_clipped`, `local_before/after`, `spill_before/after`, valid local/spill counts, selected/high/random provenance; paired bias, MAE/RMSE, constant-baseline MSE, slope/intercept, overestimate quantiles by depth |
| Ranking | Pair count, reliable inversion/tie fraction, gap-hinge violation fraction, within-state rank correlation, probed `top1_regret=max(g_probe)-g_proposed`, selected harmful/neutral/useful fractions; explicitly label subset regret as a lower bound on all-candidate regret |
| Stop quality | False stop on a useful probed candidate, false continuation on harmful measured selected action, conservative lower-bound empirical coverage, with subject counts and confidence intervals; no inference decisions use these target-derived fields |
| Update value | Post-inference `L(Z0)-L(ZK)` on the same target/mask, signed local/spill deltas, no-op baseline, unique-point and plane-footprint coverage, per-step update norms; fixed/random versus learned under matched budgets |
| Gradients | Reward direct-regression/ranking norms separately; UpdateNet executed-step gradient norm and changed-parameter fraction; frozen module hashes; pre/post clip norms |
| Runtime | `reward_states`, `candidate_probes`, `local_valid_points`, `local_padded_points`, `spill_points`, `decoder_calls`, `full_volume_validation_calls`, `mask_casts`, `plane_clone_bytes`; seconds for validation/cache, sampling, counterfactual update/write, decoder, reward fit, local/monotonic, full reconstruction/semantic, backward |

## 11. Exact next experiment

This is **one bounded three-epoch diagnostic**, with a fixed protocol and cached-score controls, not a threshold sweep. It requires implementing and locally validating section 10 first. The current report does not run it.

### Prerequisites and cohorts

1. On the server, recover the supplied run's resolved config, exact split and epoch-three checkpoint/source identity. Save hashes and effective policy. If unavailable, stop the reproduction claim; any new initialization must be labeled a new experiment. Use the verified epoch-three model's frontend/decoder/UpdateNet as the warm start; reinitialize only the v2 RewardNet final linear layer and start fresh stage optimizers. Do not select a checkpoint using the coming validation results.
2. Preserve the existing train/validation/test split. Within the existing training IDs, sort by `SHA256("route-v2|20260813|" + subject_id)`; use the first **128** for fitting, next **32** for calibrator fitting, next **32** for the error allowance. Write the fixed manifest before reading target values. Keep all existing validation IDs (expected 121; verify), and do not open test payloads. Calibration subjects are not clinical held-out evidence; they are training resources.
3. One GPU, FP32, same MedicalNet SHA-256 as metadata (`afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`), original normalization and spatial data. Separate RNG streams for route actions, candidate probes and spill repeats, keyed by subject/state/epoch so validation RNG does not depend on training global_step.
4. First run CPU tests and a two-subject server preflight, including the full inference-policy parity and sparse-counterfactual equality tests. Profile identical saved states with old/new sampling implementations before claiming speedup. Stop on any target-boundary, numerical or parity failure.

### Three epochs

| Epoch | Parameter updates and behavior | Required evidence |
|---|---|---|
| **1: updater bootstrap** | One pass over 128 fitting subjects. Freeze MedicalNet, semantic/refiner/B/A/Z0 and decoder parameters plus RewardNet; train **UpdateNet only**. Exactly four target-free seeded-uniform distinct actions per subject; decode final state before target use. Use existing actual reconstruction/local/monotonic/delta losses, no counterfactual reward branch. At epoch end freeze UpdateNet too and create an immutable target-free state/probe bank, then attach measured signed labels afterward | Updater gradients and frozen hashes; random K=4 versus Z0 paired gain; split `local_gain` from spill harm. Build scale and repeat-noise statistics from fitting subjects only. If updater shows no learnable beneficial actions, record that outcome rather than force adaptive success |
| **2: stationary reward fit** | Reconstructor fully frozen. One pass over cached fitting states using eight probes per state, max three states per subject (Z0, Z4, one seeded interior). Optimize only RewardNet using the scaled loss. Train an MSE-only shadow scorer on the same bank/order. Fit a,b on the 32 calibration-fit subjects and m on the other 32 using frozen target-free greedy four-step diagnostic routes; freeze calibration before validation | Paired gain regression, ranking and winner calibration against constant baseline; direct comparisons of shadow/primary top-1 regret. Evaluate v2 adaptive K≤4, fixed greedy K=4 and fixed uniform K=4 with identical frozen reconstruction modules |
| **3: policy-state coverage and final decision** | Keep reconstructor frozen. One pass over the same 128 fitting subjects: 64 preassigned subjects use fixed uniform K=4 contexts, 64 use target-free adaptive K≤4 contexts from the frozen epoch-two scorer/calibrator. All routes finish before any target labels are attached. Probe at most three states, including K0/terminal. Freeze collection policy during this pass; fit RewardNet and its MSE-only shadow from the completed bank, then refit/freeze calibration on the training calibration cohorts | Repeat the exact validation arms and metrics; export per-subject routes and all scalar numerators/denominators. No epoch-four continuation and no validation-selected threshold or checkpoint |

Epoch-two/three cached fitting uses one optimizer update per subject after averaging its valid state/probe losses; learning rate 1e-4. This intentionally gives the initial, terminal and sampled interior states equal diagnostic weight; it is not an unbiased estimate of the previous all-state Gate-E mean. If preserving that previous estimand is required later, the sampled interior term needs its inclusion-probability weight (K−1 for one uniform interior state), with valid-count normalization handled explicitly. Keep the complete subject bank immutable during a fitting pass; do not rewrite labels with a newly updated reconstructor. Updater is fixed throughout both passes, so measurements are stationary conditional on state/action. The epoch-three split of behavior policies is made by the prewritten subject order, not target quality.

For each validation arm, create the complete target-free context/route first, then measure outputs. Cache/share one observation frontend result across arms at the same frozen checkpoint; the arms are separate counterfactual trajectories from the same Z0. A forced greedy diagnostic continuation beyond an adaptive halt is a **separate target-free budgeted rollout**, never resumption prompted by observed error. This permits paired false-stop analysis without a target-driven route. Record Z0 and final dense reconstruction for post-inference delta; do not expose intermediate dense predictions to inference.

Compute guards: max four updates, max three reward states, eight probes, no automatic longer run. Cap total GPU wall time at **four hours** and stop after a completed subject if exceeded; an incomplete diagnostic is not a pass. Profile the first eight fitting subjects and extrapolate before consuming the remaining budget. Do not reduce cohorts or probe quality midway to obtain a pass. Expected speedup is to be measured, not assumed from the old 40,278-second epoch.

### Predeclared pass/fail criteria

These are engineering/research diagnostic thresholds, not clinical standards. Report subject-bootstrap 95% intervals with 2,000 resamples, seed 20260813; do not treat state/candidate pairs as independent subjects.

| Gate | Pass condition | Failure interpretation |
|---|---|---|
| Software and provenance | Exact source/config/checkpoint hashes; zero target-swap route changes; exact shared-policy route parity; zero selected unavailable/revisited candidates; all automatic decisions satisfy `selected calibrated gain - m > δ`; frozen module hashes unchanged | Stop before training/evaluation claims |
| Numerical equivalence | On identical synthetic and saved real states, optimized versus reference counterfactual decoded values/gains agree within `atol=1e-6, rtol=1e-5` in FP32; no-op absolute gain ≤1e-6 | Runtime optimization changed semantics or numerical support; fix first |
| Gain signal exists | At least 30 validation subjects have a probed action with signed gain above δ plus measured sampling-noise allowance; otherwise label the experiment underpowered for route-learning claims | K0 may be correct; bootstrap/measurement is unresolved, not necessarily halt failure |
| Gain/ranking fit | Paired validation MSE at least 10% below training-mean constant predictor; reliable pair ordering >60% and bootstrap lower bound >50%; primary top-1 subset regret not worse than MSE-only shadow by >5% | Scorer/descriptor or ranking auxiliary is not justified; report which part failed rather than tune threshold |
| Action and halt quality | ≥80% of automatic continued subjects have positive mean measured selected gain; harmful selected-action rate ≤10%; false-stop rate ≤20% among subjects with a useful **probed** alternative; empirical selected-gain lower-bound coverage ≥85% | Calibration, state coverage or target measurement failed; do not use K distribution to override this |
| Route contribution | Learned fixed-K mean signed proxy gain exceeds matched random K=4 with bootstrap lower bound >0. Among subjects whose separately completed greedy diagnostic has reliable positive gains at both Z0 and Z1, ≥80% take ≥2 adaptive steps; report this denominator and mark the depth check inconclusive if fewer than 20 subjects qualify | Route discrimination or multi-step continuation is not demonstrated even if reconstruction remains finite |
| Proxy/global consistency | On first eight validation subjects by the prewritten hash order, two target-free candidate probes at Z0 get full dense before/after evaluation **after** candidate identities are fixed. Reject if proxy-positive updates have negative cohort-mean global Charbonnier change beyond 1e-5, or >25% have global harm beyond 1e-5 | Local/fibre proxy cannot support the intended reconstruction-benefit claim; implement affected-volume gain measurement next |
| Runtime | ≥4× faster counterfactual substage on identical saved states/probes, no higher peak allocation, entire diagnostic within four hours | Optimize measured hot operations before another experiment; fewer probes alone cannot count as same-work speedup |

K0 or Kmax fractions alone are **not pass/fail gates**. A cohort can correctly stop entirely if no useful updates exist. Conversely, the current 119/2 split fails as evidence of useful adaptation because action gain, policy parity and decision calibration are unestablished. Under the new protocol, a binary distribution accompanied by accurate measured decisions would falsify “binary routes are necessarily pathological.” Small probe-set false-stop rates are optimistic lower bounds on missed useful actions across all N; state this limitation in every decision summary.

## 12. Risks / falsification criteria

**Leading explanation:** a raw, clipped-gain threshold creates a subject-level entry barrier; once crossed, the different-candidate continuation test and weakly changing local descriptors allow persistent continuation even when all ranking utilities become nonpositive. The forced first training step and moving updater-derived labels conceal this in train K0. This explanation is mechanistically plausible and supported by the aggregate identities, but it is not a recovered per-subject execution trace.

It would be falsified or narrowed by:

- Server source/resolved config showing that the cited route/aggregation code or Kmax was not used. Recompute the distribution from raw per-subject records before retaining the 119/2 claim.
- Same-checkpoint traces showing selected and maximum-gain indices always coincide and selected reward always clears threshold. That contradicts the current conditional selected-reward deduction under the reviewed logging semantics and signals an attribution/aggregation discrepancy.
- Raw paired calibration showing good selected-action gain calibration, together with unchanged useful-action availability on halted subjects. Then the aggregate reward/R-star difference was population mixing rather than underestimation.
- Exact same-subject train/eval inference with forcing disabled yielding a residual route difference. Investigate remaining mode/state effects instead of assigning the whole gap to the floor.
- A trained/frozen updater having no positive signed gains on broad probes. The correct action may be to stop; route tuning cannot manufacture useful updates.
- Signed local/spill gains failing the dense-change check, or spill-repeat noise exceeding candidate gaps. Improve the measurement/footprint estimator before trying a more complex policy.
- After matched-policy fixed-budget training and separate gain fitting, winners still perform no better than random despite well-fitted candidate averages. Test descriptor aliasing (`q_bar` versus full `f_spec`), top-tail model capacity and state-conditioned calibration. Do not assume ranking loss guarantees useful actions.
- A profiler showing repeated full-volume checks/casts are negligible relative to other operations. Retain the exact call counts, revise the performance priority and speedup forecast.
- Bounded two-step evaluations revealing reliable delayed improvements after individually harmful first actions. This would weaken the myopic halt criterion and motivate option 5 with an explicitly measured finite-horizon return.

Remaining risks include unseen collateral effects of plane patches, small calibration cohorts, selection bias from max over 2,048 scores, reduced coverage in the 128-subject diagnostic, and inherited validation reuse from earlier smoke runs. This three-epoch experiment is a debugging decision gate, not independent clinical/generalization evidence. A learned stop head, uncertainty model or full RL policy would inherit these issues unless measurements and training-state coverage are repaired first.

Final document checks: all 12 required sections and relative links checked; `compileall` passed for the point-guided package, trainer and tests; `git diff --check` passed. The report's only intended repository change is this file. The supplied run evidence and existing `.DS_Store` change remain outside the commit. Commit details are recorded in the handoff accompanying the report.

| Finding | Severity | Evidence | Recommended action |
|---|---|---|---|
| Continuation and selection can use different candidates | High | Solver separate maxima; executed counterexample; active selected reward≈0.0199<0.025 | Gate the selected action or filter profitable actions before ranking |
| Validation is at the K0/Kmax extremes | High | E[K\|K>0]=64; positive utility matches initial round only | Export subject/state traces and measure actual update benefit |
| Gain target erases harm and changes scale by local error | High | `spill_aware_reward_target`, clip/divide equation | Signed absolute target, declared proxy units, dense consistency check |
| Train floor hides halt failure; terminal labels do not train updater | High | `_run` forced scores; detached descriptor/no-grad counterfactual; gradient probe | Fixed-budget updater bootstrap, then stationary reward fitting |
| Validation and public inference policies differ | High | Server wrapper omits mask; Gate-G rebuild omits corrected flags | Version and unify effective policy/config; assert parity |
| Run/source parity is unresolved | High evidence limitation | Run SHA unavailable; W&B config omits trajectory | Recover server source/config/checkpoint and split hashes |
| Gate-E repeats global work inside local probes | High operational | 4M full-volume scans, 2M mask casts/state, clones, scalar syncs | Cache validated inputs, batch/sparsify exact counterfactual queries |
| Low reward loss and global norm do not establish route learning | Medium | SmoothL1 scale, inactive ranking, missing module/decision diagnostics | Paired calibration, top-1 regret, per-module gradients |
| Logging conflates populations and hides stop/train-K evidence | Medium | Subject-mean maxima, unpaired R-star, numeric-only W&B filter | Explicit reductions, per-phase K histograms and numeric stop counts |
| No target-value inference leak found in inspected path | Satisfied within reviewed scope | Observation-first loader/context; immutable no-grad measurements | Preserve boundary and target-swap tests; verify server preprocessing separately |
