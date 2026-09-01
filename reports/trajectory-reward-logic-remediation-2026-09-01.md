# Point-guided trajectory reward logic remediation

Date: 2026-09-01  
Branch: `main`  
Status: **code remediated; requires short controlled rerun before any long training**

## 1. Why this remediation was needed

The 3-epoch A/B/C diagnostic runs isolated two different failure modes in the adaptive point route.

| Run | `lambda_step` | val `fraction_K0` | val mean K | val candidate reward max | val utility-after-cost mean |
|---|---:|---:|---:|---:|---:|
| A | 0.050 | 0.9917 | ~0.008 | 0.0433 | -0.0235 |
| B | 0.025 | 0.0000 | 1.0 | 0.0726 | -0.5548 |
| C | 0.010 | 0.0000 | 1.0 | 0.0790 | -0.5344 |

Run A proved that the original first-step condition could self-stop because the best expected reward was below the fixed step expense. Runs B/C then proved the architecture could execute a transition, but the route collapsed at exactly one step.

The deeper code audit showed that this was not only a hyperparameter problem. Four logic issues were coupled:

1. RewardNet produces a bounded score in `[0,1]`, while physical travel had been `distance_mm / 4mm` with no upper bound.
2. The same scalar `reward - travel - overlap - step` both ranked candidates and decided whether the route should stop.
3. RewardNet supervision existed only for states that had already executed a transition; a K=0 route therefore produced no counterfactual RewardNet supervision.
4. A cold RewardNet/UpdateNet pair could therefore create a self-locking loop: low reward prediction -> stop -> no transition -> weak/no reward/update learning -> continue stopping.

## 2. Literature/code references used for the redesign

The remediation follows the structure of active reconstruction/acquisition literature rather than copying any one paper verbatim.

- **Isler et al., ICRA 2016, _An Information Gain Formulation for Active Volumetric 3D Reconstruction_**: active-view utility combines information gain with movement cost. The released implementation normalizes gain/cost before combining them rather than subtracting an unbounded movement quantity from a small gain score. The public code is in `uzh-rpg/rpg_ig_active_reconstruction`.
- **Zhang et al., CVPR 2019, _Reducing Uncertainty in Undersampled MRI Reconstruction With Active Acquisition_**: dynamically selects measurements and iteratively refines reconstruction to reduce uncertainty/error.
- **Pineda et al., MICCAI 2020, _Active MR k-space Sampling with Reinforcement Learning_**: formulates active MRI acquisition as a sequential decision process; reward is tied to reconstruction improvement and the episode has a separate acquisition horizon/budget.

The key design lesson for this repository is: **candidate ranking, physical/locality preference, and stopping should not be collapsed into one uncalibrated scalar.**

## 3. Changes now on `main`

### 3.1 Bounded physical travel score

File: `src/smagm/features/point_guided/trajectory_cost.py`

A new opt-in mode maps the support-radius-normalized distance

```text
d = distance_mm / 4mm
```

to

```text
d_bounded = d / (1 + d)
```

so travel is in `[0,1)` while preserving distance ordering.

This prevents a 50-100 mm separation from producing a penalty one or two orders of magnitude larger than the RewardNet score. Physical route length is still reported independently in millimetres through `path_length_mm`; only the locality cost used for ranking is bounded.

New config flag:

```json
"bounded_travel_cost": true
```

Legacy callers default to the historical unbounded form.

### 3.2 Candidate ranking and halting are separated

Files:

- `src/smagm/features/point_guided/trajectory_cost.py`
- `src/smagm/features/point_guided/trajectory_solver.py`
- `src/smagm/features/point_guided/trajectory.py`

Corrected candidate ranking is:

```text
score_i = reward_i
          - lambda_travel * bounded_travel_i
          - lambda_overlap * overlap_i
```

The constant step term is no longer subtracted from every candidate in corrected mode.

Halting instead asks a different question:

```text
continue if max(raw RewardNet gain over eligible candidates) > threshold
```

For serialized-config compatibility, the existing `lambda_step` field is temporarily retained as that raw gain threshold when:

```json
"separate_halt_from_utility": true
```

The checked-in 4070 profile now uses:

```json
"lambda_step": 0.025,
"bounded_travel_cost": true,
"separate_halt_from_utility": true
```

`0.025` is the smallest tested step value that clearly broke K0 collapse in Run B. It is now interpreted as the halt gain threshold in the corrected profile, not as a ranking penalty.

The persisted stop-reason string remains `nonpositive_utility` for downstream schema compatibility. In corrected mode its semantics are now effectively **low expected gain**; rename/migration can happen after the validation rerun.

### 3.3 Minimal training exploration floor

File: `src/smagm/features/point_guided/trajectory.py`

The corrected 4070 profile sets:

```json
"training_exploration_steps": 1
```

During training only, the first transition is therefore allowed even if RewardNet has not yet calibrated its halt score. Validation/inference do not force a transition.

This is intentionally conservative: one exploratory transition prevents UpdateNet starvation without converting the adaptive route into a fixed-K training policy.

### 3.4 Terminal-state RewardNet supervision

File: `src/smagm/features/point_guided/training_objective.py`

Previously, RewardNet counterfactual supervision was generated only for each executed state's `state_before`. The final reached state was never supervised. For K=0 that meant zero executed states and therefore no reward supervision.

The new opt-in flag:

```json
"supervise_terminal_state": true
```

adds one counterfactual RewardNet supervision probe on the final reached state, including the K=0 case.

Candidate selection for this probe remains target-free:

1. query the final dynamic state;
2. run RewardNet;
3. take RewardNet argmax;
4. only after that fixed selection, introduce T1ce inside the existing counterfactual measurement path.

This preserves the target-free inference boundary while removing the zero-supervision deadlock.

### 3.5 Backward compatibility

The logic changes are opt-in. Default `TrajectoryConfig` behavior stays historical unless the new flags are enabled, so existing Gate-F/G tests and inference configurations are not silently redefined.

The main 4070 training profile explicitly enables the corrected path.

## 4. Checked-in 4070 profile after remediation

Relevant trajectory block:

```json
{
  "lambda_travel": 0.05,
  "lambda_overlap": 0.2,
  "lambda_step": 0.025,
  "k_max": 64,
  "selection_temperature": 1.0,
  "write_scale": 0.1,
  "support_radius_mm": 4.0,
  "bounded_travel_cost": true,
  "separate_halt_from_utility": true,
  "training_exploration_steps": 1
}
```

Relevant supervision addition:

```json
"supervise_terminal_state": true
```

No architecture, MedicalNet checkpoint policy, reconstruction loss, semantic loss, overlap weight, candidate count, `k_max`, write scale, or learning rate was changed in this remediation.

`reward_ranking_weight` remains `0.0`; pairwise ranking is left for a later controlled ablation rather than mixed into this logic fix.

## 5. Regression coverage

Added:

`tests/features/point_guided/test_reward_route_logic_fix.py`

It covers:

1. bounded travel is in `[0,1)` and preserves the expected 4-mm/8-mm ordering;
2. corrected route ranking no longer subtracts the halt threshold;
3. the solver can continue when ranking utility is negative but raw expected gain is above the halt threshold;
4. a K=0 training context can still produce terminal-state RewardNet supervision.

No CI status was available from GitHub at the time this report was written, so the server owner must run the focused tests before launching the GPU diagnostic.

## 6. Required validation before any 10/100-epoch run

Do **one 3-epoch controlled rerun first**, using the exact reviewed split and MedicalNet checkpoint from A/B/C.

Recommended command shape:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
"$POINT_GUIDED_PYTHON" -m smagm.cli.point_guided_train \
  --config configs/training/point_guided_brats21_4070.json \
  --data-root "$BRATS21_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --split-file "$BASELINE_SPLIT" \
  --medicalnet-checkpoint "$MEDICALNET_CKPT" \
  --medicalnet-sha256 "$MEDICALNET_SHA256" \
  --no-amp \
  --epochs 3 \
  --run-name trajectory-logic-fix-smoke \
  --wandb \
  --wandb-project smagm-point-guided \
  --wandb-run-name trajectory-logic-fix-smoke
```

Run focused tests first:

```bash
pytest -q \
  tests/features/point_guided/test_trajectory_cost.py \
  tests/features/point_guided/test_trajectory_solver.py \
  tests/features/point_guided/test_reward_route_logic_fix.py \
  tests/features/point_guided/test_training_objective.py
```

## 7. Validation gate

Do not optimize PSNR first. The logic fix passes only if the route itself behaves coherently.

Required checks:

- `val/trajectory_fraction_K0` must not return to the ~0.99 failure regime.
- `trajectory/mean_K_used` should no longer be mechanically pinned to exactly 1 by unbounded travel. `>1` for a meaningful validation subset is the key target, but the exact final route length is not predetermined.
- bounded `trajectory_travel_cost_mean` should now remain below 1 by construction when the corrected profile is active.
- `trajectory_candidate_reward_mean/max` and halt behavior must remain on the same raw reward scale.
- RewardNet loss must remain defined even when some subjects halt immediately, because the terminal state is now supervised.
- reconstruction loss, PSNR, SSIM, gradient norm and update magnitude must remain finite.
- no NaN/OOM/runtime regression.

If K is still almost always 1 after this change, inspect the reward threshold and RewardNet target distribution before touching travel again. The old `lambda_travel = 0.05 -> 0.005 -> ...` sweep is no longer the first action because the travel term is now bounded.

## 8. Known follow-ups, deliberately not mixed into this patch

1. Rename the legacy serialized `lambda_step` field to an explicit `halt_reward_threshold` after backward-compatible config migration.
2. Rename stop reason `nonpositive_utility` to `low_expected_gain` after downstream schema migration.
3. Log the full trajectory config into W&B config; current W&B setup logs training/supervision but not the whole trajectory block.
4. Evaluate `reward_ranking_weight > 0` only after corrected route dynamics are stable.
5. If one forced training transition is still insufficient to bootstrap later states, add an epoch-scheduled exploration warm-up rather than permanently increasing forced K.
6. Only after a stable 3-epoch route diagnostic should a 5-10 epoch confirmation be launched.

## 9. Commit chain

- `b61792f1aa3eec6efc0548711cdf9a885bd3c66d` — bounded travel/config semantics
- `cfb9adf2a2205682ea28a70321a5f8f67298b4bf` — separate solver halting
- `80c2753c33a87032bf3dd7133faa293b18af3ace` — corrected trajectory execution
- `e934339d43c43c261f83abd842c1832dd2f51ed1` — terminal-state RewardNet supervision
- `8988d352cff9c6e3fb6d67ec25def90b2aa80a1a` — corrected 4070 profile
- `3a9b773ea91fe211048f41d71689a6d5dc610b48` — regression tests

## 10. Current conclusion

The A/B/C runs were useful because they exposed the failure mechanism, but the final remediation should not be described as simply "lowering travel cost." The original formulation mixed incompatible scales and coupled candidate choice, stopping, and availability of the learning signal.

The corrected path now has a clearer scientific interpretation:

```text
RewardNet = expected reconstruction gain
bounded travel = locality preference
bounded overlap = redundancy preference
ranking score = gain - locality/redundancy penalties
halt = raw expected gain below threshold
training exploration = bootstrap only
terminal counterfactual = keep RewardNet supervised at stopping states
```

This is the version that should be validated next.
