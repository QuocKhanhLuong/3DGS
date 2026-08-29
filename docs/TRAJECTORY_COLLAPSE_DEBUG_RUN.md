# Trajectory collapse diagnostic runbook

Status: **engineering diagnosis / short ablation only**. This is not a final training recipe and must not be used as paper evidence until the controlled runs below are completed.

## 1. Why this run exists

The latest 10-epoch point-guided BraTS21 run learns the main reconstruction path, but the adaptive trajectory is effectively collapsing to `K=0` / no-op.

Observed W&B symptoms from the run reviewed on 2026-08-29:

- `val/trajectory_fraction_K0` stays around `0.98`.
- `train/trajectory_fraction_positive_utility` is approximately zero after the first epoch; validation is also near zero.
- `trajectory_r_star_positive_fraction` is approximately zero on train and only a small fraction on validation.
- `trajectory_utility_before_cost_mean` is positive, while `trajectory_utility_after_cost_mean` is negative.
- candidate reward mean is only on the order of `0.025--0.035` late in training, while the configured fixed step cost is `lambda_step = 0.05`.
- reconstruction metrics still improve, so the failure is localized: the model can learn the reconstruction task while mostly bypassing the point-guided trajectory.

Do **not** start a 50/100-epoch run from the same configuration yet. First establish whether the route is being economically suppressed by its cost calibration.

## 2. Code-level diagnosis

Current 4070 MAIN configuration:

```json
"trajectory": {
  "lambda_travel": 0.05,
  "lambda_overlap": 0.2,
  "lambda_step": 0.05,
  "k_max": 64,
  "selection_temperature": 1.0,
  "write_scale": 0.1,
  "support_radius_mm": 4.0
}
```

The route utility implemented in `src/smagm/features/point_guided/trajectory_cost.py` is exactly:

```text
U = reward
    - lambda_travel * travel
    - lambda_overlap * overlap
    - lambda_step
```

At the **first route decision (`K=0`)**:

- `previous_indices == -1`, therefore `travel_cost(...) == 0`;
- the overlap map is initialized to zero;
- therefore the first-step decision simplifies to:

```text
U_K0 = reward - lambda_step
```

`AdaptiveRouteSolver` executes a step only when `max_utility > 0`. Otherwise the subject stops with reason `nonpositive_utility` and receives no update.

This makes `lambda_step` the first variable to isolate. With a reward scale around `0.025--0.035` and `lambda_step = 0.05`, the current solver can rationally learn the degenerate policy "stop immediately" even though the reconstruction branch continues learning.

**Important:** do not change the architecture, reward target definition, travel cost, overlap cost, candidate count, LR, dataset split, or backbone in the first diagnostic. We need a single-variable causal test.

## 3. Phase 1: isolate the step-cost threshold

Run four short variants for **3 epochs each** using the exact same data split, seed, checkpoint, device, and all other settings.

| Run | `lambda_step` | `lambda_travel` | `lambda_overlap` | Purpose |
|---|---:|---:|---:|---|
| A / control | 0.050 | 0.050 | 0.200 | reproduce current collapse |
| B / half-step | 0.025 | 0.050 | 0.200 | place threshold near observed reward mean |
| C / low-step | 0.010 | 0.050 | 0.200 | test clear route revival |
| D / very-low-step | 0.005 | 0.050 | 0.200 | diagnostic lower bound; not a proposed final value |

Do not interpret the best 3-epoch reconstruction metric as the winner. The purpose is to determine whether the trajectory receives a usable learning signal.

## 4. Prepare diagnostic configs

From repository root, after setting the normal server variables from `docs/POINT_GUIDED_SERVER_RUN.md`:

```bash
export POINT_GUIDED_PYTHON=/path/to/gpu-env/bin/python
export BRATS21_ROOT=/path/to/BraTS2021_TrainingData
export MEDICALNET_CKPT=/path/to/resnet_10_23dataset.pth
export MEDICALNET_SHA256=<actual_sha256>
export OUTPUT_ROOT=/path/to/point_guided_runs

# Strongly recommended: reuse the split from the reviewed 10-epoch run.
export BASELINE_SPLIT=/path/to/reviewed-10epoch-run/split.json

mkdir -p "$OUTPUT_ROOT/trajectory-debug-configs"

"$POINT_GUIDED_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

base_path = Path("configs/training/point_guided_brats21_4070.json")
out_dir = Path(os.environ["OUTPUT_ROOT"]) / "trajectory-debug-configs"
out_dir.mkdir(parents=True, exist_ok=True)

variants = {
    "A-step050": 0.050,
    "B-step025": 0.025,
    "C-step010": 0.010,
    "D-step005": 0.005,
}

base = json.loads(base_path.read_text())
for name, lambda_step in variants.items():
    cfg = json.loads(json.dumps(base))
    cfg["trajectory"]["lambda_step"] = lambda_step
    cfg["training"]["epochs"] = 3
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(path, "lambda_step=", lambda_step)
PY
```

Check that only `trajectory.lambda_step` and the diagnostic epoch count differ:

```bash
for f in "$OUTPUT_ROOT"/trajectory-debug-configs/*.json; do
  echo "===== $f ====="
  "$POINT_GUIDED_PYTHON" - <<PY
import json
c=json.load(open("$f"))
print(c["trajectory"])
print("epochs=", c["training"]["epochs"])
print("seed=", c["training"]["seed"])
PY
done
```

## 5. Run the four diagnostics

Run them sequentially on the same GPU. Do not run them concurrently on one GPU because memory pressure and timing noise are irrelevant to this experiment.

```bash
export WANDB_PROJECT=smagm-point-guided

for TAG in A-step050 B-step025 C-step010 D-step005; do
  CONFIG="$OUTPUT_ROOT/trajectory-debug-configs/${TAG}.json"
  RUN_NAME="trajectory-debug-${TAG}"

  echo "===== START $RUN_NAME ====="
  echo "git HEAD: $(git rev-parse HEAD)"

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  "$POINT_GUIDED_PYTHON" -m smagm.cli.point_guided_train \
    --config "$CONFIG" \
    --data-root "$BRATS21_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda \
    --split-file "$BASELINE_SPLIT" \
    --medicalnet-checkpoint "$MEDICALNET_CKPT" \
    --medicalnet-sha256 "$MEDICALNET_SHA256" \
    --epochs 3 \
    --run-name "$RUN_NAME" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$RUN_NAME"
done
```

If a run name already exists, use a new explicit name rather than reusing the directory.

If the reviewed run used another CUDA device, keep the same device if practical. Hardware identity is less important than keeping dataset split, code SHA, pretrained checkpoint, seed, normalization, optimizer, and architecture fixed.

## 6. Metrics that must be reported

For every epoch and for both train/validation where available, collect at least:

```text
trajectory_fraction_K0
trajectory_fraction_positive_utility
trajectory_r_star_positive_fraction
trajectory_candidate_reward_mean
trajectory_candidate_reward_max
trajectory_utility_before_cost_mean
trajectory_utility_after_cost_mean
trajectory_utility_after_cost_max
trajectory_travel_cost_mean
trajectory_overlap_cost_mean
trajectory_step_cost
trajectory_r_star_mean
trajectory_r_star_max
reward_loss
local_loss
monotonic_loss
update_regularization
reconstruction_loss
MAE
PSNR
SSIM
dice_normal
dice_edema
dice_core
grad_norm
```

If a key is logged with a `train/`, `val/`, or slightly different capitalization prefix, report the existing key rather than adding a duplicate metric merely for this experiment.

The most important plot is not reconstruction loss. Compare these four curves across A/B/C/D:

```text
fraction_K0
fraction_positive_utility
r_star_positive_fraction
utility_after_cost_mean / max
```

## 7. Diagnostic gate

A variant counts as **trajectory revival** only if the change is visible on validation as well as train.

Use the current run as the reference failure mode (`fraction_K0 ~ 0.98`, positive utility near zero). Prefer relative evidence rather than declaring an arbitrary final paper threshold.

A useful signal is:

1. non-`K0` trajectories increase by several-fold relative to control;
2. positive-utility fraction increases by at least an order of magnitude from the near-zero control regime;
3. `utility_after_cost_max` becomes positive for a non-trivial subset of subjects;
4. `r_star_positive_fraction` no longer collapses immediately;
5. reconstruction/gradient metrics remain finite and do not show a new instability.

For triage, `val/trajectory_fraction_K0 < 0.90` by epoch 3 is a strong practical indication that the dead-route regime has been broken. It is a **debug gate only**, not a claimed optimal operating point.

If B (`0.025`) already clearly revives the route, prefer it over more aggressive C/D for the next experiment because it is the smallest intervention tested. Do not choose C/D merely because they produce longer trajectories.

## 8. Decision after Phase 1

### Case A: lowering `lambda_step` revives the trajectory

Conclusion: the architecture can execute a non-trivial route; the immediate failure is primarily cost/reward scale calibration.

Next action:

- select the smallest step-cost reduction that passes the diagnostic gate;
- run a 5--10 epoch confirmation with the same split;
- only then inspect route quality and reconstruction benefit;
- keep the original `0.05` run as control evidence.

Do **not** immediately tune every cost coefficient at once.

### Case B: first step revives, but route length stays around 1

Then the first-step threshold is fixed and the next bottleneck is likely later-step travel/overlap cost.

Run a second isolated matrix while holding the selected `lambda_step` fixed:

```text
travel: 0.050 control vs 0.025
and separately
overlap: 0.200 control vs 0.100
```

Do not reduce travel and overlap in the same first follow-up, otherwise causality is lost.

### Case C: even `lambda_step = 0.005` remains near `K0 ~ 0.98`

Do not train longer. Audit the learning signal instead:

- confirm RewardNet outputs and `candidate_reward_max` are actually positive;
- inspect counterfactual reward target distribution and valid counts;
- verify `r_star` construction is not almost always non-positive before route execution;
- verify RewardNet receives non-zero gradients;
- verify no mask or availability policy suppresses candidate utility;
- inspect whether the W&B `fraction_K0` mapping matches `TrajectoryResult.route_lengths == 0`;
- test a tiny overfit subject before changing architecture.

At that point the problem is no longer explained by the fixed step threshold alone.

## 9. What not to change in this diagnostic

Do not change any of the following during Phase 1:

- MedicalNet checkpoint or frozen/detached backbone policy;
- number of points (`2048`), support radius (`4 mm`), displacement bound (`2 mm`);
- reward/counterfactual candidate counts;
- semantic loss weights;
- optimizer, LR, batch size, gradient clipping;
- dataset normalization or train/val split;
- decoder architecture or write scale;
- `k_max`, selection temperature, travel cost, overlap cost.

Changing any of these invalidates the single-variable test.

## 10. Handoff report template

After all four 3-epoch runs finish, send back:

```text
Git SHA:
GPU:
MedicalNet SHA256:
Baseline split path/hash:

A-step050 W&B:
B-step025 W&B:
C-step010 W&B:
D-step005 W&B:

Final epoch / best diagnostic epoch:
              A       B       C       D
val K0 frac
val +U frac
val r* + frac
val U mean
val U max
val reward mean
val reward max
val recon loss
val PSNR
val SSIM

Any NaN/OOM/runtime error:
Notes:
```

Also attach each run's `config.json`, `summary.json`, and W&B link. Do not report only screenshots; retain the numerical artifacts so the next decision can be reproduced.

## 11. Current working hypothesis

The current failure is **not yet evidence that the trajectory architecture is incapable of learning**. The code makes immediate execution contingent on `max(reward) > lambda_step` at `K=0`, and the reviewed reward scale is already below or near the configured `0.05` step threshold. The controlled step-cost ablation above is therefore the minimum experiment required before any architecture redesign or long training run.
