# Point-guided reward-logic smoke rerun

Date: 2026-09-01  
Branch: `main`  
Purpose: validate the corrected trajectory reward/halting logic before any 5-10/100-epoch training.

This runbook is the handoff script for the server owner. It assumes the same server paths, BraTS21 preprocessing, MedicalNet checkpoint, and reviewed split used by the A/B/C trajectory diagnostics.

Do **not** change architecture, split, seed, checkpoint, overlap weight, candidate count, `k_max`, write scale, learning rate, or normalization for this smoke run.

The corrected 4070 profile already enables:

```text
bounded travel locality       = true
separate reward-based halting = true
halt reward threshold         = 0.025
training exploration floor    = 1 step
terminal-state RewardNet sup. = true
```

Full remediation rationale: `reports/trajectory-reward-logic-remediation-2026-09-01.md`.

## 1. Pull `main` and verify the checkout

From the repository root:

```bash
cd /home/aidev/workspace/quockhanh/3DGS

# Do not overwrite local work. If this prints anything, stop and resolve it first.
git status --short

git checkout main
git pull --ff-only origin main

git rev-parse HEAD
git log -1 --oneline
```

The checkout must contain:

```text
src/smagm/features/point_guided/trajectory_cost.py
src/smagm/features/point_guided/trajectory_solver.py
src/smagm/features/point_guided/trajectory.py
src/smagm/features/point_guided/training_objective.py
tests/features/point_guided/test_reward_route_logic_fix.py
reports/trajectory-reward-logic-remediation-2026-09-01.md
```

## 2. Export the exact controlled-run environment

Use the same environment and data used by A/B/C:

```bash
export POINT_GUIDED_PYTHON=/home/aidev/miniconda3/envs/smagm-a4000/bin/python
export BRATS21_ROOT=/home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21
export OUTPUT_ROOT=/home/aidev/workspace/quockhanh/3DGS/experiments/runs
export BASELINE_SPLIT=/home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json
export MEDICALNET_CKPT=/home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth
export MEDICALNET_SHA256=afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605

export WANDB_ENTITY=khanhlq-work-hanoi-university-of-science-and-technology
export WANDB_PROJECT=smagm-point-guided

export RUN_NAME="trajectory-logic-fix-smoke-e3-$(date +%Y%m%d-%H%M%S)"

echo "RUN_NAME=$RUN_NAME"
```

Fail closed if any required input is missing:

```bash
set -euo pipefail

test -x "$POINT_GUIDED_PYTHON"
test -d "$BRATS21_ROOT"
test -f "$BASELINE_SPLIT"
test -f "$MEDICALNET_CKPT"

printf '%s  %s\n' "$(sha256sum "$MEDICALNET_CKPT" | awk '{print $1}')" "$MEDICALNET_CKPT"
```

The printed checkpoint digest must equal:

```text
afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605
```

## 3. Verify the corrected config before spending GPU time

Run:

```bash
"$POINT_GUIDED_PYTHON" - <<'PY'
import json
from pathlib import Path

p = Path("configs/training/point_guided_brats21_4070.json")
cfg = json.loads(p.read_text())
t = cfg["trajectory"]
s = cfg["supervision"]

expected = {
    "lambda_travel": 0.05,
    "lambda_overlap": 0.2,
    "lambda_step": 0.025,
    "bounded_travel_cost": True,
    "separate_halt_from_utility": True,
    "training_exploration_steps": 1,
}
for key, value in expected.items():
    assert t[key] == value, (key, t[key], value)
assert s["supervise_terminal_state"] is True

print("trajectory config OK")
print(json.dumps(t, indent=2))
print("supervise_terminal_state =", s["supervise_terminal_state"])
PY
```

Do not continue if any assertion fails.

## 4. Run focused CPU/unit regression tests first

Use the same Python interpreter as training:

```bash
PYTHONPATH=src "$POINT_GUIDED_PYTHON" -m pytest -q \
  tests/features/point_guided/test_trajectory_cost.py \
  tests/features/point_guided/test_trajectory_solver.py \
  tests/features/point_guided/test_reward_route_logic_fix.py \
  tests/features/point_guided/test_training_objective.py
```

**Gate:** all focused tests must pass. If one fails, stop here and send the first full traceback. Do not launch GPU training on a failing checkout.

## 5. Record GPU and provenance

Before launch:

```bash
nvidia-smi

echo "git_head=$(git rev-parse HEAD)"
echo "split=$BASELINE_SPLIT"
echo "split_sha256=$(sha256sum "$BASELINE_SPLIT" | awk '{print $1}')"
echo "medicalnet_sha256=$(sha256sum "$MEDICALNET_CKPT" | awk '{print $1}')"
echo "run_name=$RUN_NAME"
```

Keep this terminal output with the run handoff.

## 6. Launch exactly one 3-epoch smoke run

Use one A4000 and FP32/no-AMP so the run stays directly comparable to A/B/C:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src "$POINT_GUIDED_PYTHON" -m smagm.cli.point_guided_train \
  --config configs/training/point_guided_brats21_4070.json \
  --data-root "$BRATS21_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --split-file "$BASELINE_SPLIT" \
  --medicalnet-checkpoint "$MEDICALNET_CKPT" \
  --medicalnet-sha256 "$MEDICALNET_SHA256" \
  --no-amp \
  --epochs 3 \
  --run-name "$RUN_NAME" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$RUN_NAME"
```

Do not launch a second parameter sweep in parallel. This run validates the logic fix itself.

## 7. What must be checked in W&B

Primary route metrics:

```text
val/trajectory_fraction_K0
trajectory/mean_K_used
val/trajectory_fraction_positive_utility
val/trajectory_candidate_reward_mean
val/trajectory_candidate_reward_max
val/trajectory_r_star_positive_fraction
val/trajectory_travel_cost_mean
val/trajectory_overlap_cost_mean
val/trajectory_utility_after_cost_mean
val/trajectory_utility_after_cost_max
```

Stability metrics:

```text
train/reward_loss
train/local_loss
train/monotonic_loss
train/update_regularization
train/grad_norm
val/reconstruction_loss
val/PSNR
val/SSIM
val/MAE
```

Also inspect the route stop-reason histogram in `metrics.csv`/local logs.

## 8. Pass / investigate / fail interpretation

This is a logic-validation run, not a final quality benchmark.

### PASS candidate

Proceed to a 5-10 epoch confirmation only if all of the following are true:

- no NaN, OOM, crash, or non-finite gradient;
- `val/trajectory_fraction_K0` does **not** return to the old ~0.99 collapse regime;
- `trajectory/mean_K_used` is no longer mechanically pinned to exactly `1.0`; a meaningful amount of route execution beyond one step should be visible;
- bounded `val/trajectory_travel_cost_mean < 1.0`;
- RewardNet supervision/loss remains defined even when subjects halt;
- reconstruction loss, PSNR, SSIM, MAE, update magnitude and grad norm remain finite.

### INVESTIGATE

If K0 is solved but `mean_K_used` remains around 1, do **not** immediately lower `lambda_travel`. Report the run first. The next audit target is the raw RewardNet gain distribution versus the `0.025` halt threshold.

### FAIL

Stop and report immediately if:

- `val/trajectory_fraction_K0` returns near ~0.99;
- travel metric is not bounded below 1 despite the corrected profile;
- RewardNet loss disappears for halted/K0-heavy batches;
- any NaN/non-finite value appears;
- a regression test or runtime invariant fails.

## 9. Files to return after the run

Send these from:

```text
$OUTPUT_ROOT/$RUN_NAME/
```

Required:

```text
config.json
split.json
environment.json
metrics.csv
summary.json
train.jsonl
```

Also send:

```text
W&B run link
git rev-parse HEAD
nvidia-smi output
focused pytest output
exact RUN_NAME
```

Do not send the MedicalNet checkpoint itself.

## 10. Short teammate checklist

```text
[ ] clean checkout
[ ] git pull --ff-only origin main
[ ] same A/B/C split
[ ] same MedicalNet SHA256
[ ] corrected config assertions pass
[ ] focused pytest passes
[ ] one GPU only
[ ] --no-amp
[ ] exactly 3 epochs
[ ] W&B project = smagm-point-guided
[ ] collect six local run artifacts + W&B link
[ ] no extra tuning before review
```

The next decision is made only after reviewing this 3-epoch evidence.