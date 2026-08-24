# AGY-G recent regression audit

## Scope and verdict

- Lane: AGY-G from `reports/audit/01-sol-plan.md`.
- Frozen repository target: `main` at `0efeb94af72ffa067769e19afcd19ad358feefd2` (also the current `origin/main`).
- Audited range: `6f754ed..HEAD`, with focused review of `bc4f6bf`, `fe9fc03`, `11ba203`, `9a33058`, and `39a39d3`.
- No production, test, configuration, plan, or documentation file was modified. The only new file from this audit is this report.

The `39a39d3` reward revert is coherent at the frozen HEAD: current Gate-C uses the locked 126-dimensional, target-free descriptor, and the scoped point-guided tests pass. The recent server hardening is not fully fail-closed, however: both training and evaluation accept an arbitrary 64-character `split_hash` without recomputing it, and the clean inference checkpoint metadata does not bind a checkpoint to a split. A separate structural-inventory regression silently drops malformed subject directories, while metrics CSV schema compatibility on resume is a static operational risk; no real BraTS/NIfTI, GPU, DDP, trained-checkpoint, or held-out result was available in this audit.

## Commit classification

The range contains 77 changed files and 19,912 insertions from the relative merge base. The table records the old/current contract, current callers, available tests, and classification requested by the lane.

| Commit | Old contract -> current contract | Current callers / seams | Evidence and classification |
|---|---|---|---|
| `bc6d2e6` | Phases 1-5 point-guided frontend implementation established the target-free `B -> A -> f_spec` path. | Frontend model and point-guided feature modules. | Covered by the scoped frontend suite. Baseline implementation; no new regression isolated in this lane. **CLEAN / BASELINE**. |
| `965f29d`, `7b4df61`, `f222589` | Reconciled authority and added Gate-C/D/E and Gate-F/G plans/run documentation; no runtime contract intended. | `PLAN*.md`, `AGENTS.md`, `CODEBASE.md`, docs and run plans. | No production caller. **DOC-ONLY**; the frozen root plan remains the authority for this audit. |
| `bc4f6bf` | Completed additive Gate-C through Gate-G software, including trajectory, decoder, supervision, baseline ownership, and diagnostics. | `features/point_guided/{trajectory,decoder,reward,updater}.py`, baseline train/inference modules, tests. | 50 files, +11,745/-281. Scoped software tests pass; server/GPU/data execution is unavailable. **CLEAN SOFTWARE / RUNTIME UNVERIFIED**. |
| `fe9fc03` | Added server-ready BraTS data, deterministic split, training, evaluation, configs, and scripts around the target-free inference boundary and target-after-inference objective. | `src/smagm/data/brats21_point_guided.py`, `training/point_guided.py`, `cli/point_guided_{train,eval}.py`, configs/scripts. | 33 files, +5,204/-72. The new split/inventory seams are exercised by CPU tests, but real NIfTI and server execution remain unverified. **RISK: provenance/runtime evidence incomplete**. |
| `11ba203` | Hardened server runtime with exact split-file reuse, robust normalization checks, structural/active inventories, DDP raw-module validation, timing/finite guards, and expanded artifacts. | `training/point_guided.py`, `cli/point_guided_eval.py`, `data/brats21_point_guided.py`, configs and server-run docs. | 13 files, +691/-71. The split hash and structural inventory changes contain findings AGY-G-001 and AGY-G-002 below. **DEFECTIVE AT TWO SEAMS; OTHER HARDENING SOFTWARE-CLEAN**. |
| `9a33058` | Temporarily changed Gate-C from the 126-d state/semantic/Gate-B descriptor to a 222-d candidate-conditioned descriptor and added paired candidate updates; baseline checkpoint schema moved v1 -> v2. | `reward.py`, `updater.py`, `trajectory.py`, baseline checkpoint metadata/loaders, training configs/tests. | 29 files, +2,455/-218. This was a transient contract that did not survive the next commit. **HISTORICAL RISK, RESOLVED BY HEAD**. |
| `39a39d3` | Reverted the failed paired candidate gate: removed candidate updater/`forward_candidates`, restored scalar post-selection update, 126-d RewardNet input, and baseline checkpoint schema v1. | Current `reward.py` lines 16-18, 130-173; `trajectory.py` lines 313-332 and 399-441; `updater.py` lines 14-75; `baseline_inference.py` lines 30, 269-304. | 10 files, +25/-204. Focused tests and full scoped suite pass; no stale production symbol remains. **CLEAN AT FROZEN HEAD**. |

## Findings

### AGY-G-001 — P1 — Split hashes are accepted as opaque labels and are not checkpoint-bound

**Status: current-HEAD defect.** The exact-split hardening introduced by `11ba203` does not actually verify the split digest.

Evidence:

- Evaluation `_load_split()` validates the partition and then only checks `isinstance(split_hash, str)` and `len(split_hash) == 64` (`src/smagm/cli/point_guided_eval.py:51-76`). It returns the supplied value without canonicalizing the subject lists or recomputing a digest.
- Training `_resolve_split()` repeats the same length-only check for an external split (`src/smagm/training/point_guided.py:1073-1107`). The value is then persisted as the run's split hash by the training artifact path (`:1334-1335`) and later compared only as a string when a resume checkpoint is loaded (`:1380-1387`).
- Evaluation resolves the checkpoint-adjacent or explicitly supplied split and records the supplied hash in `evaluation_metadata.json` (`src/smagm/cli/point_guided_eval.py:124-127, 228-238`).
- `baseline_checkpoint_metadata()` contains model, trajectory, decoder, and Gate-E architecture metadata but no split hash (`src/smagm/features/point_guided/baseline_inference.py:269-282`). The strict checkpoint loader therefore cannot reject a checkpoint evaluated against a different split file (`:285-304`).
- The existing regression test itself writes `"split_hash": "a" * 64` and expects `_load_split()` to accept it (`tests/features/point_guided/test_point_guided_server_pipeline.py:391-413`). I reproduced that acceptance with a complete three-subject partition; the returned hash was the arbitrary 64-character string.

Impact: changing split membership while leaving the stored string unchanged is not detected, and a clean inference checkpoint has no expected split hash to compare against even when the split file is adjacent to the checkpoint. This defeats the stated “exact split reuse” provenance contract and can make held-out metrics appear tied to a cohort that was not actually used; it is a data-integrity/provenance defect rather than a cryptographic-authentication claim.

Recommended repair direction (not applied in this audit): define one canonical split payload/hash function, recompute and compare the digest in both training and evaluation, reject non-hex or mismatched values, and persist the expected split hash in the checkpoint metadata or another immutable training-run binding that evaluation must validate.

### AGY-G-002 — P2 — Structural pre-split inventory silently omits malformed subject directories

**Status: current-HEAD defect.** The structural inventory added in the server pipeline filters directory names before it constructs the discovered or excluded sets.

Evidence:

- `structural_inventory_point_guided_subjects()` builds `directories` only from names matching `BRATS21_POINT_GUIDED_SUBJECT_PATTERN` (`src/smagm/data/brats21_point_guided.py:729-733`), then sets `discovered_subject_ids` from that filtered tuple (`:736`) and only records missing/empty required files for matching directories (`:739-755`). A malformed directory can therefore never reach `excluded_subjects`.
- The older `inventory_point_guided_subjects()` follows the opposite, explicit policy: it iterates all directories and records malformed names as `classification="OTHER_INVALID"` (`src/smagm/data/brats21_point_guided.py:838-867`). The direct discovery path also rejects malformed IDs rather than silently omitting them.
- I reproduced the new structural path with one valid-named directory containing non-empty modality placeholders and one malformed directory. The result was `discovered=('BraTS2021_00000',)`, `eligible=('BraTS2021_00000',)`, `excluded=()`: the malformed directory was absent from the inventory.

Impact: a typoed or partially copied subject directory disappears from the cohort accounting before deterministic splitting. The resulting split can partition every *discovered* ID while hiding a source-level invalid subject from `structural_inventory.json`, making audit/recovery and cohort counts less trustworthy. This is distinct from payload-invalid matching directories, which are explicitly excluded.

Recommended repair direction (not applied): enumerate all immediate directories first, emit malformed names as explicit `OTHER_INVALID` structural exclusions, and retain the existing regex-matched path for modality metadata checks.

### AGY-G-003 — P2 risk — Resume can append a changed metrics schema under an old CSV header

**Status: static operational risk; no failing runtime test claimed.** `_write_epoch_logs()` derives CSV field names from each row and does not inspect an existing header (`src/smagm/training/point_guided.py:1177-1187`). `run_training()` treats any pre-existing `metrics.csv` as having a valid header (`:1406-1409`) and then appends the expanded current row (`:1440-1515, 1544-1545`).

If a run created before the timing/trajectory columns from `11ba203` is resumed, `header_written=True` suppresses a new header, while `csv.DictWriter` writes the current field order. The file can then contain rows with different column layouts without a schema error. This is an operator-artifact risk, not evidence that the training math is incorrect; no existing test covers resuming with a prior CSV header. Recommended repair direction (not applied): read and compare the existing header before appending, fail closed on mismatch, or version/migrate the CSV schema.

## Clean seams and unverified gates

The reward revert was checked directly at current HEAD. `REWARD_DESCRIPTOR_CHANNELS` is `96 + 3 + 24 + 3 = 126`, `build_reward_descriptor()` concatenates dynamic state, point semantic, reliability-weighted Gate-B descriptor, and reliability (`src/smagm/features/point_guided/reward.py:16-18, 130-154`), and `RewardNet` is the locked `126 -> 64 -> 1 -> sigmoid` (`:157-173`). Current trajectory scoring is dense over candidates (`trajectory.py:313-332`), availability masks utility only after scoring (`:363-376`), and `UpdateNet` is called only after selection with the 270-d selected input (`:399-441`; `updater.py:14-67`). No `CandidateCorrections`, `forward_candidates`, 222-d descriptor, or v2 checkpoint schema symbol remains in the production path.

The exact split-file existence check, normalization-space validation, raw-module DDP validation seam, finite/timing guards, and checkpoint state-dict strictness are software-level clean based on source inspection and the tests below. They do not establish correct behavior on real NIfTI geometry, CUDA kernels, multi-process DDP, trained checkpoints, or held-out cohorts. Evaluation metadata still writes `git_head: None` (`src/smagm/cli/point_guided_eval.py:228-238`); that is recorded here as an existing provenance gap and is not duplicated as a new AGY-G finding.

## Verification performed

Executed from `/Users/alvinluong/3DGS` with the frozen worktree:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/features/point_guided/test_reward.py \
  tests/features/point_guided/test_trajectory.py \
  tests/features/point_guided/test_updater.py \
  tests/features/point_guided/test_baseline_inference.py \
  tests/features/point_guided/test_baseline_training.py
24 passed, 2 skipped in 78.45s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/features/point_guided tests/data/test_brats21_point_guided.py tests/test_codegraph.py
299 passed, 37 skipped in 1657.81s (0:27:37)
```

The skipped tests include optional NIfTI-dependent coverage in this environment. No production fix was attempted, and no claim is made for a real BraTS run, GPU execution, DDP execution, trained checkpoint, or held-out evaluation.
