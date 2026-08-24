# AGY-F — Tests, CI, Deployment, and Operational Readiness Audit

## 1. Audit metadata

- **Lane:** AGY-F from `reports/audit/01-sol-plan.md`.
- **Frozen target:** `main` at `0efeb94af72ffa067769e19afcd19ad358feefd2`.
- **Upstream observed at start:** `origin/main` at the same commit.
- **Mutation scope:** this report only. Production code, tests, configs, plans,
  and existing documentation were not modified.
- **Dirty-state rule:** the pre-existing `.DS_Store` files, architecture HTML,
  and other worker reports were preserved. No reset, stash, clean, stage,
  commit, or branch operation was performed.
- **Finding contract:** every new finding below has an ID, severity, status,
  source evidence, impact, and an unimplemented repair direction. Existing
  findings from adjacent lanes are referenced by their original ID rather than
  duplicated.

## 2. Scope and verdict

The CPU software contract is locally executable and passed the focused server,
frontend, data, training, evaluation, shell-syntax, config-parse, CLI-help, and
whitespace checks listed in Section 6. This is not F3/F4 or Gate-G execution
evidence: the current host has no CUDA device, no `nvidia-smi`, no BraTS root,
no MedicalNet checkpoint, and no NIfTI dependency. The server launch path also
has an environment-selection defect that is directly reproducible on this
host.

The main operational findings are:

| ID | Severity | Finding | Evidence status |
| --- | --- | --- | --- |
| AGY-F-FIND-001 | P2 | CI and the quality catalog do not exercise the server-ready real-data and launcher gates. | Direct workflow/catalog inspection plus local optional-test skip evidence. |
| AGY-F-FIND-002 | P1 | The server wrappers do not consistently bind probes and launches to the same Python/Torch environment. | Direct shell inspection plus a reproduced bare-interpreter Torch import failure. |
| AGY-F-FIND-003 | P2 | Evaluation output directories and prediction files are not reserved or crash-safe for repeated/concurrent runs. | Direct output-path and persistence control-flow proof; no concurrency reproduction claimed. |
| AGY-F-FIND-004 | P2 | CI dependency pins, server installation instructions, and the available local environment are not one reproducible dependency contract. | Direct CI/pyproject/docs inspection plus version/package observations. |

No scientific, reconstruction-quality, GPU, DDP, trained-checkpoint, or
held-out metric claim is made.

## 3. Operational flow audited

```text
checked-in config
  -> shell wrapper / Python CLI
  -> environment and checkpoint probe
  -> structural cohort inventory and split
  -> CPU/GPU training or target-free evaluation
  -> rank-0 logs/checkpoints or evaluation JSON/NIfTI artifacts
```

The local `scripts/codegraph.py` routers for `server_pipeline`, `quality`, and
`tests` all exited successfully. The repository has no `.codegraph/` directory,
so the MCP CodeGraph index was unavailable; the repository router and direct
scoped reads were used instead. The active task router identifies the server
entrypoints, tests, configs, scripts, and docs as the authorized audit surface.

## 4. Findings

### AGY-F-FIND-001 — P2 — CI and quality do not exercise server-ready real-data/launcher gates

**Status: current readiness gap, not a claim that the CPU workflow is broken.**

Evidence:

- `.github/workflows/ci.yml:18-25` installs pinned NumPy, pytest, and CPU-only
  PyTorch, then runs `python -m pytest -q`. It does not install the project's
  `real-data` or `wandb` extras.
- `pyproject.toml:15-22` puts `nibabel`, SciPy, scikit-image, and W&B in
  optional extras. The point-guided NIfTI tests call
  `pytest.importorskip("nibabel")` at
  `tests/data/test_brats21_point_guided.py:29-30`.
- The workflow's explicit follow-up checks (`.github/workflows/ci.yml:26-41`)
  validate the old phase catalog and legacy public CLIs. Its synthetic chain
  (`:42-59`) is the legacy static pipeline; it does not invoke
  `point_guided_preflight.sh`, a point-guided train/eval CLI with a dataset, or
  any point-guided server shell wrapper. `compileall` and a clean-tree check
  (`:60-63`) remain useful software checks but cannot supply those missing
  runtime gates.
- `quality/checklists.json:6-37` has only the
  `POINT_GUIDED_FRONTEND` checks and explicitly lists downstream selection,
  decoding, reconstruction, and training as non-claims. `quality/README.md:10-17`
  still says Gate F/G are inactive/default-deny despite the active authority's
  server-ready software status. Running the catalog locally produced
  `POINT_GUIDED_FRONTEND` with Human Gate `pending`; it does not represent F3/F4
  or G1-G4 readiness.
- The local data-adapter run passed 5 tests and skipped 19 because nibabel was
  unavailable. The broader selected run passed 54 and skipped 21; the two
  additional skips were CUDA/device-diversity and CUDA-AMP tests. Thus the
  green CPU result leaves real NIfTI and CUDA branches unexecuted.

**Impact.** A green GitHub check can coexist with an untested NIfTI adapter,
MedicalNet checkpoint path, shell environment, CUDA AMP path, and real
checkpoint-to-held-out evaluation. Operators may infer more F/G readiness from
the generic `pytest -q` result than the workflow actually establishes.

**Repair direction (not applied).** Add a separately identified server smoke
job or preflight contract that installs the real-data dependencies, validates
the selected environment/checkpoint, runs synthetic CLI/config/wrapper checks,
and clearly marks real BraTS/GPU/DDP execution as server-only. Update the
quality catalog/status prose only under an explicitly authorized documentation
change.

### AGY-F-FIND-002 — P1 — Launch wrappers do not bind all commands to one Python/Torch environment

**Status: current shell-wrapper defect; server impact remains conditional on the
target host environment.**

Evidence:

- `scripts/point_guided_train_4070.sh:8,37-54` is the only main launcher that
  exposes and uses `POINT_GUIDED_PYTHON` consistently for its probe and Python
  module invocation.
- `scripts/point_guided_preflight.sh:16-22,28-34` uses bare `python` for both
  the Torch probe and the preflight CLI. It does not honor
  `POINT_GUIDED_PYTHON`.
- `scripts/point_guided_train_2xa4000.sh:16-20,26-35` uses bare `python` for
  its version probe and bare `torchrun` for the DDP launch. The overfit and
  evaluation wrappers likewise use bare `python` in their probes and launches
  (`scripts/point_guided_overfit_4070.sh:17-23,29-40` and
  `scripts/point_guided_eval.sh:27-33,39-48`).
- On this host, bare `python` is `/Users/alvinluong/miniforge3/bin/python`
  (3.12), the working project interpreter is
  `/Users/alvinluong/3DGS/.venv/bin/python` (3.10), and bare `torchrun` has the
  shebang `#!/usr/local/bin/python3.13`. The correct `.venv` imports CPU Torch
  2.13.0. The bare Python Torch probe failed before the wrapper could run:
  `OSError: .../miniforge3/.../torch/lib/libtorch_global_deps.dylib ... no such
  file`.
- Shell syntax itself passed, and direct CLI help passed when invoked through
  the working `.venv`; those checks do not repair or validate the wrappers'
  interpreter selection.

**Impact.** On a host where the project environment is valid but the global
`python`/`torchrun` is absent, broken, or built against another Torch/CUDA
stack, preflight and the 2×A4000/overfit/evaluation scripts can fail during the
probe or launch. A DDP job can also use a different `torchrun` than the Python
that owns the installed project dependencies. This blocks reliable server
operations before any model/data evidence exists.

**Repair direction (not applied).** Define one explicit interpreter contract
for every wrapper, use it for version probes and module/torchrun execution, and
fail closed if the selected interpreter cannot import the required package.
The target server's actual environment still needs an independent preflight.

### AGY-F-FIND-003 — P2 — Evaluation output persistence is not run-isolated or crash-safe

**Status: static operational risk; no concurrent evaluation reproduction is
claimed.**

Evidence:

- `scripts/point_guided_eval.sh:14-22` chooses a default output directory using
  a UTC timestamp with one-second resolution and then calls `mkdir -p`; it does
  not reserve a new directory or reject a non-empty existing directory.
- `src/smagm/cli/point_guided_eval.py:28-33` writes every JSON artifact through
  the fixed sibling name `.<filename>.tmp` followed by `replace`. The evaluator
  writes four shared artifacts at `:225-238`, so concurrent calls to one output
  directory can race on both the temporary names and final last-writer-wins
  payloads.
- Predictions are written directly with `nib.save` at
  `src/smagm/cli/point_guided_eval.py:79-87,201-205`; there is no temporary
  prediction file plus atomic rename.
- The scoped test inventory covers split/checkpoint semantics and sequential
  result assembly, but contains no test for concurrent evaluation, a reused
  output directory, interrupted JSON writes, or interrupted NIfTI writes.

**Impact.** Repeated or simultaneous evaluation can mix metadata, metrics,
trajectory diagnostics, and predictions from different checkpoints/splits, or
leave a directly written NIfTI artifact incomplete after interruption. The
strict checkpoint and split checks do not provide output-directory isolation.

**Repair direction (not applied).** Reserve a unique evaluation run directory
or require an empty destination, write all artifacts to unique temporary files
with collision-safe names, atomically rename predictions, and add failure/
collision tests. Existing training run-collision and logger-cleanup findings
remain owned by AGY-D-FIND-001/002/006; this finding is limited to evaluation
outputs.

### AGY-F-FIND-004 — P2 — CI, server install, and local dependencies are not one reproducible contract

**Status: deployment reproducibility gap; no runtime model failure attributed to
the version difference.**

Evidence:

- CI pins `numpy==2.2.6`, `pytest==9.1.1`, and CPU `torch==2.12.1` in
  `.github/workflows/ci.yml:18-23`.
- The project metadata allows broad ranges (`numpy>=1.26,<3` and
  `torch>=2.1,<3`) and leaves real-data/W&B support optional
  (`pyproject.toml:10-22`). The server guide instructs an unpinned editable
  install of `.[test,real-data,wandb]` at `docs/POINT_GUIDED_SERVER_RUN.md:15-24`.
- The available `.venv` is Python 3.10 with Torch 2.13.0 CPU and pytest 9.1.1,
  but has no nibabel, SciPy, scikit-image, or W&B. CUDA is unavailable.

**Impact.** The tested CPU dependency set is not the documented server
dependency set, and the broad server ranges permit a different Torch/CUDA or
NIfTI stack than the one exercised by CI. This weakens reproducibility and can
hide install-time/API drift until the server run.

**Repair direction (not applied).** Publish a server-tested environment lock or
fully recorded package manifest, keep the CPU contract separately identified,
and make the preflight record/compare the versions required by the selected
profile. No dependency installation or lockfile change was authorized here.

## 5. Coverage and operational assumptions

The tests provide useful software evidence:

- `test_point_guided_server_pipeline.py` exercises checkpoint/config/split
  helpers, sampler behavior, and CPU trainer paths, but its distributed checks
  use synthetic `DistributedContext`/sampler objects rather than launching two
  processes or an NCCL/Gloo process group.
- Frontend boundary/forward, data, baseline training, and baseline inference
  tests cover deterministic and target-boundary contracts. The data tests that
  require NIfTI are skipped when nibabel is absent.
- There are no repository tests for shell-wrapper environment selection,
  actual CLI preflight against a NIfTI cohort, CUDA AMP, NCCL/DDP rank failure,
  W&B lifecycle on exceptions, disk-full/quota behavior, output collision, or
  partial evaluation artifact recovery.
- Existing adjacent findings are not duplicated here: AGY-D-FIND-001/002/003/
  004/005/006/007 cover DDP failure cleanup, training run collisions, resume
  RNG/protocol/padding, logger cleanup, and the server-profile parameter-count
  mismatch; AGY-E-FIND-001/003 cover evaluation metadata and no partial-run
  resume; AGY-G-FIND-001/002/003 cover split-hash binding, malformed-directory
  inventory, and resumed CSV schema; AGY-A covers stale public documentation.

## 6. Verification performed

All commands below ran from `/Users/alvinluong/3DGS` without writing test
caches (`PYTHONDONTWRITEBYTECODE=1`, `-B`, and pytest's cache plugin disabled
where applicable):

| Check | Result |
| --- | --- |
| `python scripts/codegraph.py --task server_pipeline` | PASS; scoped router printed successfully. |
| `python scripts/codegraph.py --task quality` | PASS; scoped router printed successfully. |
| `python scripts/codegraph.py --task tests` | PASS; scoped router printed successfully. |
| `PYTHONPATH=src .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/features/point_guided/test_point_guided_server_pipeline.py --tb=short` | **14 passed in 188.03s**. |
| Selected frontend/data/training/evaluation tests (`test_frontend_forward.py`, `test_frontend_boundaries.py`, `test_brats21_point_guided.py`, `test_baseline_training.py`, `test_baseline_inference.py`) | **54 passed, 21 skipped in 742.12s**. |
| `test_brats21_point_guided.py -rs` | **5 passed, 19 skipped in 52.73s**; all 19 skips were `nibabel` import skips. |
| `scripts/check_phase.py --list` | PASS; one `POINT_GUIDED_FRONTEND` catalog entry, Human Gate pending. |
| `scripts/check_phase.py POINT_GUIDED_FRONTEND` | PASS as a status read; checks were `NOT_RUN` because the tree was dirty and `--run` was not requested. |
| `bash -n` over all five `scripts/point_guided_*.sh` wrappers | PASS. |
| JSON parsing of the three point-guided training configs plus point-guided evaluation config | PASS; 4 configs parsed. |
| Point-guided train/eval `--help` through `.venv` | PASS; both CLIs printed usage and exited 0. |
| `git diff --check` | PASS. |

The task handoff reports that the predecessor also completed the relevant
compileall and broader quality/test checks before quota. Compileall was not
rerun in this continuation because it writes bytecode artifacts, while this
lane's mutation scope is limited to the report. The direct tests above are the
current continuation's executed software evidence.

## 7. Environment and storage evidence

- `HEAD`: `0efeb94af72ffa067769e19afcd19ad358feefd2`; branch `main`.
- `.venv`: Python 3.10.0, Torch 2.13.0, CPU-only (`torch.cuda.is_available()`
  false), pytest 9.1.1.
- Optional/runtime packages: nibabel, SciPy, scikit-image, and W&B absent.
- `nvidia-smi` was not available. `BRATS21_ROOT`, `MEDICALNET_CKPT`,
  `MEDICALNET_SHA256`, `OUTPUT_ROOT`, `POINT_GUIDED_PYTHON`,
  `CUDA_VISIBLE_DEVICES`, and `WANDB_PROJECT` were unset.
- The filesystem had approximately 13 GiB available and the checkout was
  approximately 684 MiB. No data or checkpoint storage, filesystem quota,
  network logger, or GPU memory behavior was exercised.
- The local host's generic `torchrun` help command succeeded, but that binary
  is bound to Python 3.13 rather than the project `.venv`; this is part of
  AGY-F-FIND-002, not DDP evidence.

## 8. Unverified gates and final assessment

Not executed or not available locally:

- real BraTS21 NIfTI discovery/loading, affine/spacing payloads, and full
  train/validation/test cohort execution;
- MedicalNet checkpoint loading against a real supplied digest;
- CUDA AMP, RTX 4070 overfit, single-GPU MAIN training, two-process A4000
  NCCL training, rank failure/recovery, GPU memory, throughput, and OOM
  behavior;
- trained-checkpoint Gate-G inference, held-out metrics, prediction NIfTI
  roundtrip, and artifact inspection from a real run;
- W&B/network logging, disk-full/quota behavior, and long-lived process
  cleanup.

The local result is **software-contract evidence only**. The server pipeline is
not eligible for a trained-checkpoint or scientific-quality conclusion until a
target server supplies the missing environment/data/checkpoint and resolves or
explicitly accepts the launcher and artifact-isolation findings. No production,
test, config, or documentation remediation was performed.

