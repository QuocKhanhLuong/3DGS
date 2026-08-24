# AGY-D Audit Report: Training, DDP, Multi-Process Concurrency, and Resource Lifecycle

- **Audit target:** `main` at frozen HEAD `0efeb94af72ffa067769e19afcd19ad358feefd2`
- **Audit lane:** AGY-D (training, DDP, multi-process concurrency, and resource lifecycle)
- **Codegraph tasks started:** `baseline_training`, `server_pipeline`
- **Mutation boundary:** only this report was written; production code, tests, configs, plans, and server scripts were not modified.
- **Runtime boundary:** the audit environment has no BraTS21 payload or CUDA multi-process run, so DDP/AMP conclusions are code and CPU-test evidence, not GPU/server evidence.

## 1. Executive summary

The frozen training path has a clear module-level optimizer boundary and a coherent normal-path DDP state machine. `resolve_parameter_ownership()` and `build_baseline_optimizer()` cover exactly the eight named authorized trainable modules and keep the 14,399,424-parameter MedicalNet backbone out of the optimizer, while the fixed SWT-Haar buffers remain non-trainable. The focused CPU model uses `offset_hidden_channels=12` and therefore has 69,723 trainable parameters; the checked-in server profiles use `offset_hidden_channels=128` and have 83,411 (AGY-D-FIND-007). Training uses a DDP wrapper only for the target-free context call, a raw module for uneven validation shards, explicit AMP guards, and a rank-0 artifact writer.

The normal path is not concurrency-safe for independent jobs or failure recovery. There is no run-directory reservation, the default timestamp has one-second resolution, explicit run names are reused by the supplied overfit script, and the JSON temporary name is fixed per destination; concurrent jobs can overwrite artifacts or raise a temporary-file `FileNotFoundError`. In DDP, an exception on one rank can leave the other ranks in a collective while every failing rank enters the unconditional teardown barrier, so the advertised failure cleanup can itself deadlock.

Resume is strict about schema and split hash, but not about the saved training protocol or per-rank RNG state. Only rank 0 saves Python/NumPy/Torch/CUDA RNG state and every rank restores that same state; current sampler and Gate-E generator choices bound the immediate effect, but this is not a complete rank-reproducible resume contract. Resume also resets early-stopping patience and overwrites the run's `config.json` with the current invocation without comparing the checkpoint's saved training settings.

### Findings at a glance

| ID | Severity | Finding | Evidence status |
| --- | --- | --- | --- |
| AGY-D-FIND-001 | P1 | DDP failure teardown can deadlock at an unconditional barrier after rank-divergent failure. | Direct control-flow proof; no two-process GPU reproduction available. |
| AGY-D-FIND-002 | P1 | Run directories and artifacts are not reserved against simultaneous jobs; same-name/same-second jobs collide, and fixed JSON temp names race. | Direct code proof plus temporary threaded reproduction. |
| AGY-D-FIND-003 | P2 | Rank-0-only resume RNG state is restored identically on every rank, collapsing rank-specific streams. | Direct code proof; immediate main-path impact is bounded by explicit generators/samplers. |
| AGY-D-FIND-004 | P2 | Resume does not validate the saved training settings and does not persist patience state; current config can replace run provenance. | Direct code proof. |
| AGY-D-FIND-005 | P2 | `DistributedSampler` uses default padding, so non-divisible training cohorts duplicate a subject per epoch. | Direct code proof; validation sampler is correctly non-padding. |
| AGY-D-FIND-006 | P2 | W&B and in-process gradient/worker cleanup are incomplete on exceptions; CLI exits, but library reuse is not cleanly recoverable. | Direct exception-path proof. |
| AGY-D-FIND-007 | P1 | The locked F1 offset-predictor count is 1,419, but every checked-in server profile configures a 15,107-parameter predictor; the optimizer audit does not enforce the count. | Direct authority/config/model-count proof. |

## 2. Entry points and state machine

The operator CLI parses the training config and calls `run_training()` at `src/smagm/cli/point_guided_train.py:63-130`. The relevant runtime state machine in `src/smagm/training/point_guided.py` is:

```text
CLI/config
  -> initialize_distributed(device)                         (242-265)
  -> rank-specific seed                                  (231-239, 1256-1259)
  -> model/config/split/data inventory on every rank       (1259-1286)
  -> rank 0 chooses run_dir; broadcasts it                 (1289-1298, 274-281)
  -> rank 0 writes metadata; barrier                       (1296-1352)
  -> one Adam optimizer over the authorized set             (1354-1366)
  -> DDP-wrapped context for training; raw context for val  (1367-1376)
  -> optional resume load                                   (1378-1392)
  -> DistributedSampler train / non-padding eval sampler   (1406-1407)
  -> train epoch -> validation epoch -> rank-0 artifacts    (1425-1599)
  -> W&B finish (normal return only) -> teardown barrier     (1600-1606)
```

All ranks independently perform structural and active payload inventory before the run directory is selected (`_prepare_structurally_eligible_split`, `1144-1174`, called at `1276-1286`). This is read-only and deterministic under an immutable dataset, but duplicates startup filesystem/NIfTI work per rank; it is not a synchronization or artifact race fix.

The dedicated `baseline_training.py` module remains an optimizer/policy helper and synthetic smoke path (`1-7`, `208-224`, `259-339`). The server trainer, not the synthetic helper, owns scheduler/checkpoint/DDP behavior. No scheduler is configured or implemented in this path; therefore there is no scheduler state to resume, and no scheduler consistency claim is made here.

## 3. Optimizer ownership and gradient flow

### 3.1 Authorized set

`_components()` at `src/smagm/features/point_guided/baseline_training.py:106-130` names the only optimizer modules:

| Module | Focused smoke (`hidden=12`) | Server profiles (`hidden=128`) | Ownership |
| --- | ---: | ---: | --- |
| `semantic_prior.semantic_head` | 1,539 | 1,539 | Gate-F MAIN |
| `point_refiner.offset_predictor` | 1,419 | 15,107 | Gate-F MAIN; the 2-mm bound remains in the existing predictor |
| `base_plane_projector` | 579 | 579 | Gate-F MAIN |
| `spectral_anchor_builder.band_projector` | 520 | 520 | Gate-F MAIN |
| `trajectory.state_initializer` | 2,080 | 2,080 | Gate-F MAIN |
| `trajectory.reward_net` | 8,193 | 8,193 | Gate-F MAIN |
| `trajectory.update_net` | 47,072 | 47,072 | Gate-F MAIN |
| `decoder` | 8,321 | 8,321 | Gate-F MAIN |
| **Total trainable** | **69,723** | **83,411** | **all included once** |

`resolve_parameter_ownership()` (`142-205`) rejects mixed frozen/trainable modules, duplicate component IDs, missing/unknown model parameters, optimizer members outside the component set, a trainable MedicalNet backbone, and trainable SWT filters. `build_baseline_optimizer()` (`208-224`) constructs one Adam parameter group from those eight modules and revalidates membership. It does not enforce a fixed parameter count, so it cannot catch the server-profile mismatch described in AGY-D-FIND-007. The frozen backbone count is 14,399,424 in the focused tests and production model construction.

### 3.2 Gradient and numerical guards

The trainer checks all non-`None` gradients before stepping (`_assert_finite_optimizer_gradients`, `336-347`), unscales AMP gradients before the check (`797-806`), clips with `error_if_nonfinite=True` (`807-812`), and checks parameters plus tensor-valued Adam state after the step (`349-365`, called at `818` and `865`). This is a good fail-closed numerical boundary. It does not, however, turn a failed step into a resumable/retryable state; the exception path is covered by AGY-D-FIND-006 below.

The target-free context is returned through `_TrainingContextModule` (`284-310`). In distributed training, `_forward_objective()` adds `parameter.sum() * 0.0` for every trainable parameter (`730-739`) before backward. With `find_unused_parameters=False` (`1369-1374`), this keepalive connects route-dependent modules even for a zero-step/early-stop sample without changing the objective value.

## 4. DDP normal-path audit

### 4.1 Initialization and model ownership

`initialize_distributed()` reads `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`, selects NCCL for CUDA or Gloo for CPU, binds `LOCAL_RANK` for CUDA, and calls `init_process_group(env://)` (`242-265`). `run_training()` seeds before constructing the model (`1256-1259`), creates the optimizer over the raw model (`1366`), then wraps `_TrainingContextModule(model)` in DDP (`1367-1374`). DDP's default parameter synchronization occurs after the optimizer has captured the same parameter objects, so the optimizer remains attached to the synchronized tensors.

The model's frozen-backbone policy also freezes BatchNorm state: `SemanticPrior.set_backbone_frozen()` calls `backbone.eval()` and `SemanticPrior.train()` restores that evaluation mode (`src/smagm/features/point_guided/semantic_prior.py:104-123`). This prevents rank-local running-stat updates from being mistaken for trainable optimizer state.

### 4.2 Training and validation collectives

Training uses `DistributedSampler` with `shuffle=True` and a shared seed (`1004-1020`); `set_epoch(epoch)` is called before each epoch (`1428-1431`), so all ranks have equal batch counts for DDP backward collectives. Validation uses `DistributedEvalSampler` (`1021-1023`), whose strided indices (`139-145`) are disjoint and do not pad. Validation binds the raw context module rather than DDP (`1375-1376`), runs under `torch.no_grad()` (`785-786`), and performs only one final metrics reduction (`573-595`). This is the correct shape for an uneven validation cohort.

Rank 0 computes the best metric, writes artifacts, and broadcasts the early-stop flag (`1533-1599`). On the normal path, this keeps all ranks in the same epoch state. It does not protect against rank-divergent exceptions; see AGY-D-FIND-001.

## 5. Automatic mixed precision (AMP)

`_autocast_context()` enables CUDA autocast only when `settings.amp` is true and selects fp16/bf16 (`323-326`). `_scaler()` creates a CUDA fp16 GradScaler only for fp16 CUDA (`329-333`); CPU and bf16 paths intentionally have no scaler. The step sequence is forward/autocast -> scaled backward if needed -> unscale -> finite check -> clip -> step/update -> finite parameter/state check (`785-819`).

The focused CPU server tests exercise the no-AMP branch and numerical guards, not CUDA autocast, GradScaler overflow behavior, NCCL, or GPU memory. The checked-in server profiles enable AMP by default (`configs/training/point_guided_brats21_4070.json:45-69`, `point_guided_brats21_2xa4000.json:43-60`), so those runtime gates remain server-only evidence. The 4070 launcher exposes `POINT_GUIDED_DISABLE_AMP` (`scripts/point_guided_train_4070.sh:15-24`); the 2x A4000 launcher does not expose a no-AMP profile (`scripts/point_guided_train_2xa4000.sh:26-35`).

## 6. Findings

### AGY-D-FIND-001 — P1: failure teardown can deadlock the remaining ranks

**Evidence.** `run_training()` places `destroy_distributed(context)` in a broad `finally` (`1240-1257`, `1605-1606`). `destroy_distributed()` unconditionally calls `torch.distributed.barrier()` before destroying the process group (`268-272`). The main loop has rank-sensitive collectives and rank-0-only file writes: validation reductions (`573-595`), rank-0 checkpoint/log writes (`1533-1587`), and stop-flag broadcast/barrier (`1589-1597`). `run_epoch()` catches CUDA OOM only to wrap and re-raise (`840-847`); it does not notify or abort peer ranks.

**Failure sequence.** If rank 1 raises during a batch, it leaves `run_training()` and enters the teardown barrier. Rank 0 can still be inside a DDP backward all-reduce, epoch-end `_reduce_stats()` all-reduce, rank-0 checkpoint write, or the stop-flag broadcast. These are different collective sequences; rank 0 may never reach the teardown barrier, or ranks can enter mismatched collectives and wait until the process-group timeout. The same issue occurs if a rank-0 artifact write fails while another rank is already at the stop broadcast. The final barrier is therefore safe only after all ranks have already followed the same control flow; it is not failure-safe.

**Impact.** A real-data OOM, corrupt input, failed checkpoint write, or Python exception can leave the torchrun job hung rather than failing promptly. The advertised “failure cleanup” in `docs/POINT_GUIDED_SERVER_RUN.md:131-136` is not established by this control flow. No two-process GPU reproduction was run in this environment, so the deadlock claim is a direct collective-order proof, not a measured timeout.

### AGY-D-FIND-002 — P1: independent jobs have no run reservation and collide on artifacts

**Evidence.** Without `--resume`, `run_dir` is `output_root / run_name` or `output_root / datetime.now(...).strftime("point-guided-%Y%m%dT%H%M%SZ")` (`1289-1295`). The timestamp has one-second resolution and `mkdir(exist_ok=True)` does not reserve a new run (`1296-1298`). The overfit launcher supplies a reusable default `RUN_NAME=point-guided-overfit-4070` (`scripts/point_guided_overfit_4070.sh:8`, `29-36`), while the 4070 main launcher accepts a reusable `POINT_GUIDED_RUN_NAME` (`scripts/point_guided_train_4070.sh:22-24`). There is no lock, `O_EXCL` directory claim, PID/UUID manifest, or “existing run requires resume” check.

Rank-0-only writes prevent *intra-torchrun* rank races, but not two independent training processes targeting the same `run_dir`. The JSON helper uses one fixed temporary name per destination (`_atomic_json`, `217-221`); the log helper appends to shared files and suppresses the header whenever `metrics.csv` already exists (`_write_epoch_logs`, `1177-1186`); checkpoints use atomic rename but still have last-writer-wins semantics (`baseline_checkpoint._atomic_torch_save`, `20-30`). Two processes resuming the same `last_train.pt` likewise load the same state and race to append logs and replace `best_model.pt`, `last_train.pt`, and `summary.json`.

**Temporary reproduction.** An 8-thread, 40-write reproduction against `_atomic_json()` in a temporary directory observed `10` `FileNotFoundError` failures for the shared `.config.json.tmp` path while leaving a final destination file. This was written outside the repository; it demonstrates the fixed-temp-name race, not a production-data run.

**Impact.** Concurrent jobs can lose checkpoints, mix epoch records, produce a malformed/column-shifted CSV, or leave a run whose `config.json`, summary, and checkpoint came from different invocations. Per-file atomic rename protects readers from a partially serialized torch payload, but it does not provide run-level isolation or provenance integrity.

### AGY-D-FIND-003 — P2: DDP resume restores rank 0 RNG state on every rank

**Evidence.** `_set_seed(seed, rank)` deliberately creates distinct rank streams (`231-239`). `save_training_resume_checkpoint()` captures only the calling process's Python, NumPy, CPU Torch, and all-visible-CUDA RNG states (`baseline_checkpoint.py:33-43`, `59-87`). The save call is inside `if context.is_main` (`1533-1565`), so the file contains rank 0's state. On resume, every rank calls `load_training_resume_checkpoint()` (`1380-1392`), and `restore_rng_state()` overwrites its local state from that one payload (`baseline_checkpoint.py:46-56`, `137`).

The current main path limits immediate divergence: `DistributedSampler` uses its explicit `seed` plus `set_epoch()`, and Gate-E counterfactual candidates use a per-batch generator seeded from `settings.seed + global_step` (`_forward_objective`, `703-709`). Those choices make the main candidate/sampler sequence rank-independent rather than proving that restoring rank 0 state is correct. DataLoader workers (`num_workers=2` in the main profiles) and any future/random Python, NumPy, CPU, or CUDA operation would inherit rank-collapsed state after resume.

**Impact.** A resumed DDP run does not reproduce the pre-resume per-rank RNG streams and can correlate worker/random augmentations or other rank-local stochastic behavior. The issue is a reproducibility and future-proofing defect, not evidence that current target-free routing receives rank-specific random input. A per-rank checkpoint state or an explicit post-load rank reseed/state broadcast protocol is required for a complete contract.

### AGY-D-FIND-004 — P2: resume accepts protocol drift and resets early-stopping progress

**Evidence.** The resume loader verifies the schema, exact key set, model state, optimizer state, scaler shape, and split hash (`baseline_checkpoint.py:100-143`), and returns the saved `training_config` and `metadata` (`139-144`). `run_training()` consumes only `epoch`, `global_step`, and `best_validation_reconstruction_loss` (`1380-1390`); it does not compare the returned training settings/metadata with the current config or overrides. Before loading the checkpoint it writes the current invocation's `config.json` to the existing run directory (`1296-1333`).

`patience_count` is initialized to `0` after the resume block (`1408-1412`) and is not stored in the checkpoint payload. Consequently, early stopping starts a fresh patience window even when the prior epoch was close to the configured limit. Current overrides can change AMP/dtype, gradient accumulation, decoder chunking, supervision weights, semantic weight, epochs, or other training behavior while retaining the old model/optimizer state. No scheduler exists in this path, so scheduler state is absent rather than incorrectly restored.

**Impact.** A “resume” can silently continue with a different optimization protocol, overwrite the original run configuration, and run longer than the original early-stopping state would allow. The split hash protects cohort identity only; it does not protect training-protocol provenance.

### AGY-D-FIND-005 — P2: training `DistributedSampler` pads non-divisible cohorts

**Evidence.** Training constructs `DistributedSampler` without `drop_last` (`_make_loader`, `1004-1020`), so PyTorch's default is to pad the index set to equal per-rank lengths when the dataset size is not divisible by `world_size`. The checked-in 2x A4000 semantic-weight provenance records `train_subject_count: 965` (`configs/training/point_guided_brats21_4070.json:55-60`); an active 965-subject training split on two ranks therefore has 966 sampler positions and one repeated subject per epoch. Small overfit cohorts can duplicate the only subject on every rank. `set_epoch()` changes the shuffle but does not remove padding (`1428-1429`).

This is distinct from validation: `DistributedEvalSampler` explicitly uses disjoint strided indices and has a test proving no padding duplicates (`tests/features/point_guided/test_point_guided_server_pipeline.py:327-332`). Training padding is a standard DDP mechanism that preserves equal backward collective counts, but it changes subject weighting and contradicts an unqualified reading of “one sample per GPU” for uneven cohorts.

**Impact.** One or more subjects receive extra gradient exposure each epoch; for small/overfit runs the duplication can dominate the update. The behavior is deterministic and not a cross-process file race, but the run metadata does not report padded/unique sample counts or compensate for the weighting.

### AGY-D-FIND-006 — P2: exception cleanup is incomplete for W&B and in-process retry

**Evidence.** W&B is initialized only on rank 0 (`1413-1424`) but `wandb_run.finish()` is reached only after the epoch loop completes normally (`1600-1601`). Any exception from data loading, forward, backward, optimizer guards, checkpoint serialization, logging, or the explicit OOM wrapper bypasses `finish()` and proceeds directly to the distributed teardown in `finally`. During `run_epoch()`, a failure after one or more accumulation micro-batches also leaves partial gradients in the optimizer; there is no exception cleanup/`zero_grad()` before re-raising (`765-868`, especially `793-819` and `840-847`). DataLoaders configured with `persistent_workers=True` (`1027-1036`) have no explicit shutdown path in the training function.

**Impact.** The command-line process normally exits after an exception, so the OS eventually reclaims resources, but a long-lived caller that catches the error cannot safely retry the same trainer without clearing partial gradients and worker state. W&B runs may remain open or report an incomplete lifecycle. This finding is separate from AGY-D-FIND-001: even if process-group teardown were made failure-safe, external logger and in-process state would still need cleanup.

### AGY-D-FIND-007 — P1: checked-in server profiles violate the locked 1,419-parameter offset-predictor contract

**Evidence.** The active authority states that `point_refiner.offset_predictor` is a MAIN Gate-F trainable with exactly 1,419 parameters (`AGENTS.md:200-207`, `PLAN.md:1476-1483`). That count corresponds to the focused test model's `offset_hidden_channels=12` (`tests/features/point_guided/test_baseline_training.py:22-37`, `50-76`). Every checked-in server training profile instead sets `model.offset_hidden_channels` to `128` (`configs/training/point_guided_brats21_overfit.json:13-18`, `point_guided_brats21_4070.json:13-18`, `point_guided_brats21_2xa4000.json:13-18`). Direct model construction from the 4070 profile gives 15,107 offset-predictor parameters and 83,411 trainable parameters total; the focused smoke model gives 1,419 and 69,723 respectively.

`PointGuidedConfig` validates only that `offset_hidden_channels` is positive (`src/smagm/features/point_guided/config.py:84-93`), and `resolve_parameter_ownership()` validates module membership but not the locked count (`baseline_training.py:142-205`). The optimizer therefore legitimately includes all parameters of whichever width the config selects, while the server presets silently select a different width from the Gate-F authority.

**Impact.** A server F3/F4 run can be reported as using the locked Gate-F baseline while optimizing an offset predictor more than ten times larger than the locked 1,419-parameter contract. This is an optimizer/trainable-set and experiment-provenance defect, not an unauthorized module leak; the predictor remains observation-only and its displacement bound is unchanged. The report records both counts so downstream reviewers do not mistake the synthetic 69,723 count for the server run.

## 7. Verified controls and non-findings

| Area | Result | Evidence |
| --- | --- | --- |
| Optimizer ownership | **PASS, with count mismatch** | Exact eight-module membership, duplicate/exhaustiveness checks, frozen backbone exclusion, and SWT buffer guard in `baseline_training.py:142-205`; focused ownership/smoke tests pass. The server-profile width/count mismatch is AGY-D-FIND-007. |
| DDP wrapper ownership | **PASS on normal path** | One `_TrainingContextModule` wrapped for training, raw module for validation (`1367-1376`); optimizer points at the same raw model parameters. |
| DDP reducer participation | **PASS on normal path** | Exact-zero keepalive in `_forward_objective:730-739` with `find_unused_parameters=False`. |
| Validation duplication | **PASS** | Non-padding `DistributedEvalSampler` (`123-145`, `1021-1023`) and focused disjoint-shard test. |
| Normal-path metric reduction | **PASS** | Sample-weighted numeric accumulation and one epoch-end all-reduce (`511-595`). |
| AMP numerical guards | **STATIC PASS** | Autocast/scaler selection and pre/post-step finite checks (`323-365`, `785-819`); CUDA/NCCL behavior not exercised locally. |
| Intra-run rank writes | **PASS, limited** | Only rank 0 writes ordinary metadata/logs/checkpoints (`1296-1350`, `1533-1587`), and per-file atomic operations avoid partial destination payloads. This does not solve independent-run collisions (FIND-002). |
| Early-stop synchronization | **PASS, normal path only** | Rank-0 flag broadcast plus barrier (`1589-1597`); not failure-safe (FIND-001). |
| Checkpoint split binding | **PASS, limited** | Exact resume schema and split-hash check (`100-143`); training protocol/patience/RNG limitations remain (FIND-003/004). |

## 8. Verification performed

1. `python scripts/codegraph.py --task baseline_training` — exited 0; printed the scoped baseline entrypoints/read paths/write paths.
2. `python scripts/codegraph.py --task server_pipeline` — exited 0; printed the scoped server entrypoints/read paths/write paths.
3. `PYTHONPATH=src .venv/bin/python -m pytest -q tests/features/point_guided/test_baseline_training.py tests/features/point_guided/test_point_guided_server_pipeline.py` — **18 passed in 244.94s**.
4. Temporary threaded `_atomic_json()` reproduction under the system temporary directory — 40 concurrent writes from 8 threads produced 10 fixed-temp-path `FileNotFoundError` observations and a final destination file; no repository path was written.
5. `PYTHONPATH=src .venv/bin/python` model construction from `configs/training/point_guided_brats21_4070.json` — `15,107` offset-predictor parameters and `83,411` trainable parameters total; this is the direct count used in AGY-D-FIND-007.
6. Targeted `compileall` for the audited trainer, baseline checkpoint/training helpers, and training CLI — exited 0; an `awk` trailing-whitespace check on this report — exited 0.

Not executed: real BraTS21 loading, CUDA AMP, NCCL/Gloo multi-process training, rank-failure injection, checkpoint resume across multiple ranks, W&B lifecycle, GPU memory/throughput, or trained-checkpoint evidence. The focused tests are software evidence only.

## 9. Final assessment

The authorized module set and normal DDP/validation/AMP control flow are well-bounded, but the locked parameter-count authority is not enforced by the server presets, and the training runtime is **not concurrency-safe or failure-recoverable at the run level**. Before treating F3/F4 execution as operationally robust, the owner should reconcile the offset-predictor width/count, address run reservation/collision rejection, coordinated rank failure handling without a mismatched teardown barrier, per-rank resume RNG state, strict resume protocol/progress validation, and exception cleanup for logger/gradient/worker state. No production remediation was performed in this audit.
