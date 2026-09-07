# PFGR-Lite implementation review

Status: **SOFTWARE REVIEW ACCEPTED — scientific/runtime acceptance remains pending**.
Last updated: 2026-09-07. Publication requires the normal push and independent remote verification recorded in the final handoff.
This document records software evidence; it does not establish trained improvement.

## Authority and versions

- Initial source and last fetched origin/main: `5456a9eb0721cbf788782af2fbbff50222b99e3b`.
- Integration branch: `pfgr-lite/integration-20260907`; current committed head
  `3f08288be607c0b5214d3f2af686cd768f1e907f`, including the independently accepted metadata packaging patch. The root Vietnamese runbook, CI and persistent command regressions are accepted. CLI/calibration/checkpoint integration is accepted through `b1daf18`, and the legacy expectation-only repair is `0eb3582`. The strict factory-source binding and actual NIfTI resume are accepted. W3b and W5 services are accepted through `f467ebc` and `c2b9f4b`.
- Primary report actually read: supplied root `deep-research-report.md`,
  “Astra Deep Research Final — Point-Guided MRI Reconstruction”,
  SHA256 `918dd465a8d674bcdcdb0cc62918c8db4c083eacebbe4d744d2af28c504f3fc7`.
  Intended `reports/astra-deep-research-final.md` was absent. No older report was substituted.
- Frozen implementation plan: `docs/implementation/PFGR_LITE_IMPLEMENTATION_PLAN.md`,
  SHA256 `5cd2cd9ecc720aee6b2463de6dcfbf5f4b0a8b5c48723409e5c29af515de7985`.
- Interface companion SHA256:
  `0fafe9413644893e516b21c0fd61caeabd6bfbb28c4b597c4689ee7dd325c61a`.
- `docs/implementation/PFGR_LITE_EVIDENCE_CONTRACT.md` amendments through
  `81fecbd` freeze actual diagnostic teacher, parallel, runtime, input identity,
  S5 collection, cached-fit continuation, selected-state audits, S2 measurement budgets and narrow legacy loader seams.
- Principal coordinator verified as `gpt-6-astra/high`; implementation workers
  verified as `gpt-5.6-luna/max` through Orca task/dispatch provenance.
  Reused-terminal receipts with null launch fields are not new model launches.
  Earlier planning artifacts were independently reviewed without asserting
  uninterrupted model attribution for their interactive model changes.

## Accepted design clarifications

MAIN retains ordered T1/T2/FLAIR, one shared frozen MedicalNet ResNet10 traversal,
frozen BN, the pre-MaxPool spectral tap, observation-only geometry/points/masks,
three 32-channel dynamic planes and final-Z-only pointwise decoder.
N=2048, support=4 mm, point displacement bound=2 mm, delta=96 dimensions,
write_scale=0.1 and FP32 remain reference controls.

The four separately versioned base controls are B0/B1/B2/B-light. B2's ordered
source conditioning remains inside Z0. No measured claim that B2 is strongest
is made. Existing B/A/spectral producers remain separate from the dynamic base.

The report's expected-mean/SmoothL1 ambiguity is resolved as fixed training-bank
scale MSE in MAIN, with signed raw Charbonnier gains and a positive scale floor
of 1e-8. SmoothL1 is a named robust ablation. No local pre-error division,
positive clipping, optionally stopped label means or hidden target mining is
accepted. Fixed-Q complete-support sampling is default; exact enumeration is
reference. Independent confirmation is required after noisy winner screening.

Detached teacher and differentiable updater objectives are separate. Frozen D
must transmit input gradients to U. S1 has U-only and U-plus-spectral controls;
producer freezing and evidence precede the immutable ValueBank. S3 fits cached
descriptors only. V366 is o270 plus the actual 96-dimensional proposal, signed
linear output; V126/V270 and historical V222 are explicit controls.

One action-consistent policy scores and applies the stored proposal, rejects
stale actions, permits fresh revisits and deterministic ties, and supports K0/1/2/4.
Adaptive STOP uses positive affine raw gain minus nonnegative empirical allowance,
practical margin and explicit optional compute cost. Numerical tolerance is
separate. No travel/overlap penalty, forced first adaptive step or ST gradient
into V is introduced. Missing/stale calibration blocks adaptive deployment.

Calibration requires disjoint training-only roles, at least 32 independent
groups and 64 unique measured winners per fit/allowance role, and actual complete
forced K4 traces before target measurement. No conditional-coverage or clinical
claim is made. Insufficient data is INCONCLUSIVE and must produce no fitted
adaptive artifact. Correction headroom (oracle versus Z0) and selection headroom
(oracle versus random) remain distinct.

These are implementation choices, not measured research findings.

## Component and integration evidence

Interpreter: `/Users/alvinluong/miniforge3/bin/python`, Python 3.11.16,
torch 2.13.0, macOS arm64. CUDA unavailable. Team real-data/checkpoint variables
were not configured. W&B was not installed in this environment.

| Scope / exact command or probe | Actual result |
|---|---|
| Initial `python -m pytest tests/features/point_guided -q` | 333 passed, 2 failed, 18 skipped, 143.03 s |
| Unchanged distributed test with `PET_LOCAL_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo0` | 1 passed, 2.94 s; resolves local rendezvous host issue |
| Latest legacy: `PYTHONPATH=src PET_LOCAL_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo0 python -m pytest -q tests/features/point_guided --ignore=tests/features/point_guided/pfgr_lite` | 336 passed, 18 skipped, 71.08 s |
| Clean-worktree full repository `PYTHONPATH=src PET_LOCAL_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo0 python -m pytest -q -ra` at abe252d | 982 passed, 18 skipped, 26 subtests passed, 1 pre-existing warning, 115.34 s; checkout stayed clean; final packaging changes checked separately |
| W1 final three owned files | 38 passed, 11.39 s; accepted through 9f03660 |
| W2 final original owned suite plus direct-context guard | 45 passed, 3.55 s, then 1 passed, 1.38 s; committed 4127a97 |
| W3a bank/value two owned files | 36 passed, 1.75 s; committed 81e9b22 |
| W4 five owned action/policy/calibration/checkpoint/inference files | 30 passed, 5.71 s; committed db855fd |
| `python -m pytest -q tests/features/point_guided/pfgr_lite/test_stages.py tests/features/point_guided/pfgr_lite/test_data_boundary.py tests/features/point_guided/pfgr_lite/test_bank_audit.py tests/features/point_guided/pfgr_lite/test_value_bank.py tests/features/point_guided/test_frontend_boundaries.py` | 75 passed, 2 skipped, 8.96 s; W3b accepted |
| W5 test_experiments/test_oracle/test_metrics/test_parallel/test_benchmark plus test_teacher | 46 passed, 7.73 s independently after service review; committed c2b9f4b |
| `python -m pytest -q tests/features/point_guided/pfgr_lite/test_end_to_end.py` | 2 passed, 8.91 s; this chain alone did NOT cover real continuation or production S5 acceptance |

Commands above use `rtk proxy` in the coordinator environment and `PYTHONPATH=src`
where needed. Counts overlap and must not be summed into a final acceptance total.
Full-repository tests and final CLI/runbook/package acceptance pass. A fresh pre-publication fetch kept origin/main at 5456a9e; no concurrent source changes required reconciliation. The current factory/data/stage/frontend subset additionally passed 46 tests with 2 CUDA skips in 6.18 s. The 18 legacy skips comprise 2 frontend CUDA/AMP checks, 12 point-refinement CUDA/AMP checks, 2 RewardNet CUDA/AMP checks and 2 spectral-query CUDA/AMP checks.

The initial legacy configuration failure was a stale expected lambda_step=0.05;
commit `8988d352cff9c6e3fb6d67ec25def90b2aa80a1a` already changed the 4070
configuration to 0.025 with bounded/separate-halt exploration controls. Only
history-backed test expectations were repaired. The new import exception
allows exactly two existing loader symbols in PFGR data.py/stages.py; it does
not allow target/data imports throughout inference.

A pristine `5456a9e` versus accepted W1 comparison used seed 41037, FP32,
four points, 9³ observations, anisotropic spacing and legacy K2. All 108
state-dict entries, frontend/trajectory and final reconstruction byte hashes
matched. Historical smoke SHA `d02e50b57d5d82165641f1f39a16b83a9d6e431b`
could not be recovered (`git cat-file` exit 128); historical reproduction
remains UNVERIFIED.

## Independent numerical and actual-model probes

- Canonical affine build/query chunk invariance: 20 random affine fixtures
  per dtype, 120 output voxels, chunks 120/120 versus 13/7; FP64 and FP32
  maximum difference zero. Earlier FP32 chunk-order failures were fixed
  without relaxing tolerances. This does not assert all legacy BLAS rounding
  paths are bitwise identical to the new canonical lattice.
- Full-write versus sparse-query bounded correction gradient, frozen nonlinear
  D, indexed/fallback paths: four FP64/FP32 cases passed original tolerances.
  Maximum prediction/gradient differences: FP64 5.55e-17/5.42e-20;
  FP32 1.49e-8/2.91e-11.
- Actual random K2 model/teacher/final-decode integration: exact telescoping
  residual 1.8936546767513653e-10. Negative random-weight net gain is retained.
- Local-positive/global-negative FP64 counterexample with actual 4-mm writer:
  local gain +0.022165315450515075, global gain -0.0057392046121678385;
  257 sphere voxels, 1815 affected output voxels, 2673 total.
- Primitive actual gradient probe: U norm 0.0006483004783789253,
  spectral projector norm 0.000039936114100679986, D/backbone parameter norms
  zero, one MedicalNet traversal. This establishes gradient flow only.
- Actual saved-checkpoint CLI S0→S1→S2 (fixed-Q=4)→bank-verify (two snapshot rows)→V chain passes. The independent direct staged S0→S1→S2→S3 engineering chain also executes. Four bank rows;
  S3 one of four minibatches is correctly pending, fit_subject_count=1,
  and cached fitting observes zero MRI/teacher/U/D calls.
- Actual strict S0 and S1 continuation: three subjects, full three updates
  versus one update plus resumed remainder; model weights bitwise equal
  (max_abs=0), Adam states equal. S1's 12 executed action identities match.
  Actual S0 CLI disk continuation also passes two subjects, two updates versus one plus resume: model/Adam bitwise equal. S1 CLI disk continuation now also passes with two subjects: model tensors and Adam states bitwise equal; the stage-provenance sentinel was corrected.
- Actual cached V CLI disk continuation: eight explicitly manual engineering rows, four minibatches uninterrupted versus one plus saved resume and remaining work. V tensors and complete ValueFitIdentity digest match exactly; saved cached optimizer is nonempty. This is software continuation evidence, not measured MRI gain evidence. Production external role/split joins fail closed before cached fitting; the resolved normalization identity is retained in cached artifacts.
- Generated NIfTI observation adapter, anisotropy/shear/translation, actual
  PFGR encoding: PASS, one observation read and zero target/segmentation reads. The persistent non-synthetic CLI adapter suite independently passed 3 tests in 10.90 s: saved S0→S1→S2, bitwise S0 disk resume (model/Adam), and metadata-only R9 preparation. Its generated one-channel checkpoint actually loaded 72 keys, adapted to three channels and matched the supplied digest; official-pretrained verification remains false.
- Actual N=2048, FP32, 17³ engineering encoding/random K1/final decode:
  finite [1,1,17,17,17] output, 2048 proposals and one write. No real-data claim.
- Corrected actual Oracle K2 executes both winners including final action:
  exact gain sum 0.00022222573170438409, dense final gain
  0.0002222257386392812, residual about 6.94e-12. Constant positive
  engineering U/D and target 100 only force an audit route.
- Actual parallel K2 uses two unchanged state-version-0 proposals. Joint
  versus independent gain interaction is measured separately, not claimed
  as sequential telescoping. Actual local/footprint audits contain two
  action rows with distinct global/local denominators.
- Direct random/oracle artifact comparison joins actual two-subject Z0/producer records and exact per-subject initialization maps; returns INCONCLUSIVE, with both subject gains measured. Same probe executes two-subject benchmark with SOFTWARE_PASS.
- Actual S5 two-subject collection/confirmation passes fixed-Q=4 and exact-Q=0, with 4 fit and 4 allowance winners, 4 MedicalNet traversals, 2 target validations, 2 replays and 16 decoder calls. Sampled and exact query-output counts are 32 and 4376 respectively. Both are INCONCLUSIVE and produce no adaptive checkpoint.
- Existing-directory CLI failure probe now exits 1 and preserves only the
  preexisting teammate.txt; it no longer writes a receipt into that directory.

## Bounded same-work benchmark

The measured record is [pfgr-lite-cpu-software-benchmark.json](pfgr-lite-cpu-software-benchmark.json); it retains the actual source/diff identities, environment, counts and noisy timings.

Executed through the actual CLI:
`python -m smagm.cli.pfgr_lite benchmark --synthetic --config configs/pfgr_lite/synthetic.json --output-root <owned-temporary-repo-directory> --run-name bench --max-subjects 1 --max-states 1 --candidate-count 2 --query-count 32 --repeats 3`.

FP32, six measured rows, identical actions and fixed-Q mixture draws.
Maximum query/prediction/gain errors: 5.960464477539063e-8,
1.4901161193847656e-8, 7.663345513719833e-10; zero parity failures.
Actual decoder calls: shared-before 6, sparse-after 6, reference-after 6.
One target read; zero patient files. Detached/eval service execution is used.
Timing is a tiny noisy CPU observation, not a stable speedup result.
The same benchmark was independently rerun on committed head f467ebc plus the recorded working diff; all three maxima and zero failures reproduced. Its source scope SHA256 was e1411e1994c3873d44cef7ad3e49d7b4d3d73b5bfdc330a90a8583304b923522; dirty diff SHA256 c86a4c2f370d709b10bbc48c00027c09d8e648575e2fff7732b6980b293e9cce. Raw timings remain diagnostic only.

Legacy 4-mm support accounting was independently checked: 257 valid sphere
voxels versus 1331 padded slots on a 17³ fixture; with 12 spill queries,
2686 before/after decoder outputs per action. Valid voxels, padded slots,
output rows, decoder calls, candidates and states are separate quantities.

## Final integration checks

- Actual generated-NIfTI S0 disk continuation, cached V normalization/role joins, metadata-only review config drift rejection, and S5 counters were repaired and independently reviewed. Final targeted command: `PYTHONPATH=src python -m pytest -q tests/features/point_guided/pfgr_lite/test_cli.py tests/features/point_guided/pfgr_lite/test_real_adapter_pipeline.py tests/features/point_guided/pfgr_lite/test_end_to_end.py tests/features/point_guided/pfgr_lite/test_checkpoint.py tests/features/point_guided/pfgr_lite/test_calibration_runner.py`: **28 passed, 22.78 s**.
- `ruff check --isolated --select E4,E7,E9,F src/smagm/features/point_guided/pfgr_lite src/smagm/cli/pfgr_lite.py tests/features/point_guided/pfgr_lite`: PASS. The repository has no configured Ruff/mypy/pyright gate. An exploratory inherited-user Ruff configuration reported 205 style/import/exception findings; this was not the project CI contract. The isolated basic correctness lint above is the explicitly reported check, not a claim that every inherited style rule passes.
- `python -m compileall -q src tests scripts` and `git diff --check`: PASS.
- `PYTHONPATH=src PET_LOCAL_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo0 python scripts/check_phase.py POINT_GUIDED_FRONTEND --run --allow-dirty --report-dir .pytest_cache/pfgr-final-quality`: all three automated checks PASS (601 tests passed, 18 skipped in 99.37 s), **PENDING_HUMAN_GATE**. The dirty override records preserved teammate files; it does not approve a scientific gate. Without the macOS rendezvous environment a worker saw the known distributed timeout; the unchanged test passes with the explicit local environment.
- Replaced obsolete CI gate names T1C/T2/T3/T5 (actual unknown-gate exit 2) with the existing POINT_GUIDED_FRONTEND catalog. Kept the legacy full_static_pipeline two-step train/reconstruct/evaluate/audit chain unchanged: root independently executed all four commands successfully; COMPLETE scope is self_prediction_smoke only.
- Root independently executed the new CI block exactly, substituting only a repository-local temporary output path: smoke→updater→bank-build→bank-verify→V126 fit→paired evaluate→same-work benchmark; all commands and required artifact checks exit 0. All stages are bounded synthetic CPU engineering checks.
- Root executed the receipt writer and V join Python heredocs and R2 synthetic Bash block extracted verbatim from `RUNBOOK_PFGR_LITE.md`. Actual same-bank V126/V270/V366 fit/evaluation produced two rows with identical bank identity; join PASS. Actual `calibrate --dry-manifest` context passed the writer and strict receipt validator; repeat write was refused. R2's synthetic fallback executed successfully with its matching config/source flags. This also caught and repaired Linux-sensitive `S2` versus `s2` paths and the missing `scientific_status` review-context key.
- `PYTHONPATH=src python -m pytest -q tests/features/point_guided/pfgr_lite/test_runbook.py tests/features/point_guided/pfgr_lite/test_acceptance_boundaries.py tests/quality`: **24 passed, 16.70 s**, including the persistent extracted-helper regressions.
- Final metadata packaging independently passed the actual multi-run CLI probe: bank generation, V126/V270/V366 fitting/evaluation, extracted paired join, exact review writer, bounded S5 collection, R2 benchmark, and actual split/role serialization for 1,251 generated IDs. The archive included 38 metadata files and no `.pt`/`.npz` payloads. The R7 evidence remains a two-subject engineering INCONCLUSIVE result. Exact filename/category handling preserves raw-payload/credential checks, scalar counter bounds and typed metadata lists; it also handles the full unchanged split in a tiny pilot.
- Final affected command `PYTHONPATH=src python -m pytest -q tests/features/point_guided/pfgr_lite/test_artifacts.py tests/features/point_guided/pfgr_lite/test_runbook.py tests/features/point_guided/pfgr_lite/test_end_to_end.py`: **35 passed, 32.66 s**. Isolated basic Ruff and diff checks pass after the final package fixes.
- No software correctness blocker remains in the reviewed scope. The committed package at 3f08288 was checked again in the clean release worktree: the same 35 affected tests passed in 33.60 s, isolated basic Ruff passed, and status/diff stayed clean. Only this final review/status documentation follows those code checks. Normal-push remote verification is recorded in the final handoff; no experimental Human Gate is approved by software acceptance.

CPU software success is not GPU/AMP or real-data success. Full calibration,
scientific headroom, strong static comparison, trained reconstruction quality,
resource pilot and final matched cohort/seed studies remain pending team
execution and scientific review. No long real-data training was started.

## Intentionally deferred and runtime input prerequisites

- Optional S4 learner-state collection is deferred; the implemented S4 control is a cached ValueNet refit with fixed producers and scale. PFGR-Full, online RL, Taylor/Jacobian teachers, new backbones and custom CUDA/Triton kernels are outside MAIN.
- Real pretrained source verification is unresolved. The inherited `APPROVED_OFFICIAL_MEDICALNET_RESNET10_SHA256` registry is empty, and no official weights were supplied in this session. A local SHA match proves byte integrity and strict architecture loading only. The narrow PFGR factory binds supplied weights correctly for engineering/static/updater work; production ValueBank and adaptive calibration/checkpoint source gates remain unchanged and require a vetted source registration. No digest was invented or added. The team must review its actual MedicalNet source before real MAIN S2/adaptive work.
- Generated one-channel random checkpoints exercise adaptation and resume only. They retain `official_pretrained_verified=false` and engineering scope. No patient reconstruction, learned ranking improvement or pretrained-quality claim follows from these tests.
- CUDA, AMP, real MRI cohorts, training convergence, static-base selection, correction/selection headroom and full calibration remain pending team execution. A lack of these experimental inputs does not turn a CPU software result into scientific acceptance.

## Required-test navigation

The table maps the requested invariant groups to executable files. It is a
coverage navigation aid, not a substitute for the final test result above.

| Requested checks | Test files under `tests/features/point_guided/pfgr_lite/` |
|---|---|
| 1–2 frozen single traversal, adaptation and ordered source | `test_base.py`, `test_provenance.py` |
| 3 legacy policy/checkpoint separation | `test_checkpoint.py`, `test_policy.py`; legacy suite and independent pristine-source comparison |
| 4–5 batch/serial proposal parity, stored proposal and stale guards | `test_actions.py`, `test_contracts.py` |
| 6–8 complete footprint, sparse/full output and gradient parity | `test_footprint.py`, `test_sparse_parity.py`, `test_teacher.py` |
| 9–12 signed no-op/harm/local-global counterexample/telescoping/fixed-Q expectation | `test_gain_accounting.py`, `test_teacher.py` |
| 13–15 detached V inference, U-through-frozen-D and cached-only V fit | `test_inference_parity.py`, `test_stages.py`, `test_value_fit.py` plus actual gradient probes |
| 16 bank/split invalidation and deterministic resume | `test_value_bank.py`, `test_bank_audit.py`, `test_value_fit.py`, `test_stages.py`, `test_checkpoint.py`, CLI continuation probes |
| 17 selection/STOP/ties/K0/budgets/revisits | `test_policy.py`, `test_inference_parity.py`, `test_actions.py` |
| 18–20 public/validation parity, target replacement and privileged oracle separation | `test_acceptance_boundaries.py` (4 passed, 2.76 s; actual final tensors and stored action identities), `test_data_boundary.py`, `test_oracle.py` |
| 21–22 parallel/sequential semantics and partition-invariant metrics | `test_parallel.py`, `test_experiments.py`, `test_metrics.py` |
| 23 CLI and executable runbook | `test_cli.py`, `test_end_to_end.py`, `test_runbook.py` |

## First team execution

Use a clean main checkout and the Vietnamese root `RUNBOOK_PFGR_LITE.md` as the single entrypoint. Preserve local changes; synchronize only with `git pull --ff-only`. Set the repository environment block (POINT_GUIDED_PYTHON, BRATS21_ROOT, MEDICALNET_CKPT, MEDICALNET_SHA256, BASELINE_SPLIT, OUTPUT_ROOT), then run R0 software/preflight and R1 bounded engineering smoke. A missing vetted MedicalNet source is an explicit runtime prerequisite, not permission to fabricate a digest or bypass the production bank/calibration guard.

Proceed to R2 same-work parity before R3 static controls and R4 updater/headroom. Review R4's distinct correction/selection outcomes before funding bank/V/calibration work. R7 requires its exact training-only cohort receipt; all-N oracle, full-cohort/final-seed studies and R9 matched test evaluation require explicit scientific/resource review. Tiny pilot comparisons remain INCONCLUSIVE. No long real-data training was launched by this implementation phase.
