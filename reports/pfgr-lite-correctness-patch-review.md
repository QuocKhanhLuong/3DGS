# PFGR-Lite correctness patch review

Date: 2026-09-09. Principal reviewer: Astra High. Implementation: GPT-5.6-Luna / max through Orca. **ACCEPTED by Astra for software delivery under the tested CPU scope. Real CUDA execution remains unverified; scientific R5 remains CLOSED.**

## Verdict and scope

The remediation implements a shared device contract, legal target/mask placement, dependency-complete source manifests, the explicit PFGR frontend configuration correction, and separate R4A/R4B/R5 gates. The new R4B diagnostic reuses the existing producer, proposals, writer, decoder and teacher. It does not establish useful correction or routing benefit. No real training, real-data diagnostic, ValueBank experiment, ValueNet fitting, final evaluation or MedicalNet download was run.

The accepted [patch plan](../docs/implementation/PFGR_LITE_CORRECTNESS_PATCH_PLAN.md) governs this change. UpdateNet capacity, writer family, ValueNet architecture, teacher estimand, loss, spectral design and deployment K policy remain unchanged. The refiner width correction is the explicitly authorized configuration repair; shared legacy frontend defaults remain unchanged.

## Starting state and source evidence

Local branch `main`, local HEAD and freshly fetched `origin/main` were all `5780458190825da97323cc9300331ed25b6eb69d`; its production source matches the preceding R0–R4 audit. The prior production commit is `c93f90c9f442c1d343fe84e7dd54df6c98af5d26`. `git status`, branch/HEAD queries, fetch and all-branch recent history were inspected before planning. Historical source `49e0ece2dcaeac146ee1e2a7450f9717be8af193` still does not resolve through local `cat-file`, `show --stat`, `branch --contains` or visible all-branch history. This is an unavailable object in this checkout, not proof that the server commit never existed.

Unrelated modified `.DS_Store` and untracked `03-09-2026-reports/`, `deep-research-report.md`, `experiments_R0_R4/` and `w5a-counter-lgw54dfn/` are preserved and excluded from delivery. All 59 files under `experiments_R0_R4/` and the other preserved evidence paths matched their pre-patch content fingerprints at delivery. The original [evidence audit](astra-r0-r4-evidence-review.md) is unchanged (SHA256 `7c8920394bf20f371355420925c5cffb2ee2bec718e17b8682df28c72495ca78`). The historical dirty diffs remain unavailable; no patch was invented or retroactively certified.

Future source receipts use `pfgr-lite-executable-source-scope-v2`. They record Git SHA, branch, origin/main SHA, whole-tree dirtiness, scoped source/diff hashes, scope version and deterministic `{path,size,sha256}` manifest. The filesystem scope conservatively includes executable repository `src` dependencies, including imported MedicalNet, geometry, decoder and data/loading, plus PFGR configuration and build/dependency manifests. Untracked and ignored executable files affect identity and prevent CLEAN scientific provenance. Unrelated documentation does not alter the scoped digest, but whole-tree dirtiness remains visible. Outside-repository symlinks fail reconstructability checks.

The manifest contains identities, not patient data, checkpoint bytes or arbitrary source-byte archives. A dirty hash does not reconstruct missing bytes. Capture occurs before reserving run output; completion checks detect source changes during execution. Clean pinned Git source plus its manifest is the supported reproducible scientific path. Existing historical receipts stay readable but do not satisfy the new MAIN permit contract. The local checkout remains dirty because the user's unrelated files are preserved; scientific execution should use a clean server checkout.

## Implemented contract and changed-file inspection

Astra inspected every changed implementation, configuration, test and document file. `P/` below means `src/smagm/features/point_guided/pfgr_lite/`; test paths use `tests/features/point_guided/pfgr_lite/`.

| Area / inspected files | Final behavior | Status |
|---|---|---|
| `P/device.py`, CLI `src/smagm/cli/pfgr_lite.py`, `P/stages.py`, `P/experiments.py`, `P/calibration_runner.py` | Explicit CLI device wins over configuration for smoke, static/updater training, benchmark, oracle/evaluate, calibration, bank and headroom. Shared resolver canonicalizes CUDA indices and CPU aliases; unavailable CUDA/index raises without fallback. Model, input masks/observations and generated queries use effective placement. Receipts retain requested and effective device separately. | IMPLEMENTED |
| `P/teacher.py` and device tests | Supplied bool/numeric masks normalize onto the requested device; target aligns with the already-created context dtype/device only after legal target join. Existing gather methods normalize query indices. Target substitution cannot alter the observation context or target-free controls. | IMPLEMENTED |
| CLI source receipt and `test_provenance.py` | Deterministic dependency manifest; clean/dirty, imported/untracked/ignored source, docs exclusion and source drift checks. Dirty execution is engineering/non-final, never a publishable MAIN permit. | IMPLEMENTED |
| `configs/pfgr_lite/main.json`, `P/model.py`, `P/checkpoint.py`, implementation-plan amendment, `test_checkpoint.py` | PFGR MAIN width 12/multiplier 4 enforced. Historical width 128 requires explicit engineering hydration with exact serialized sidecars and strict state loading. MAIN rejects it; no silent conversion. | IMPLEMENTED |
| `P/headroom.py`, `P/oracle.py`, CLI, `test_headroom_gate.py` | Frozen paired R4B: K0, three target-free Random-1 choices, 32-candidate fixed-Q1024 screening and independent exact-footprint confirmation of the same Oracle-1 action. Privileged oracle; evidence/metrics/decision outputs; early decision INCONCLUSIVE. | IMPLEMENTED |
| `P/stages.py`, `P/value_bank.py`, `test_stages.py`, `test_value_bank.py` | MAIN S2/S4 require a reviewed matching headroom permit at stage, public generator and canonical writer boundaries. Custom writer adapters cannot bypass admission. Engineering fixtures remain explicit. Original production role/prior/atomicity checks are retained. | IMPLEMENTED |
| `P/benchmark.py`, `P/artifacts.py`, `test_benchmark.py` | Raw shared-before/optimized/reference/optimized-end-to-end timings; cold/warm summaries, p50/p95, actual workload/device and CUDA synchronization method. Historical harness timing alias remains readable. Exact sparse teacher algebra retained. | IMPLEMENTED |
| `P/__init__.py`, `test_cli.py`, `test_device.py`, `test_calibration_runner.py` | Public typed exports and focused integration/compatibility regressions. | IMPLEMENTED |
| Root runbook, implementation plan, correctness plan and this review | Explicit R4A/R4B/human-review/R5 flow, case table, compliant and historical engineering NEXT-1 commands, accepted-decision prerequisite. | IMPLEMENTED |

Device is operational metadata and does not alone stale a scientific producer. Requested device means the explicit CLI spelling when supplied, otherwise the selected configured/default request; the receipt's argv retains whether the CLI option was supplied.

## Configuration drift and checkpoint compatibility

| Setting | History / authority | Resolution |
|---|---|---|
| `offset_hidden_channels=128` | PFGR JSON introduction `b1daf1865737e8775f0868e05cfa73b4fd6fa7eb` copied legacy width 128. Frozen implementation plan section 5 explicitly locks width 12; no accepted width amendment was found. | **B: accidental PFGR drift.** Restore width 12 only in PFGR MAIN. Historical width 128 scientific producers/checkpoints are stale for MAIN. |
| `point_candidate_multiplier=4` | Shared legacy configuration uses4 since `4ccffcef0d3df0b2734335c34223fc98eda900af`; PFGR production JSON also uses4. No frozen PFGR authority chooses another value; the no-sidecar constructor previously chose3. | **A: retained production convention, explicitly amended now.** Align MAIN constructor to4, retain reduced alternatives only in engineering. No scientific superiority claim for4. |

The dated implementation-plan amendment records this resolution. Producer compatibility already binds the refiner's state/shape and candidate geometry multiplier; source scope v2 separately versions executable identity. Old checkpoints are not reshaped, relabeled or overwritten. A strict real-model width 128 synthetic checkpoint round trip verified every state tensor unchanged under explicit engineering hydration and rejection under MAIN. This is software compatibility evidence, not verification of the unavailable historical checkpoint/source.

## R4B evidence and R5 admission

For each subject the diagnostic builds one actual observation context/Z0 and one typed proposal batch. It freezes the legal sampled pool and three random choices, applies/decodes the random controls, and only then invokes the target provider once. Target is never an input to proposals or random choice. Screening retains signed gains, including negative winners. Exact confirmation reuses the selected action/delta and original pool index; dense final metrics describe that same update. K0 preserves the original state exactly. The additive writer is unclipped, so saturation is explicitly `not_applicable`; correction tensor norm and write norm are distinct quantities.

Artifacts bind candidate IDs/positions, action/delta hashes, screening rows, exact confirmation, mask denominator, query counts, dense no-op/random/oracle metrics, paired effects, uncertainty and timing. Model state hashes before/after must match. Random seeds are averaged within subject and never counted as additional independent subjects.

The scientific enum contains exactly `HEADROOM_CONFIRMED`, `CORRECTION_USEFUL_SELECTION_NOT_NEEDED`, `NO_HEADROOM_OBSERVED` and `INCONCLUSIVE`. Only the first may authorize MAIN R5. The gate verifies actual evidence bytes and a typed human-review record, live producer/source/model and base/updater/split/development-subject/teacher identities, source cleanliness, role allocation and one development subject per related group. MAIN checkpoint lineage also verifies that S1 changed only U and the permitted learned spectral projector, retaining the same base/decoder.

It rechecks pool membership, proposal-before-target flags, screening ranking, same-action exact confirmation, dense/confirmation gain agreement and exact no-op zero. It recomputes paired Oracle-minus-Z0 and Oracle-minus-Random uncertainty from retained subjects. Both normal-approximation 95% lower bounds must exceed a positive predeclared practical margin. The accepted patch adds a conservative independent-subject floor 32; this is a governance floor, **not a power calculation or guarantee of adequate precision**. A four-subject NEXT-1 cannot open MAIN even when apparent gains are positive. Later cohort options are available at the lower-level API; the initial CLI remains locked to four subjects. Human scientific review remains necessary; hashes validate consistency, not the truth of measurements or reviewer identity.

| Observed pattern | Interpretation / action |
|---|---|
| Oracle approximately Z0; Random approximately Z0 with adequate precision | Little observed correction headroom in this trained family; improve base/U or stop routing work. MAIN R5 closed. |
| Random improves Z0; Oracle equivalent to Random | Correction may be useful without learned selection. MAIN selector bank closed. Equivalence needs a predeclared margin; failure of superiority is insufficient. |
| Oracle improves both Z0 and Random with adequate precision and matching accepted provenance | Selection headroom can justify conditional MAIN R5. |
| Wide/noisy intervals, insufficient cohort or incomplete provenance | INCONCLUSIVE; MAIN R5 closed. |

No software maps all outcomes to continuation. Learned-versus-random value is a later experiment, outside NEXT-1.

## Independent verification

Environment: `/Users/alvinluong/miniforge3/bin/python`, PyTorch `2.13.0`, no CUDA device. Astra ran the checks below independently of Luna's pass claims. Counts overlap and must not be summed.

| Check | Observed result |
|---|---|
| Baseline PFGR suite before edits | 274 passed in 53.11s. |
| Focused device/headroom/checkpoint/CLI/provenance/bank/stage/benchmark/calibration modules | 132 passed, 1 CUDA skip in 19.94s. |
| PFGR suite before the final assertion-only device extension: `python -m pytest tests/features/point_guided/pfgr_lite -q -ra` | 328 passed, 1 CUDA skip in 80.97s. |
| Final legacy frontend smoke: `python -m pytest tests/features/point_guided/test_frontend_forward.py -q` | 24 passed in 6.28s. |
| Final target-placement assertion extension and CLI checks | 28 passed, 1 CUDA skip in 8.46s for device + CLI modules after the assertion extension. |
| Final complete repository suite | `PET_LOCAL_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo0 python -m pytest -q -ra`: 1041 passed, 19 CUDA-related skips, 26 subtests passed, 1 warning in 136.89s; zero failures. The warning is the existing tensor-to-scalar assertion in `tests/test_torch_compat.py:15`. |
| Final compileall and `git diff --check` | PASS (both exit0); NEXT-1 shell syntax and literal CLI parser checks also pass without executing the command. |
| Actual CUDA tensor execution / GPU parity | BLOCKED locally; guarded CUDA test skipped. Mock availability/override tests do not establish GPU execution. |

Additional independent bounded probes established the actual typed composition, beyond mocked controller tests:

- A real typed CPU synthetic NEXT-1 CLI fixture completed with four subjects and 32 proposals each; K0 was exact, winner/confirmation identities matched, model state was unchanged and decision stayed INCONCLUSIVE.
- An actual typed two-target substitution probe wrapped original context/proposal/write functions, changed only the deferred target by a fixed offset, and asserted one context, one proposal batch and three random applications before each target callback. All initial XY/XZ/YZ tensors remained exactly equal; candidate/delta hashes and random choices were identical while target-dependent metrics changed. Every confirmation retained its selected action. No target argument entered the write helper.
- An actual toy module state digest was bound into a reviewed synthetic permit and passed through the real stage gate, public generator and canonical writer; its fixture row was readable. Mutated model state and current non-final source were rejected. A separate S2 integration test covers argument propagation. These are unit/integration fixtures, not a real ValueBank experiment.
- Rehashed adversarial evidence with deleted confirmations, broken target-boundary flags, random actions outside the pool, a non-maximal winner or confirmation/dense gain disagreement was rejected after repair. Checkpoint existence, stale identities, INCONCLUSIVE/correction-only decisions and genuinely noisy paired effects also reject MAIN.

Interim runs exposed CLI prerequisite and production bank-fixture integration failures. They were repaired; no production assertion was replaced by engineering-only scope to obtain acceptance. An attempted weakening of the original production bank tests was explicitly rejected. Final results above supersede those interim failures without hiding the causal fixes.

## Deferred work and blockers

| Item | Status / boundary |
|---|---|
| Compressed dirty source/patch archive | DEFERRED. Deterministic manifest is implemented; clean Git source is preferred. Missing historical bytes remain unreconstructable. |
| Integrated Oracle-2 progression / larger-cohort command and power study | DEFERRED. No automatic Oracle-4; existing privileged diagnostic primitives remain separate. Only investigate Oracle-2 after Oracle-1 shows useful correction. |
| Performance optimization / representative CUDA scaling study | DEFERRED. Instrumentation only; no speedup claim, including no claim of at least 4×. |
| Historical source 49e0… and uncaptured dirty diff | BLOCKED by unavailable source evidence. Successor patch does not certify prior exports. |
| Official MedicalNet pretrained source chain | BLOCKED for scientific provenance where prior receipt reports official verification false. Checkpoint integrity alone is insufficient; the existing MAIN guard remains. This does not establish that weights were random. |
| Compliant trained width 12 base/updater pair and real CUDA execution | External inputs/validation still required. No new training was performed. Historical width 128 can be used only for clearly marked engineering NEXT-1. |
| Useful correction, selection headroom, learned routing value | NOT ESTABLISHED. R5 remains CLOSED pending accepted real evidence. |

## Exact NEXT-1 handoff

Use the retained historical producer first for the bounded **engineering** action-capacity diagnostic, without retraining in this phase. Required inputs are the actual matching R3 static and R4 updater PFGR checkpoint bundles, the existing baseline split and reviewed role manifest with four fixed development subjects, observation/target data through the existing loader, and any MedicalNet checkpoint path required by the serialized frontend sidecar. Filenames alone do not verify lineage. Preserve these input hashes; use a clean checkout of the delivered commit. This is a successor-source diagnostic, not a reproduction claim for source 49e0….

```bash
set -euo pipefail
: "${REPO_ROOT:?Set the clean checkout path}"
: "${POINT_GUIDED_PYTHON:?Set the tested environment Python executable}"
: "${BRATS21_ROOT:?Set the existing data root}"
: "${BASELINE_SPLIT:?Set the existing baseline split JSON}"
: "${PFGR_ROLES:?Set the reviewed role manifest JSON}"
: "${PFGR_STATIC_CHECKPOINT:?Set the matching historical R3 base checkpoint}"
: "${PFGR_UPDATER_CHECKPOINT:?Set the historical R4 updater checkpoint}"
: "${OUTPUT_ROOT:?Set the output parent directory}"
PFGR_DEVICE="${PFGR_DEVICE:-cuda:0}"
cd "$REPO_ROOT"
for input in "$BASELINE_SPLIT" "$PFGR_ROLES" "$PFGR_STATIC_CHECKPOINT" "$PFGR_UPDATER_CHECKPOINT"; do
  test -f "$input" || { echo "Missing input: $input" >&2; exit 1; }
done
test -d "$BRATS21_ROOT"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite headroom-evaluate \
  --engineering-only --config "$REPO_ROOT/configs/pfgr_lite/main.json" \
  --data-root "$BRATS21_ROOT" --split-file "$BASELINE_SPLIT" --roles-file "$PFGR_ROLES" \
  --base-checkpoint "$PFGR_STATIC_CHECKPOINT" --checkpoint "$PFGR_UPDATER_CHECKPOINT" \
  --split-role validation --max-subjects 4 --candidate-count 32 --query-count 1024 \
  --teacher-mode iid_fixed_q --practical-margin 0 --device "$PFGR_DEVICE" --no-amp \
  --output-root "$OUTPUT_ROOT" --run-name "R4B-NEXT1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

Seeds 17/29/41 are fixed internally; exact-footprint same-winner confirmation is automatic. Expected output under the new run directory: root `receipt.json`, `resolved_config.json` and `service_receipt.json`, plus `next1/next1_evidence.json`, `next1/headroom_metrics.json` and `next1/headroom_decision.json`. The decision is early INCONCLUSIVE and the engineering receipt cannot open MAIN banking. For a new compliant width 12 producer, use the separate runbook recipe; merely removing `--engineering-only` from a historical width 128 command is invalid.

**R5 is still CLOSED until R4B headroom evidence is accepted.**

## Delivery acceptance

Astra accepts the inspected 28-file implementation/test/document change as one atomic correctness patch after the independent full-suite pass. No unrelated file or experiment artifact belongs in the commit. Normal push to origin/main is authorized after a fresh non-destructive reconciliation; remote HEAD is verified separately in the final handoff. The implemented gate is software evidence only. All DEFERRED and BLOCKED items above remain explicit and no real R5 permission has been issued.
