# Astra R0–R4 evidence review

Date: 2026-09-08. Repository: `QuocKhanhLuong/3DGS`. Audited production reference: `c93f90c9f442c1d343fe84e7dd54df6c98af5d26` (`origin/main`). Scope: report only; no production edits, training, new experimental executions, or artifact renaming. This report audits the locally supplied export, not unseen server files or live W&B history.

Evidence language: **FACT** means directly observed repository/export content; **SOURCE EVIDENCE** identifies the implementation or retained record; **INFERENCE** is a conclusion conditional on that evidence; **HYPOTHESIS** is untested; **RECOMMENDATION** is future work, not a result. Numerical observations below refer to the run aliases and files defined in section 3. All proposed sample counts and budgets in section 16 are recommendations, not measured costs or power guarantees.

## 1. Executive verdict

**DO NOT PROCEED to scientific R5 ValueBank generation. Current R4 is an S1 updater engineering smoke with embedded training diagnostics, not an executed scientific headroom gate.** The two R4 receipts contain no oracle evaluation, independent winner confirmation, frozen final-checkpoint random evaluation, or learned selector. Current evidence therefore supports none of headroom cases A–D: **INCONCLUSIVE**.

The export establishes successful bounded S0/S1 optimization mechanics and reported FP32 sparse/full-write numerical agreement on CPU and CUDA. It does not establish a strong base, useful trained corrections, correction-family exhaustion, selection headroom, spectral benefit, or learned routing value. The tiny negative R4 increments are effectively negligible in the observed forwards; they are not evidence that an oracle cannot find useful actions. The R3 objectives are training losses on different subjects during optimization, not validation scores or a base ranking.

Provenance is mixed. The recorded SHA is unavailable locally and through origin, but current main exactly matches the **scoped source content hash** of the first clean R0/R1 attempts. Those attempts are **PARTIALLY VERIFIED**, not invalid by fiat. Successful later runs have different dirty-scope hashes and no retained patch: their exact executable source is **UNVERIFIED**. Their metrics remain interpretable as reported observations with restricted causal attribution. Missing checkpoints, resolved configs, roles/splits, and detailed benchmark rows make this an incomplete evidence export.

## 2. Evidence/provenance status

### Repository inspection

**FACT — audit commands, before report creation:**

| Check | Observed result |
|---|---|
| Local branch / HEAD | `pfgr-lite/integration-20260907` / `c93f90c9f442c1d343fe84e7dd54df6c98af5d26` |
| Local `main` branch at entry | `5456a9eb0721cbf788782af2fbbff50222b99e3b` |
| Refreshed `origin/main`; `git ls-remote origin refs/heads/main` | `c93f90c9f442c1d343fe84e7dd54df6c98af5d26` |
| `git fetch origin main` | Successful; no remote advancement |
| `git status --short --branch` | Modified `.DS_Store`; untracked `03-09-2026-reports/`, `deep-research-report.md`, `experiments_R0_R4/`, `w5a-counter-lgw54dfn/` |
| Whole working tree dirty? | **YES**; unrelated ownership preserved |
| PFGR receipt scope dirty? | **NO** at audit entry; it excludes those unrelated paths |
| Shallow repository? | `git rev-parse --is-shallow-repository` → `false` |

Recent `git log --all --oneline --decorate --graph -45`: `c93f90c` acceptance report; `3f08288` packaging allowlist; `abe252d` runbook/CI; `0eb3582` legacy test constants; `b1daf18` CLI integration; `fbd40dc` factory/NIfTI tests; `81fecbd` source boundary/CPU parity; `f0f7edb` dry review/ranking joins; `f467ebc` stage services. The graph places these on the integration/origin-main ancestry, with the local `main` ref still at `5456a9e` at entry. No visible branch contains the run SHA.

For `49e0ece2dcaeac146ee1e2a7450f9717be8af193`, `git cat-file -t` fails with “could not get object info”; `git show --stat` fails “bad object”; `git branch -a --contains` fails “no such commit”; explicit `git fetch origin <sha>` fails “upload-pack: not our ref”. These checks establish unavailability in this non-shallow object database and from this origin, not proof that the commit never existed on the execution server. It cannot presently be compared commit-by-commit with main or shown to be an ancestor of origin/main.

### Scoped source fingerprints

All execution receipts report the same SHA above and a scope of 66 files. The following exact hashes come from their `source` objects. Current `_source_receipt()` was called read-only with bytecode writes disabled and reproduced S0-clean below.

| Source alias | Runs | Dirty diff SHA256 | Scoped source SHA256 | Classification |
|---|---|---|---|---|
| S0-clean | R0a, R1a | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e73f4c764894da5345b6a5d82ce97a6cb32d53be50a686cc98b5de0051642d62` | **PARTIALLY VERIFIED**: same scoped bytes as current main, unavailable full commit |
| S1-dirty | R0b, R1b, R2cpu | `d852129814ea05cd14f94a63d55c417d12dd15594c7e37c553fe012b7a9c121e` | `099f7ed4d8e18bfb62842b1320610356f08b0eb8f76b2dc7bc53e46d230c7b0b` | Exact source **UNVERIFIED** |
| S2-dirty | R2gpu-fail | `451b1efe101ac9f264aae89afca0b2de3940ca2b5cabb4912f57b2e6e75ab047` | `5b4bdf9352f163f7b790cc1bfa52ac0d63ab18e91202f1b59d74b414facb168c` | Exact source **UNVERIFIED** |
| S3-dirty | R2gpu, all R3 and R4 | `eb4fa992e53c44a1fb5b24843f876c3d0af15087f186dff69d8ceefdc5195fe5` | `49d31433b0dc9db68ff7d6e99e1826c047758b2ed86adbf8450987650a5713a1` | Exact source **UNVERIFIED** |

**SOURCE EVIDENCE:** [CLI source receipt](../src/smagm/cli/pfgr_lite.py), `_scoped_source_paths`, `_scoped_source_digest`, `_scoped_dirty_diff`, `_source_receipt` (lines 386–488). Scope includes PFGR package/CLI/tests/configs/runbook/README indexes, but excludes imported legacy frontend, MedicalNet, geometry, decoder and data modules. The dirty hash covers tracked `git diff --binary HEAD` bytes in that scope; relevant untracked files affect the content hash but their bytes are not supplied by the diff hash. Neither a hash nor a matching file count reconstructs missing content. Whole-tree “clean” is not established by a scoped clean flag.

### Source documents and export completeness

`reports/astra-deep-research-final.md` is absent. The implementation plan and prior review explicitly identify root [deep-research-report.md](../deep-research-report.md) as the supplied final research attachment; its current SHA256 is `918dd465a8d674bcdcdb0cc62918c8db4c083eacebbe4d744d2af28c504f3fc7`, matching their recorded source. It remains untracked and unchanged. Relevant authorities also include [core optimization](astra-core-optimization-sota-expectations.md), [trajectory synthesis v2](astra-trajectory-deep-synthesis-v2.md), [implementation review](pfgr-lite-implementation-review.md), [frozen plan](../docs/implementation/PFGR_LITE_IMPLEMENTATION_PLAN.md), and [runbook](../RUNBOOK_PFGR_LITE.md). Their older synthetic results and external-paper comparisons are context, not additional R0–R4 executions.

**FACT — recursive inventory:** 47 descendant directories (48 including `experiments_R0_R4` itself), 59 regular files, 348,265 bytes, 13 top-level execution directories under `pfgr-lite`, 13 receipts, 11 W&B-exported runs, 11 receipt-declared successful executions and 2 failed executions. The successes comprise 2 preflights, 1 S0 smoke, 2 teacher benchmark diagnostics, 4 tiny S0 base-training runs, and 2 tiny S1 updater-training runs. There are **no exported standalone reconstruction evaluation or oracle runs**. No empty/setup directory was counted as successful. W&B subdirectories are packaging structure, not extra experiments.

“Scientifically interpretable” needs a denominator: **2 bounded numerical-method diagnostics** (R2cpu/R2gpu) are interpretable for sparse/full equivalence; **2 R4 training diagnostics** additionally permit descriptive paired-effect analysis; **0 runs** establish a powered causal claim about base quality, headroom, spectral benefit or routing. None has fully verified executable-source provenance. All receipts retain `scientific_status=NOT_EVALUATED`; this report does not rewrite them.

Every exported JSON was parsed, all YAML metadata inspected, and duplicate records reconciled. All 7 standalone `metrics.json` files equal their receipt metric objects. Both R2 parity/source/service copies agree. All numeric receipt metrics/counts equal the corresponding exported W&B summaries, including nested list elements; there are no missing numeric keys in those summary copies. W&B is a duplicate logging channel here, not independent replication. Its `_step=0`, short runtime and metadata start time describe late summary publication: current CLI starts W&B in `_publish_receipt` after work, so those values are not training duration or optimizer-step counts. W&B `config.yaml` contains `_wandb` metadata/argv, not a resolved scientific configuration.

The export contains **no** `.pt` checkpoints, `weights.json`, `resolved_config.json`, `roles.json`, `split.json`, `stage_runtime.json`, saved dirty patches, `traceback.txt`, `rows.jsonl`, `privileged_oracle.jsonl`, or standalone paired/action evaluation records. Paths mentioned in receipts are evidence of claimed artifacts, not locally inspected artifacts. Recovery should request metadata and checkpoint hashes first; raw patient data need not be committed.

## 3. Complete run inventory

Aliases below are used throughout. Links reference the supplied local evidence folder; it is intentionally not included in this report-only commit. A downstream clone needs the separate export to follow those links. “Reported” denotes a receipt observation rather than independent execution reproduction.

### Table A — execution and scientific inventory

| Run | Stage | Subjects | Mode | Source | Dirty | Metrics | Scientific status |
|---|---|---|---|---|---|---|---|
| [R0a](../experiments_R0_R4/pfgr-lite/R0-20260908T115116Z-473456/receipt.json) — `R0-20260908T115116Z-473456` | preflight | not recorded | metadata preflight | S0-clean | False | target_reads only | preflight only |
| [R0b](../experiments_R0_R4/pfgr-lite/R0-20260908T121953Z-473456/receipt.json) — `R0-20260908T121953Z-473456` | preflight | not recorded | metadata preflight | S1-dirty | True | target_reads only | preflight only |
| [R1a](../experiments_R0_R4/pfgr-lite/R1-real-20260908T115116Z-473456/receipt.json) — `R1-real-20260908T115116Z-473456` | failed before stage receipt | not recorded | S0 requested; device error | S0-clean | False | error only | failed engineering attempt |
| [R1b](../experiments_R0_R4/pfgr-lite/R1-real-20260908T121953Z-473456/receipt.json) — `R1-real-20260908T121953Z-473456` | S0 | 2 | static B2 smoke | S1-dirty | True | training objective/gradients/frozen hashes | successful engineering training smoke; no quality claim |
| [R2cpu](../experiments_R0_R4/pfgr-lite/R2-20260908T121953Z-473456/receipt.json) — `R2-20260908T121953Z-473456` | teacher benchmark | 2 | random K4 trace; sampled parity; CPU | S1-dirty | True | parity/timing aggregates | bounded numerical diagnostic; source caveat |
| [R2gpu-fail](../experiments_R0_R4/pfgr-lite/R2-gpu-20260908T133702Z-473456/receipt.json) — `R2-gpu-20260908T133702Z-473456` | benchmark attempt | not recorded | CUDA transition failure | S2-dirty | True | error only | failed engineering attempt |
| [R2gpu](../experiments_R0_R4/pfgr-lite/R2-gpu-fixed-20260908T134203Z-473456/receipt.json) — `R2-gpu-fixed-20260908T134203Z-473456` | teacher benchmark | 2 | random K4 trace; sampled parity; CUDA | S3-dirty | True | parity/timing aggregates | bounded numerical diagnostic; source caveat |
| [R3-B0](../experiments_R0_R4/pfgr-lite/R3-b0-20260908T140929Z-473456/receipt.json) — `R3-b0-20260908T140929Z-473456` | S0 | 2 | static B0 | S3-dirty | True | training objective/gradients/frozen hashes | successful engineering training smoke; no quality claim |
| [R3-B1](../experiments_R0_R4/pfgr-lite/R3-b1-20260908T140929Z-473456/receipt.json) — `R3-b1-20260908T140929Z-473456` | S0 | 2 | static B1 | S3-dirty | True | training objective/gradients/frozen hashes | successful engineering training smoke; no quality claim |
| [R3-B2](../experiments_R0_R4/pfgr-lite/R3-b2-20260908T140929Z-473456/receipt.json) — `R3-b2-20260908T140929Z-473456` | S0 | 2 | static B2 | S3-dirty | True | training objective/gradients/frozen hashes | successful engineering training smoke; no quality claim |
| [R3-light](../experiments_R0_R4/pfgr-lite/R3-blight-20260908T140929Z-473456/receipt.json) — `R3-blight-20260908T140929Z-473456` | S0 | 2 | static ordered B-light | S3-dirty | True | training objective/gradients/frozen hashes | successful engineering training smoke; no quality claim |
| [R4-U](../experiments_R0_R4/pfgr-lite/R4-u-only-20260908T140929Z-473456/receipt.json) — `R4-u-only-20260908T140929Z-473456` | S1 | 2 | random S1, U-only | S3-dirty | True | loss/gradients/frozen hashes; paired dense training effects | successful engineering training smoke; no quality claim |
| [R4-US](../experiments_R0_R4/pfgr-lite/R4-updater-20260908T140929Z-473456/receipt.json) — `R4-updater-20260908T140929Z-473456` | S1 | 2 | random S1, U+spectral | S3-dirty | True | loss/gradients/frozen hashes; paired dense training effects | successful engineering training smoke; no quality claim |

### Full execution fields (same aliases, one row per execution)

Precision: all argv request `--no-amp`; current MAIN mandates FP32. Only successful R2 explicitly records actual FP32 work. For S0/S1, FP32 is a source/config inference; requested CUDA is not independently certified by an actual-device metric. `N/A` means the field does not apply; `missing` means required evidence is absent. `MN` below means the external MedicalNet path in the exact command ledger; `I/R` means claimed output `inference.pt` and `resume.pt` inside that run's server directory. No output PFGR checkpoint byte SHA was exported.

| Run / scenario | Command | Config hash | Checkpoint input → output | Checkpoint SHA | MedicalNet source channels / adaptation | Device | Precision | Optimizer steps | Route/action mode | K/budget | Teacher mode | ValueNet used? | Oracle used? | Random policy used? | W&B run ID | Exit status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0a | `preflight`; exact argv below | `588f35c9c3e1b39c1676ccc135176f5742310f92033bf5ddb33107649d482ba4` | MN → none | MN requested: afa8055f…; PFGR missing | not exported | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | N/A | N/A | N/A | none | no evidence of V use | no | no | `ujvokv7i` | 0: SOFTWARE_PASS |
| R0b | `preflight`; exact argv below | `588f35c9c3e1b39c1676ccc135176f5742310f92033bf5ddb33107649d482ba4` | MN → none | MN requested: afa8055f…; PFGR missing | not exported | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | N/A | N/A | N/A | none | no evidence of V use | no | no | `cu68ijxm` | 0: SOFTWARE_PASS |
| R1a | `smoke`; exact argv below | `missing` | MN → no output recorded | MN requested: afa8055f…; PFGR missing | not exported | CUDA requested; CPU/CUDA mismatch | no AMP requested; MAIN FP32 | not reached/unknown | N/A | N/A | none; exact dense reconstruction objective | no evidence of V use | no | no | `not exported` | 1: SOFTWARE_FAIL |
| R1b | `smoke`; exact argv below | `702c7eb89c558add7db4aa7e13266206c3dd11f3135287227b495a45aba1dd39` | MN → I/R | MN requested: afa8055f…; PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | N/A | N/A | none; exact dense reconstruction objective | no evidence of V use | no | no | `5crfliaj` | 0: SOFTWARE_PASS |
| R2cpu | `benchmark`; exact argv below | `108ac21905fbfcbdc09a178b82524eb9b149044ace96bb2065212c2853b7f6fb` | R1b/inference.pt → no checkpoint | PFGR missing; MN recorded afa8055f… | 1 → 3, adapted=true (direct) | CPU actual; CUDA requested | FP32 actual | N/A | random forced diagnostic | trace K4; first 2 states/subject labeled | iid_fixed_q, Q64 | no evidence of V use | no | yes | `lnfjnkrc` | 0: SOFTWARE_PASS |
| R2gpu-fail | `benchmark`; exact argv below | `missing` | R1b/inference.pt → no checkpoint | PFGR missing | not exported | CUDA requested; CPU/CUDA mismatch | no AMP requested; MAIN FP32 | N/A | N/A | N/A | requested iid_fixed_q/Q64; failed | no evidence of V use | no | not reached/unknown | `not exported` | 1: SOFTWARE_FAIL |
| R2gpu | `benchmark`; exact argv below | `108ac21905fbfcbdc09a178b82524eb9b149044ace96bb2065212c2853b7f6fb` | R1b/inference.pt → no checkpoint | PFGR missing; MN recorded afa8055f… | 1 → 3, adapted=true (direct) | cuda:0 actual | FP32 actual | N/A | random forced diagnostic | trace K4; first 2 states/subject labeled | iid_fixed_q, Q64 | no evidence of V use | no | yes | `w8h0fnpc` | 0: SOFTWARE_PASS |
| R3-B0 | `static-train`; exact argv below | `67b09eb3809ba20f8cacf0d7ae1d6284053778659c8dc4e8f70f377838104997` | MN → I/R | MN requested: afa8055f…; PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | N/A | N/A | none; exact dense reconstruction objective | no evidence of V use | no | no | `khdjdvva` | 0: SOFTWARE_PASS |
| R3-B1 | `static-train`; exact argv below | `bb5956fb37fb702f7ab19899b006c02107799dc567d8e3f1bfab75abee3f189e` | MN → I/R | MN requested: afa8055f…; PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | N/A | N/A | none; exact dense reconstruction objective | no evidence of V use | no | no | `2j9zk9hs` | 0: SOFTWARE_PASS |
| R3-B2 | `static-train`; exact argv below | `702c7eb89c558add7db4aa7e13266206c3dd11f3135287227b495a45aba1dd39` | MN → I/R | MN requested: afa8055f…; PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | N/A | N/A | none; exact dense reconstruction objective | no evidence of V use | no | no | `cofa61n2` | 0: SOFTWARE_PASS |
| R3-light | `static-train`; exact argv below | `b9c62ea7294ff3ab679c1f4c0873dfcf0ca4b3393fc5e92c6b12d676f5d8d8f2` | MN → I/R | MN requested: afa8055f…; PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | N/A | N/A | none; exact dense reconstruction objective | no evidence of V use | no | no | `xvev57r4` | 0: SOFTWARE_PASS |
| R4-U | `updater-train`; exact argv below | `702c7eb89c558add7db4aa7e13266206c3dd11f3135287227b495a45aba1dd39` | R3-B2/inference.pt → I/R | PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | random target-free training | sampled K1 once, K2 once | none; exact dense reconstruction objective | no evidence of V use | no | yes | `jynsup8x` | 0: SOFTWARE_PASS |
| R4-US | `updater-train`; exact argv below | `702c7eb89c558add7db4aa7e13266206c3dd11f3135287227b495a45aba1dd39` | R3-B2/inference.pt → I/R | PFGR missing | not exported; lineage inference only | CUDA requested; no tensor-device receipt | no AMP requested; MAIN FP32 | 2 | random target-free training | sampled K1 once, K2 once | none; exact dense reconstruction objective | no evidence of V use | no | yes | `zwbcsaly` | 0: SOFTWARE_PASS |

Subjects in all successful S0/S1 `metrics.json.history`: `BraTS2021_00002`, `BraTS2021_00005`. R2 `benchmark.json.source_receipt.sample_ids`: `BraTS2021_00000`, `BraTS2021_00016`. These distinct IDs support separate observed training/benchmark samples, but missing split/roles prevent complete membership or related-subject disjointness verification. Outer training receipts say `role=validation`; current `_production_inputs` actually selects `producer_fit` for S0/S1/S2. Treat this as misleading generic receipt metadata, not proof of validation-target training and not proof of a validated split.

### Exact command ledger

Each command is reconstructed verbatim from `receipt.json.argv`, prefixed with the W&B-recorded Python/program where available (same interpreter in receipt environment otherwise). Original absolute server paths are preserved. These are historical commands, **not instructions to execute now**. No missing checkpoint SHA, LR, seed override or config content has been added.

<details><summary>R0a — R0-20260908T115116Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite preflight --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --write-roles --run-name R0-20260908T115116Z-473456
```

</details>

<details><summary>R0b — R0-20260908T121953Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite preflight --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --write-roles --run-name R0-20260908T121953Z-473456
```

</details>

<details><summary>R1a — R1-real-20260908T115116Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite smoke --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T115116Z-473456/roles.json --run-name R1-real-20260908T115116Z-473456 --max-subjects 2 --max-steps 2
```

</details>

<details><summary>R1b — R1-real-20260908T121953Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite smoke --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --run-name R1-real-20260908T121953Z-473456 --max-subjects 2 --max-steps 2
```

</details>

<details><summary>R2cpu — R2-20260908T121953Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite benchmark --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --checkpoint /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R1-real-20260908T121953Z-473456/inference.pt --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R2-20260908T121953Z-473456 --max-subjects 2 --max-states 2 --candidate-count 4 --teacher-mode iid_fixed_q --query-count 64 --repeats 3
```

</details>

<details><summary>R2gpu-fail — R2-gpu-20260908T133702Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite benchmark --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --checkpoint /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R1-real-20260908T121953Z-473456/inference.pt --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R2-gpu-20260908T133702Z-473456 --max-subjects 2 --max-states 2 --candidate-count 4 --teacher-mode iid_fixed_q --query-count 64 --repeats 3
```

</details>

<details><summary>R2gpu — R2-gpu-fixed-20260908T134203Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite benchmark --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --checkpoint /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R1-real-20260908T121953Z-473456/inference.pt --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R2-gpu-fixed-20260908T134203Z-473456 --max-subjects 2 --max-states 2 --candidate-count 4 --teacher-mode iid_fixed_q --query-count 64 --repeats 3
```

</details>

<details><summary>R3-B0 — R3-b0-20260908T140929Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite static-train --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --base b0 --epochs 1 --max-subjects 2 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R3-b0-20260908T140929Z-473456
```

</details>

<details><summary>R3-B1 — R3-b1-20260908T140929Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite static-train --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --base b1 --epochs 1 --max-subjects 2 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R3-b1-20260908T140929Z-473456
```

</details>

<details><summary>R3-B2 — R3-b2-20260908T140929Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite static-train --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --base b2 --epochs 1 --max-subjects 2 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R3-b2-20260908T140929Z-473456
```

</details>

<details><summary>R3-light — R3-blight-20260908T140929Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite static-train --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --medicalnet-checkpoint /home/aidev/workspace/quockhanh/3DGS/checkpoints/medicalnet/resnet_10_23dataset.pth --medicalnet-sha256 afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605 --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --base b_light --epochs 1 --max-subjects 2 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R3-blight-20260908T140929Z-473456
```

</details>

<details><summary>R4-U — R4-u-only-20260908T140929Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite updater-train --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --checkpoint /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R3-b2-20260908T140929Z-473456/inference.pt --spectral-arm u_only --epochs 1 --max-subjects 2 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R4-u-only-20260908T140929Z-473456
```

</details>

<details><summary>R4-US — R4-updater-20260908T140929Z-473456</summary>

```bash
/home/aidev/miniconda3/envs/smagm-a4000/bin/python -m smagm.cli.pfgr_lite updater-train --config /home/aidev/workspace/quockhanh/3DGS-pfgr-run/configs/pfgr_lite/main.json --data-root /home/aidev/workspace/quockhanh/3DGS/data/preprocessed/BraTS21 --split-file /home/aidev/workspace/quockhanh/3DGS/experiments/runs/point-guided-gpu1-e10-fp32-20260824-230240/split.json --output-root /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite --device cuda --no-amp --roles-file /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R0-20260908T121953Z-473456/roles.json --checkpoint /home/aidev/workspace/quockhanh/3DGS/experiments/pfgr-lite/R3-b2-20260908T140929Z-473456/inference.pt --spectral-arm u_plus_spectral --epochs 1 --max-subjects 2 --wandb --wandb-entity khanhlq-work-hanoi-university-of-science-and-technology --wandb-project smagm-point-guided --run-name R4-updater-20260908T140929Z-473456
```

</details>

## 4. R0 interpretation

**FACT — R0a/R0b receipts:** both execute non-synthetic `preflight --write-roles`, declare software success and no target reads. Neither includes the promised weights, split, roles, environment sidecar or resolved config. R0b is explicitly dirty. There is no exported R0 test-output or `runbook-check` execution. Thus “R0 complete” would overstate what this export contains.

**SOURCE EVIDENCE — current CLI `_preflight` (2611–2677):** validates required paths/device availability and config, loads the baseline split, optionally derives training roles, strictly loads the supplied MedicalNet checkpoint into a three-channel backbone, and writes integrity/adaptation information. It does not run MRI encoding, validate every volume, test useful reconstruction, fit a policy, or certify an adaptive release. `_input_missing` and `--no-amp` checks are configuration/environment checks. Receipt target_reads is a declared preflight counter, not a target-access instrumentation trace.

| Does exported R0 prove… | YES / NO / PARTIAL | Reason |
|---|---|---|
| Software integrity | **PARTIAL** | Preflight success recorded; scoped source match for R0a. Full source, promised sidecars and test records missing. The subsequent R1 failure disproves any inference that preflight guarantees GPU pipeline execution. |
| MedicalNet official-pretrained provenance | **NO** | Filename/expected digest insufficient; no R0 weights receipt; downstream R2 explicitly says official verification false. |
| Real-data validity | **PARTIAL** | Real-path/non-synthetic intent and accepted split loading; no exported affine/modality/mask/normalization/subject manifest validation. Preflight does not inspect the image corpus. |
| Scientific correctness | **NO** | No tested causal contrast, headroom, convergence or held-out quality. |

The intended target boundary is compatible with preflight: no T1ce should be loaded. Role creation is deterministic partitioning of a supplied split, not human approval of that split. No signed Human Gate receipt is exported; none is needed merely to describe R0, and no R0 success substitutes for later calibration/final-evaluation review. Clean-tree prerequisites in the runbook were not met by R0b and the later dirty runs.

## 5. R1 interpretation

### R1a: failed engineering execution

**FACT — R1a/receipt.json:** exit 1, no completed stage, CPU/CUDA mismatch at `wrapper_CUDA__equal`; only a relative `traceback_path=traceback.txt`, whose file is missing. Do not count requested subject/step limits as completed work.

**SOURCE EVIDENCE / high-confidence root-cause inference:** current S0 moves observations/mask to `options.device` (`stages.py:_default_static_step`), while the deferred target loader retains the CPU sample mask (`data.py:_DeferredSupervision`). `teacher.py:_normalise_mask`, lines 89–120, accepts a `device` argument but returns a boolean clone without moving it; its numeric branch also casts dtype without moving. `validate_target` then compares this supplied mask with the CUDA observation-context mask at line 596. This chain explains precisely the reported CPU/CUDA equality failure. Its PFGR scoped bytes match R1a, but the absent traceback prevents claiming the historical stack frame was independently recovered.

**Fixed later?** R1b succeeds under S1-dirty, so a later execution gets through the failure. **Fix present on current main? NO for the identified mask-device defect:** `_normalise_mask` still ignores its device argument for supplied tensors. The server's exact repair is unknown; a hash change plus successful rerun cannot identify the patch. Do not silently edit this defect during this audit.

### R1b: successful S0 optimization smoke

**FACT — R1b/metrics.json and receipt:** S0, the two training IDs in section 3, two optimizer/gradient steps, completed=true. Authorized modules are static head, B projector and decoder. Each has nonzero measured gradients on both steps and changed parameter tensors; backbone, semantic head, point refiner and spectral projector before/after hashes agree. The static variant is B2 by current main config/command semantics, with no explicit `--base` override; missing resolved config limits independent reconstruction of that selection.

Training objectives are `0.26420876383781433` then `0.22051088511943817` in `history`; the subjects differ, so the decline is not a same-subject before/after learning curve. The objective combines Charbonnier/SSIM/gradient terms with coefficients `1.0/0.2/0.1`, recorded in `loss_components`. Maximum module gradient norms are static `0.21785494685173035`, B `0.00043361852294765413`, D `0.813970685005188` (same file). These demonstrate an active optimization graph, not useful generalization.

T1ce joins after target-free prediction in the audited source; the record reports two deferred target reads and zero segmentation reads. `observation_reads=0` is explicitly scoped to the **stage delta** after factory loading and does not mean the model saw no MRI. `artifact_context_traversal_count=1` is a post-stage artifact-context traversal, not the total number of backbone calls across training. Claimed `inference.pt`, `resume.pt`, and `stage_runtime.json` are absent from the export. The run proves reported bounded execution and parameter movement; it does **not** prove useful reconstruction learning, a reproducible saved checkpoint, or a validation improvement.

## 6. R2 correctness and performance

### Correctness verdict

**NUMERICALLY EQUIVALENT on the reported FP32 query work; based on EXACT ALGEBRA.** For fixed action/write W and bilinear query S, `S(Z+Wδ)=S(Z)+S(Wδ)`. The sparse engine passes that feature vector through the same nonlinear decoder D; it is not a Taylor approximation of D. Source: [sparse_write.py](../src/smagm/features/point_guided/pfgr_lite/sparse_write.py), `query_write_delta` (653–709) and `reference_full_write` (761–806); [benchmark.py](../src/smagm/features/point_guided/pfgr_lite/benchmark.py), `_one_parity_case` (249–350). The reference creates actual legacy compact writes in cloned planes, independently of sparse-query coefficients, then queries those written planes at the same IDs.

**Important qualification:** these real-path R2 runs use `iid_fixed_q`, not full-footprint enumeration. Their **gain estimator is APPROXIMATE as a finite Monte Carlo estimate of global gain**, even though the optimized and reference computations on its identical draws are numerically equivalent. They do not establish full-volume gain accuracy, sampling variance/sign reliability, top-action ranking or exhaustive footprint completeness on these subjects. Correct arithmetic for a sampled label is distinct from an accurate scientific label.

### Table B — R2 paths, errors and latency

Sources for every number in this table: the indicated run's `parity.json` and `benchmark.json.timing/options/source_receipt`; speed ratios are reference/optimized arithmetic recomputed from those means.

| R2 path | Error | Latency | Workload | Verdict |
|---|---|---|---|---|
| R2cpu full-write reference | Compared with sparse below | 4.735696362 ms mean method-specific | 2 subjects × 2 states × 4 candidates = 16 cases; 3 repeats = 48 rows; Q64, FP32, actual CPU | Valid reported reference on sampled IDs; not GPU evidence |
| R2cpu sparse query-delta | query max `1.1920928955078125e-7`; prediction max `2.9802322387695312e-8`; gain max `3.51284279442865e-11`; 0 failures | 2.631662724 ms; ratio **1.799507×**; shared-before 2.182420886 ms | Same actions, states and fixed draws as reference; chunk 1024; candidate chunk 1 | Reported parity PASS; stable performance unestablished |
| R2gpu-fail | CPU `FloatTensor` input versus CUDA weight; no parity output | Missing | Same requested bounds; actual completed work unknown | Failed engineering device transition; not a negative parity result |
| R2gpu full-write reference | Compared with sparse below | 5.714556241 ms mean method-specific | Same declared 16 cases / 48 repeated rows / Q64; FP32, actual `cuda:0` | Valid reported CUDA reference on sampled IDs |
| R2gpu sparse query-delta | query max `1.1920928955078125e-7`; prediction max `2.9802322387695312e-8`; gain max `4.4497738826976274e-11`; 0 failures | 3.220779006 ms; ratio **1.774278×**; shared-before 2.621143513 ms | Same work within this run | Reported CUDA parity PASS; stable performance unestablished |

Both files declare absolute/relative FP32 tolerances `1e-6/1e-5`. Top-1 agreement, Spearman/ranking agreement and near-tie cases are **not exported and not computed by this benchmark's summary logic**. Small absolute gain error does not guarantee ranking when action gaps are smaller. FP64, FP16/BF16/AMP, correction-gradient parity and cross-device tensor-by-tensor parity are not R2 export results. Existing `test_sparse_parity.py` covers independent full-write/query and gradient checks in FP32/FP64 on synthetic CPU fixtures; it was inspected, not rerun or relabeled as server evidence.

CPU and GPU share the same producer compatibility digest and subject/context IDs, but their **per-subject initialization hashes differ** (`benchmark.json.source_receipt.subject_initialization_hashes`). Same-work is established separately within each run, not bitwise CPU-versus-GPU reproduction. Different floating execution or dirty source may explain those differing initialization hashes; the export cannot isolate the cause.

### Device failures and current-main behavior

R2cpu's receipt requests CUDA while `benchmark.json.source_receipt.actual_devices=["cpu"]`; use the latter for measured work. Current `_production_inputs` resolves stage options from JSON, and `_config_for_command` overrides `options.device` only when a stage is supplied. Benchmark/evaluation services call it without one. Current `main.json.stage_options.device` is CPU, so `--device cuda` does not reliably control their actual device. This is an **IMPLEMENTED WITH WRONG SEMANTICS** operational path, still in main.

The next R2 attempt fails after some device change with CPU input/CUDA weights. Current `experiments.py:_context_for_sample` (560–593) passes CPU-loaded observations and mask directly to `model.encode_observations`, without moving them to the model device. `build_stage_inputs` does move the model. These are concrete current-main defects consistent with the failed transition. R2gpu's later receipt proves a CUDA execution succeeded under another dirty scope, not that either fix landed in main. Recover the exact patch and first traceback before assigning a historical code-level fix.

### Performance verdict

**INCONCLUSIVE for stable speedup and representative teacher throughput; encouraging method-specific savings in this tiny workload.** Do not fail the method merely because these ratios are below the proposed ≥4× target. The root research attachment's E7 proposes 32 saved states and 32 candidates, whereas this run has only the workload in Table B. Candidate count and Q are small; the global mask denominator is not necessarily small (do not conflate candidate M in the older plan with voxel-mask M in the teacher equation).

The aggregate combines cold/warm-labeled repeats. Raw `rows.jsonl` is missing, so dispersion, p50/p95, paired timing intervals, warm-only means, cache-build/validation times, GPU allocations/peaks, full-plane clone bytes, unique queries, support size and workload dimensions cannot be recovered. GPU model name is missing; the environment name `smagm-a4000` is not hardware verification. CUDA availability/device count in a receipt does not identify the accelerator.

Current benchmark synchronization uses `torch.cuda.synchronize(device)` around `perf_counter` method measurements. It times shared-before query/D separately, then optimized, then reference, in a fixed order. It releases a lattice cache entry for each case's first repeat, but the shared-before query precedes both method-specific windows. Thus “cold” is not an independently cold end-to-end teacher comparison. Footprint construction and target gathering occur outside those method-specific timers. Summed shared-before+method means give only **1.437058× CPU / 1.426876× CUDA** here (derived from the corresponding Table B source means); they still omit other pipeline costs. Do not quote the larger ratios as end-to-end acceleration.

**HYPOTHESIS:** launch/synchronization, Python validation/hashing and other fixed costs may dominate Q64, especially with serial single-candidate evaluation. Larger candidate/state batches could amortize setup if the implementation actually reuses it; merely looping more cases may repeat the same overhead. Larger Q increases common nonlinear-decoder work and can **reduce** relative speedup; larger planes increase avoided clone/query costs and may improve it. Neither monotonic scaling nor overhead dominance is measured here. The current serial benchmark does not prove a batched sparse evaluator was exercised.

**RECOMMENDATION — next benchmark, after source recovery:** use 32 frozen state snapshots distributed over 8 development subjects, 32 predeclared candidates/state, FP32 on the actual named CUDA device, Q1024 fixed draws, identical reference/sparse actions and decoder, fixed chunks; retain a Q64 control on the same saved work to expose scaling. Use 5 fresh-process paired timing repetitions with balanced method order and 10 warm repeats after separately recorded setup/warm-up. These are proposed bounds, not a time guarantee. Save every row, actual state/action/query IDs or replayable identities, probabilities/seeds, dtype/device, sync method, setup/footprint/query/D times, p50/p95, subject/state-clustered variability, clone bytes, allocated/reserved peak memory and complete teacher/label throughput. Add exact-footprint/full-volume checks on a bounded subset and explicit ranking/near-tie agreement. Recover old rows first; do not rerun correct old cases merely to replenish files that still exist on the server.

## 7. MedicalNet adaptation evidence

**FACT — direct evidence is narrower than “every R1–R4 receipt”:** R2cpu and R2gpu contain the complete adaptation tuple in `benchmark.json.source_receipt.contexts[*].source_provenance`, also duplicated in `service_receipt.json` and top-level receipt metrics:

| Field | Recorded value in both successful R2 runs |
|---|---|
| source_input_channels / adapted_input_channels / input_conv_adapted | `1 / 3 / true` |
| checkpoint SHA256 | `afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605` |
| checkpoint_integrity_verified / official_pretrained_verified | `true / false` |
| synthetic_untrained | `true` |
| source/loaded backbone key count | `72 / 72` |
| adaptation digest | `393794407b20234d558324b1fee3a7324b2b83fefd667c8b915354db791fed9e` |
| parameter hash | `d667de682155d13ed9129a474d0b3c7106250d4b524c0311e34168c01185f04c` |
| frozen BN hash | `2eba1b67065c5ad8609c8b2bd5ddd7257725441d14edc5d1db5f236391cfd687` |
| traversal count per context | `1` |

This converts the prior **conditional** adaptation observation into a **confirmed reported property of these R2 executions**, including the R1b checkpoint lineage they load. R1b's own exported receipt does not include that tuple. R3 requests the same MN path/expected digest; R4 loads R3-B2, and their backbone frozen-hash entries match the R1 S0 record. This is consistent lineage evidence for the same adaptation across R1/R3/R4, but their missing weights and bundle provenance mean it is **not independently verified for every run**. No adaptation fact is inferred for failed attempts from a filename alone.

**SOURCE EVIDENCE:** `medicalnet_resnet10.py:adapt_medicalnet_input_conv_weight` repeats the one-channel kernel and divides by the adapted channel count. In exact arithmetic on already normalized observations,

`Conv(W repeated/3, [T1,T2,FLAIR]) = Conv(W, (T1+T2+FLAIR)/3)`.

With frozen downstream backbone/eval statistics, its feature maps depend on that mean. This does **not** imply whole-model permutation invariance: ordered raw observations reach the point refiner, and B2/B-light explicitly condition static Z0 on ordered source channels. Reordering channels before distinct preprocessing policies is also a different claim from permutation of the normalized tensor. B0/B1 provide no ordered-source bypass in their static head under the adapted stem; B2/B-light do. The entire routed model can remain modality-sensitive through point locations and static conditioning.

A source-conditioned static control is scientifically justified because a function of a channel mean cannot recover arbitrary contrast identity. Whether that information improves T1ce prediction is still untested. “Official pretrained false” is a provenance blocker, not proof that the file contains random tensors: `provenance.py:source_provenance_from_semantic_prior` sets `synthetic_untrained = not official`. The code conservatively conflates unverified source with synthetic/untrained capability. Interpret it as an unverified-pretraining flag unless the actual file acquisition provenance is recovered.

## 8. R3 base-control interpretation

### Table C — architecture, capacity and modality controls

Capacity below is a **current-main source count**, independently obtained by constructing only `StaticSynthesisHead` and summing parameters (no forward/training/data access). It is not an exported checkpoint measurement. Architecture source: `static_synthesis.py` and frozen plan section 5; train/validation source: each R3 `metrics.json` and `receipt.json`.

| Base arm | Capacity | Ordered modalities | Training evidence | Validation evidence |
|---|---|---|---|---|
| B0 | 2,080 static-head parameters; B projector plus shared 64→32 map | No direct ordered-source head; MedicalNet mean path under adapted stem | 2 subjects, 1 epoch, 2 updates; all authorized modules receive gradients | None |
| B1 | 488,681 static-head parameters; three existing feature scales, 67/67/515→64 projections, axis collapse, two residual blocks/scale, physical resampling and sum fusion | Three reserved source slots/scale explicitly zero | Same bounded work; active gradients and parameter movement | None |
| B2 | 488,681, same architecture/registered capacity as B1 | Ordered source resampled at each live feature lattice before projection | Same bounded work; active gradients and parameter movement | None |
| B-light | 80,288; B64 + three separate ordered source plane means, shared 67→64 projection, one residual block, shared 64→32 | Yes; averages spatially over omitted axis, not across modality channels | Same bounded work; active gradients and parameter movement | None |

All use the same locked decoder architecture `96→64→32→1` with SiLU, trained separately with the arm; MAIN does not include the older report's wider-decoder proposal. B1/B2 match registered parameter count and tensor width; B1's `3 scales × 3 source channels × 64 outputs = 576` input weights have no data signal because those source slots are zero. The property's name `effective_parameter_count` actually counts all registered parameters, so it does not subtract dormant B1 weights. Disclose this intended information-control distinction. B0 and B-light are intentionally not matched in capacity to B2.

Current source seeds initialization before model construction (`stages.py:build_stage_inputs`), zeros final residual convolutions and collapse logits, but does not zero the entire static head. B1/B2 allocate identical modules, permitting paired initialization under the same seed. Different-arm construction can consume RNG differently before the shared decoder is initialized; same numeric seed alone is not proof of identical initial D weights for all four arms. Their actual resolved seed, initialization hashes and checkpoint tensors are missing. Record them in the fair pilot, and compare training work and convergence as well as capacity.

**FACT — R3 training values, each `metrics.json.history`:**

| Run | Objective on first training subject | Objective on second training subject / reported final loss | Changed static parameter tensors | Max static gradient L2 |
|---|---:|---:|---:|---:|
| R3-B0 | 0.769942045211792 | 0.702970564365387 | 2 | 0.10795077681541443 |
| R3-B1 | 0.26446428894996643 | 0.21993452310562134 | 50 | 0.21691206097602844 |
| R3-B2 | 0.26420876383781433 | 0.22051088511943817 | 50 | 0.21785496175289154 |
| R3-light | 0.42392751574516296 | 0.3058571219444275 | 8 | 0.16305260360240936 |

The changed counts refer to parameter **tensors**, not scalar parameters, capacities or update norms. All R3 metrics show nonzero static/B/D gradient evidence on both steps and unchanged backbone/semantic/refiner/spectral hashes. The loss includes the same Charbonnier + SSIM + gradient coefficients reported in R1. R1b and R3-B2 have identical rounded/stored objective values and nearly identical gradient values; they are separate receipts/W&B executions of closely repeated work, not independent scientific replications or evidence of further training.

A. **SOFTWARE: reported YES for all four bounded arms; independent exact-source reproduction remains partial.** No exported evidence of an arm failing its optimization graph.

B. **OPTIMIZATION: YES for gradient flow and parameter changes; NO for convergence or a demonstrated same-subject improvement curve.** Each history step uses a different subject. Whole-model/module state before/after alone does not establish good conditioning or useful representation.

C. **SCIENTIFIC: NO ranking conclusion.** There is no held-out MAE/PSNR/SSIM/Charbonnier, repeated-seed ranking, convergence curve, initial/final common evaluation, or selected-base review. B2 was used downstream by filename; this is not evidence that B2 was best. The narrow B1/B2 loss difference is not modality benefit.

D. **MODALITY: architecture preserves ordered information in B2/B-light; reconstruction benefit NOT YET TESTED.** Compare B1/B2 after a fair convergence pilot with paired initial weights, common data/order/objective and validation protocol. B-light is a useful simpler competitor, not the independently tuned strong static B3 proposed in the research attachment. Their roles must not be conflated in a paper.

The minimum ranking pilot is section 16 NEXT-2, not automatic full-dataset training. A selected strong Z0 must receive sufficient optimization to make its validation ranking stable; a weak base creates artificial correction headroom.

## 9. R4 actual semantics

**FACT — R4-U/R4-US exact commands and metrics:** both execute `updater-train`, stage S1, loading the same **named R3-B2/inference.pt**; neither executes `evaluate` or `oracle-evaluate`. R4-U uses `u_only`, R4-US `u_plus_spectral`; both have the same two training subjects, one epoch and two optimizer steps. The base/B/D frozen hashes and paired Z0 values match between arms. Because the input file's byte SHA is absent, same path and component hashes support common-base inference without proving identical whole checkpoint bytes.

**SOURCE EVIDENCE:** `_run_s1` samples K uniformly from `{1,2,4}` with a seeded Python RNG. `_default_random_route` proposes current U corrections, chooses uniformly from writer-support-eligible candidates, with replacement across steps, and refreshes proposals after each applied write. No ValueNet is consulted. A complete differentiable target-free route is built before the deferred target join. Then the objective updates U and, in MAIN, the shared spectral projector; all other producer modules and D weights remain frozen, with D retaining input derivatives.

For each actual run, `sampled_k_counts={1:1,2:1,4:0}`, `route_count=2`, `executed_writes=3`, `behavior_states=3`, `candidate_proposals=6144`, and `value_evaluations=0` (`metrics.json`). The proposal count is dense candidate generation at the visited decision states, not 6144 supervised actions, independent observations, or unique locations. No exported selected action IDs allow exact coverage reconstruction. There are no K4 samples.

| Required headroom element | R4 export |
|---|---|
| Frozen producer evaluation after training | Missing; metrics are from live S1 forwards before each optimizer update |
| U-only / U+spectral | Both training arms present |
| Target-free candidate choice | Supported by current source; random-route metric labels agree; exact dirty implementation unavailable |
| True signed effect | Dense paired Z0/final difference available for the realized training route |
| Explicit no-op control | Missing; Z0 baseline is present, but no independent no-op pipeline check |
| Random baseline | Random behavior during training only; no repeated frozen final-checkpoint evaluation |
| Oracle action / sampled Oracle-1 / greedy Oracle-2 or Oracle-4 | All missing |
| Independent winner confirmation | Missing |
| Learned policy / ValueNet | Not used |
| Strong Z0 selected by validation | Missing |

The **current runbook as a whole implements both updater training and separate headroom evaluation commands** (R4 lines 414–562). It does not claim that `updater-train` itself runs an oracle. The **exported execution stops after its training subsection**. Therefore labeling these folders “R4 passed/headroom passed” conflates engineering training with the scientific gate. The runbook's adjacent R5 prerequisite only names a trained U+spectral artifact and its shell check tests file existence; it should explicitly require an accepted paired selection-headroom result before scientific R5. This is a transition/governance gap, not a reason to say the oracle service is absent.

A further causal mismatch: frozen plan section 7 requires a meaningful U-only comparison to branch from a **verified trained U+spectral warm-up**. The runbook instead branches both runs from S0, and the exported U-only arm freezes the S0 spectral projector. It is a cold/frozen-projector engineering control. It is not a matched trained-spectral ownership ablation; “U-only” also does not mean spectral features are removed from U's inputs.

## 10. R4 quantitative evidence

The following tables are derived directly from `R4-U/metrics.json` and `R4-US/metrics.json`, `paired_dense_metrics` and their aggregates. Here positive gain is before−after for error metrics and after−before for quality metrics. “Final” means the last state of that **training forward**, not evaluation of the final saved optimizer checkpoint. At the first row U has received no S1 update; at the second it has received one. The second row is measured before the second optimizer update.

### R4-U: exact paired training-forward values

| Subject / K / stage update | Metric | Z0 | Final route state | Signed improvement |
|---|---|---:|---:|---:|
| `BraTS2021_00002` / 1 / 1 | masked_charbonnier | 0.11184063089840131 | 0.11184077640666117 | -1.4550825985781835e-07 |
| `BraTS2021_00002` / 1 / 1 | mae | 0.11182040821326729 | 0.11182055360620101 | -1.4539293372028972e-07 |
| `BraTS2021_00002` / 1 / 1 | psnr | 15.843354737824434 | 15.843345325020993 | -9.4128034415774664e-06 |
| `BraTS2021_00002` / 1 / 1 | ssim | 0.11989673704430993 | 0.11989465072333427 | -2.0863209756610823e-06 |
| `BraTS2021_00005` / 2 / 2 | masked_charbonnier | 0.092089533004324473 | 0.09208954056531829 | -7.5609938171572466e-09 |
| `BraTS2021_00005` / 2 / 2 | mae | 0.092065431377912926 | 0.092065439947805999 | -8.5698930729627421e-09 |
| `BraTS2021_00005` / 2 / 2 | psnr | 16.971829162610877 | 16.971827211226202 | -1.9513846751806341e-06 |
| `BraTS2021_00005` / 2 / 2 | ssim | 0.17778084492298069 | 0.17778153542011405 | 6.9049713335989082e-07 |

| Subject-mean metric | Z0 | Final | Signed improvement |
|---|---:|---:|---:|
| masked_charbonnier | 0.10196508195136289 | 0.10196515848598972 | -7.6534626837487796e-08 |
| mae | 0.10194291979559011 | 0.1019429967770035 | -7.698141339662623e-08 |
| psnr | 16.407591950217657 | 16.407586268123598 | -5.6820940583790502e-06 |
| ssim | 0.1488387909836453 | 0.14883809307172416 | -6.9791192115059575e-07 |

### R4-US: exact paired training-forward values

| Subject / K / stage update | Metric | Z0 | Final route state | Signed improvement |
|---|---|---:|---:|---:|
| `BraTS2021_00002` / 1 / 1 | masked_charbonnier | 0.11184063089840131 | 0.11184077640666117 | -1.4550825985781835e-07 |
| `BraTS2021_00002` / 1 / 1 | mae | 0.11182040821326729 | 0.11182055360620101 | -1.4539293372028972e-07 |
| `BraTS2021_00002` / 1 / 1 | psnr | 15.843354737824434 | 15.843345325020993 | -9.4128034415774664e-06 |
| `BraTS2021_00002` / 1 / 1 | ssim | 0.11989673704430993 | 0.11989465072333427 | -2.0863209756610823e-06 |
| `BraTS2021_00005` / 2 / 2 | masked_charbonnier | 0.092089533004324473 | 0.092089541486561316 | -8.482236843465607e-09 |
| `BraTS2021_00005` / 2 / 2 | mae | 0.092065431377912926 | 0.092065440874100998 | -9.4961880720001446e-09 |
| `BraTS2021_00005` / 2 / 2 | psnr | 16.971829162610877 | 16.971827164919407 | -1.9976914700237103e-06 |
| `BraTS2021_00005` / 2 / 2 | ssim | 0.17778084492298069 | 0.17778153323274232 | 6.8830976163170021e-07 |

| Subject-mean metric | Z0 | Final | Signed improvement |
|---|---:|---:|---:|
| masked_charbonnier | 0.10196508195136289 | 0.10196515894661123 | -7.6995248350641976e-08 |
| mae | 0.10194291979559011 | 0.101942997240151 | -7.7444560896144932e-08 |
| psnr | 16.407591950217657 | 16.407586244970201 | -5.7052474558005883e-06 |
| ssim | 0.1488387909836453 | 0.14883809197803829 | -6.9900560701469105e-07 |

Both runs record data range `1.0`, Charbonnier epsilon `0.001`, SSIM window `11`, observation-mask reductions and mask voxel counts `1,392,388` and `1,293,736` for the two subjects (`paired_dense_metrics[*].before/after`). Those counts are denominators, not independent sample sizes. The formula identifier contains “global-ssim”; window/mask metadata also specify the implemented window treatment. Retain that complete definition rather than comparing these SSIM values with another protocol. Aggregate means are equal-subject means, not pooled-voxel errors or PSNR from a pooled MSE.

**INFERENCE:** both recorded routes slightly worsen mean Charbonnier, MAE and PSNR, with mixed per-subject SSIM changes. At the displayed scale they are **effectively no-op in observed reconstruction**, with tiny adverse signed error changes; they do not establish clear material harm or useful correction. General correction capacity remains **unresolved because the experiment is too small and lacks post-training evaluation**. Averaging the two distinct training states is descriptive only, not an estimate of frozen-producer generalization. Do not conduct a significance claim or arm ranking from these rows.

The identical first-subject outputs are consistent with the shared S0 initialization before either S1 optimizer has stepped. The later tiny arm differences are consistent with spectral optimization starting to diverge; they do not establish spectral value. The objective logged at K2 averages intermediate/final Charbonnier terms, whereas the paired table measures the terminal state; small disagreement between `history.objective` and terminal Charbonnier is not automatically an inconsistency. Floating reduction precision can also differ.

No update-vector L2/L∞, plane-write L2, decoded correction norms, saturation fractions, distribution by location, or per-action local/global gain rows are exported. Changed-tensor counts and gradient norms cannot reconstruct those magnitudes. A bounded write architecture alone does not prove saturation or insufficient capacity.

## 11. Headroom decision matrix

### Table D — signals actually observed

| Headroom signal | Observed? | Interpretation |
|---|---|---|
| Dense Z0 versus realized random training route | Yes, R4 paired rows | Tiny effects for immature, changing U; useful for mechanics only |
| Frozen trained random−Z0 | No | Useful sparse correction in deployment is unestablished |
| No-op−Z0 within measurement tolerance | No independent no-op run | Baseline available; no-op pipeline consistency still needs execution |
| Confirmed Oracle−Z0 | No | Cannot judge current-U attainable correction headroom |
| Confirmed Oracle−Random | No | Cannot authorize ValueBank/router work |
| Learned−Random / oracle-gap recovery | No | Learned allocation not tested |
| Same-action local versus full-footprint gain | No R0–R4 record | Real-data necessity/prevalence remains untested; prior synthetic counterexamples are separate |

| Case | Necessary pattern on one frozen producer/cohort | Scientific interpretation / action | Supported now? |
|---|---|---|---|
| A | Oracle≈no-op, Random≈no-op, with adequate precision and candidate coverage | This trained U/action family/base offers little measured headroom; examine U/base/write family or stop routing research | **No**: oracle absent, current U untrained beyond smoke |
| B | Random improves Z0; Oracle≈Random under equivalence evidence | Sparse correction useful; adaptive selection may be unnecessary; retain fixed/random/parallel method candidate | **No** |
| C | Oracle>Random, Learned≈Random | Correction and selection headroom exist; diagnose bank/V/ranking | **No** |
| D | Learned>Random and approaches Oracle while improving Z0 | Learned allocation has value; then study calibration/STOP/sequential versus parallel | **No** |

**Current classification: INCONCLUSIVE.** Random≈no-op is one observed-policy outcome; oracle asks whether other legal actions from the same proposal family yield better signed effect. No logical implication connects absence of gain in these random training forwards to absence of better actions. Even a negative best-of-subset oracle would not prove all-N absence or the impossibility of a better-trained U.

## 12. UpdateNet capability assessment

**FACT — gradient/parameter evidence from the respective R4 `metrics.json`:**

| Quantity | R4-U | R4-US |
|---|---:|---:|
| U gradient L2, step 1 | 3.465330564722535e-6 | 3.465330564722535e-6 |
| U gradient L2, step 2 | 1.9774556676566135e-6 | 1.9796998458332382e-6 |
| Spectral gradient L2, step 1 | 0 | 2.641909588874114e-7 |
| Spectral gradient L2, step 2 | 0 | 1.680706702700263e-7 |
| Changed spectral tensors accumulated over steps | 0 | 4 |
| Changed trainable tensors accumulated over steps | 8 | 12 |
| Optimizer steps / routes / actual writes | 2 / 2 / 3 | 2 / 2 / 3 |

`changed_parameter_count` is the spectral-projector changed-tensor accumulator in current `_run_s1`, not the total model or scalar-parameter count. Both arms' D/static/B/semantic/refiner/backbone state hashes stay unchanged; U-only additionally retains the spectral hash. Main `StageOptions`/`main.json` specify Adam LR `0.001`, weight decay `0`, batch size `1`, delta regularization weight `0`; **the run LR/optimizer state is not directly exported**, because runtime/resolved config is missing and the top-level `config_hash` hashes PFGR config rather than the complete stage sidecar. Report these as expected current defaults, not independently verified historical hyperparameters.

| Claim | Assessment |
|---|---|
| U trains mechanically | **Supported as reported:** nonzero U gradients and changed tensors, completed updates; exact-source qualification remains |
| U learns useful corrections | **Not demonstrated:** no post-training common-state improvement; paired effects near zero |
| U has converged | **No evidence:** tiny work, no convergence curve, no repeated evaluation |
| Correction family has sufficient capacity | **Unresolved:** no current-U oracle or controlled capacity diagnostic |

The source keeps frozen D's input gradient live in S1; observed U/spectral gradients are consistent with it, but no explicit frozen-D input-gradient norm was exported. Small gradient norm alone is not starvation: masked-global averaging can dilute sparse effects, and Adam rescales gradients. Conversely, nonzero gradients do not prove effective corrections. Need measured correction norms, parameter update magnitudes, loss conditioning and fixed-producer action gains before changing LR or architecture.

The random-route stream covers very few states/actions here. Dense generation of proposals is not supervision of every candidate. There is no evidence that spatial/semantic support coverage is adequate, nor that the frozen ungrounded semantic head's class names imply accurate tissue predictions. K is a sampled training budget, not learned adaptive stopping. A short bootstrap in section 16 retains the existing target-free random selection and normalized intermediate/final loss; it does not introduce a learned-route bootstrap or target-selected inference actions.

### Must R3 precede interpretable headroom?

**INFERENCE:** headroom on a weak current base answers “can this producer currently make a useful action?”, which is a legitimate early diagnostic. It does not answer “does sparse correction add value beyond a fair strong static predictor?”. A stronger Z0 may remove easy residuals, alter decoder sensitivity, or change which U actions are useful. U trained on one base is not automatically a valid frozen producer on another.

Sequence 1, selected strong Z0 → U training → freeze → headroom, is necessary for the final scientific comparison but can waste base-training effort if a cheap capacity diagnostic exposes a catastrophic problem first. Sequence 2, current-producer headroom → bounded U bootstrap if needed → base pilot → retrain U/freeze → headroom again, is the efficient triage sequence here. An early negative on the tiny current U is not permission to declare the family impossible. An early positive is not permission to skip the strong-base comparison. Final paper headroom must be measured on the selected strong Z0; any changed producer invalidates prior bank compatibility.

## 13. Runbook-vs-code audit

Statuses classify current-main callable semantics; “implemented” does not assert that the missing server runs occurred. Sources: runbook sections R0–R10; CLI dispatch/config/input functions; `stages.py`, `oracle.py`, `benchmark.py`, `value_bank.py`; frozen plan sections 5–11.

| Scenario / transition | Classification | Evidence and mismatch |
|---|---|---|
| R0 preflight | **IMPLEMENTED**, exported evidence incomplete | Strict paths/split/checkpoint loader; no full imaging validation. Promised sidecars/tests missing; scoped dirty flag cannot certify whole checkout. |
| R1 bounded S0 | **IMPLEMENTED BUT ENGINEERING-ONLY** | Correct S0 semantic scope; no scientific-quality claim. Current-main mask-device defect blocks the requested real CUDA path until resolved. |
| R2 sparse/reference CPU service | **IMPLEMENTED** numerical diagnostic | Same sampled work; missing rows prevent ranking/noise/resource audit. |
| R2/evaluate/oracle `--device cuda` | **IMPLEMENTED WITH WRONG SEMANTICS** | Service config preserves JSON CPU stage option; context builder does not align samples to GPU model. Corrected exported R2 source is not current main. |
| R3 four tiny training arms | **IMPLEMENTED BUT ENGINEERING-ONLY** | Runbook honestly says one epoch is not convergence; nevertheless no actual static-evaluation commands are supplied in its R3 subsection, despite the frozen plan requiring them. Training metrics cannot satisfy its advertised D(Z0) quality fields. |
| R3 selected-static prerequisite for R4 | **BLOCKED** scientifically | B2 filename wired as producer; no retained selection review or validation ranking. |
| R4 S1 MAIN updater | **IMPLEMENTED BUT ENGINEERING-ONLY** for exported commands | Two-step U+spectral training; paired metrics are before optimizer steps, not frozen headroom evaluation. |
| R4 U-only as meaningful spectral ownership ablation | **IMPLEMENTED WITH WRONG SEMANTICS** | Runbook and actual run branch from S0/cold spectral projector, contrary to plan's trained-spectral warm-start comparison. Valid only as explicitly labeled engineering control. |
| R4 static/no-op/random and privileged Oracle-1/greedy-K services | **IMPLEMENTED**; **not executed in export** | Runbook contains separate commands and source contains actual teacher/winner/application/confirmation path; no basis to say headroom API is not implemented. CUDA operational blockers still apply. |
| R4→R5 scientific acceptance | **IMPLEMENTED WITH WRONG SEMANTICS** if file existence is treated as the gate | R5 shell prerequisite is a trained-producer file; no CLI check enforces accepted oracle−random evidence. Runbook R4 prose gives correct branching logic but its transition needs an explicit reviewed outcome. |
| R5 MAIN bank on supplied source provenance | **BLOCKED** | Current `ValueBankWriter` rejects synthetic/untrained source for non-engineering banks; downstream R2 source has that flag. No R5 execution exists here. |
| R6 cached V; R7 calibration; R8 controls; R9 final evaluation; R10 resume/package | **IMPLEMENTED**, with dependency/review gates; no R0–R4 export evidence of execution | Prior implementation review describes CPU fixtures; do not count those as new server evidence. No independent claim of completed scientific phases. |

Additional plan/config drift: frozen plan section 5 locks default refiner hidden width `12` and the small predictor, while current `configs/pfgr_lite/main.json.frontend_sidecar.config` explicitly sets `offset_hidden_channels=128` and `point_candidate_multiplier=4`. The example production command consumes that file. This is a concrete current-main configuration mismatch, even though the report-only audit does not change it. Missing resolved configs/checkpoints prevent certifying these as actual run settings. Resolve whether there is an accepted amendment; otherwise restore the locked choice in a separately authorized change and record which producers it affects. Do not silently “fix” a historical config and call it the same experiment.

Other semantic cautions: R2's `value_input_variant=366` inside a random effective policy does not mean V366 was used; V-fit identity is null. S1's route-only operation counter zeros for decoder/MedicalNet/target validations do not mean those operations were absent; their metric namespace does not instrument the whole pipeline. This is inadequate end-to-end resource evidence. No target leakage is demonstrated by these counter inconsistencies; full source and proper counters are needed to verify that stronger claim.

The oracle implementation measures a fresh winner confirmation but **applies the screening-selected winner regardless of the confirmation's sign** (`oracle.py` around 919–993). Confirmation is measurement, not a guaranteed conservative accept/reject rule. Assess final dense/confirmed gains, not optimistic screening gains; do not describe the sampled oracle as an exact upper bound. Later oracle states depend on previous privileged winners, so the trajectory is privileged even though proposal generation receives no target tensor. It must remain outside deployment and MAIN target-free behavior banks.

The root research attachment calls for an independently tuned strong static comparator; B-light is not evidence that such a comparator was optimized. Wider D, PFGR-Full, direct-vector optimization and new write families are proposals outside the frozen Lite MAIN; this audit authorizes none of them. Optional capacity expansion requires a separate scoped proposal after current-U measurement.

## 14. Source reproducibility assessment

**Current verdict:** artifacts and metric copies **PARTIALLY VERIFIED**; clean-attempt PFGR scoped source **PARTIALLY VERIFIED**; exact source for dirty successful R1–R4 **UNVERIFIED**. No recorded run qualifies for fully VERIFIED executable/input/checkpoint reproduction from this export alone.

The run SHA may be a server-only commit outside this clone. The scoped match shows why absence on origin cannot itself invalidate the early attempts. But it cannot recover imported source outside that scope, nor the later dirty patches. There is no proof that `49e0ece` contains only harmless documentation changes relative to main. Current-main histories do not include the server fixes; three distinct later dirty hashes must not be collapsed into one presumed patch.

**RECOMMENDATION — recovery order, not actions performed here:**

1. On the original execution checkout, preserve the commit object/parents (bundle or reachable ref), full binary tracked diff, relevant untracked source bytes, complete source tree/file manifest, exact resolved config, package/module import paths and environment. Recover historical snapshots for each dirty hash from retained server/W&B/code archives if they exist; a newly captured current diff is not automatically the historical diff.
2. Recover split/roles and their hashes, source acquisition/official MedicalNet evidence, actual MN digest, original failure tracebacks, stage receipts/runtime/provenance, input/output PFGR checkpoint SHA256 and benchmark rows. Checkpoint files can stay in controlled artifact storage; report/public Git needs their identities and access instructions, not patient volumes.
3. Recompute scoped hashes using the recorded algorithm, then compare recovered files with current main including imported legacy modules. Match each run separately. If bytes match, no rerun is needed merely because the original SHA was local-only.
4. Commit/review/push the **recovered source or an explicit successor with documented fixes** in a separately authorized implementation task. Record exact source and dirty state on future runs. This audit commits only its report; it does not push unreviewed reconstructed production changes.
5. If a historical patch is irretrievable, retain the old execution as reported engineering evidence, mark exact source unreconstructable, and rerun only the minimum CUDA preflight/S0 or benchmark checks affected by the successor source. Rebuild scientifically used producer checkpoints when a changed computation affects them. Do not rerun all R0–R4 by default.

A supplied digest matching local MN bytes would establish integrity; it still needs an authoritative source chain for official-pretrained registration. `synthetic_untrained=true` cannot be bypassed or manually relabeled to get R5 running. If verified source bytes differ from those used before, prior base/U results belong to a different producer.

## 15. Go/no-go for R5

**DO NOT PROCEED.** This is a scientific funding/transition decision, independent of script exits.

| Required prerequisite | Current status | Minimum resolution |
|---|---|---|
| Strong enough fair static base | Missing | Small matched convergence/validation pilot; select and hash Z0/D |
| Usable U action capacity | Unresolved | Current-producer action diagnostic, then short U bootstrap and fixed-producer reassessment |
| Oracle selection headroom over random | Not tested | Paired no-op/random/confirmed sampled Oracle-1, then conditional greedy Oracle-2 |
| Frozen reproducible source/producer | Missing for successful dirty runs | Recover exact source, configs, checkpoint identities; publish a reviewed reproducible producer |
| MAIN MedicalNet capability | Unverified official provenance; bank guard would reject reported flag | Verify provenance through the legitimate source review/registration path |

Even an available R4 checkpoint is only a dependency artifact. If confirmed oracle and random both improve equally, pursue the simpler correction method instead of generating a large ValueBank for a selector with no established benefit. If Oracle exceeds Random materially but no learned model exists yet, that establishes the prerequisite for R5; it does not prematurely claim case C's learned failure.

## 16. Minimum next experiment

**RECOMMENDATION.** Use a staged, conditional sequence. The exact next action is **NEXT-0 evidence recovery**. The next new experimental execution after recovery is **NEXT-1 current-producer action-capacity diagnostic**, before spending on full static convergence. None of these runs was executed in this audit. Counts below are deliberately bounded starting designs; freeze subject identities/roles, seeds, masks, metric range, candidate draws and decision margins before target measurement. Development evidence remains separate from final untouched test and calibration roles.

### NEXT-0 — recover and freeze

No subjects, actions, K, teacher or optimizer work. Recover the S3-dirty source and R3-B2/R4-US checkpoints first, then the earlier repair snapshots/rows needed for audit. Produce source bundle/patch/file hashes, full dependency map, resolved config, roles/split, MN integrity/official-source evidence and PFGR checkpoint byte hashes. Reconcile current-main device and refiner-config discrepancies. If exact history is lost, declare a successor experiment identity. This unlocks attributable diagnostics, not R5.

### NEXT-1 — early action-capacity diagnostic on current R4-US

**Evaluation only:** 4 fixed development subjects, all producer modules frozen, using the recovered R4-US producer and its own Z0. Run independent no-op K0 and random K1 with 3 predeclared seeds, then sampled Oracle-1 over 32 candidates/subject, fixed-Q1024 screening, and **exact-footprint winner confirmation**. Retain dense Z0/final paired metrics. Record the actual screened candidate IDs, scope relative to all eligible points, signed labels, Q/variance/SE, confirmation discrepancy, masks/denominators, selected correction and write norms/saturation, and resource costs. Audit an exact 8-candidate subset on the same producer if near-zero/noisy screening would otherwise drive a negative decision; do not relabel it all-N.

No-op and Random are target-free inference with a post-route target join. Oracle proposals must precede target measurement; its winner choice/state history remain privileged diagnostic-only. A confirmed positive result demonstrates existence of useful actions in this producer/scope. A negative result only motivates the bounded bootstrap below; it cannot stop the entire action family because current U has barely trained. This small diagnostic does **not** satisfy the final R5 statistical gate.

**Conditional bootstrap if current actions remain negligible:** 8 producer-fit subjects, at most 200 new S1 optimizer updates, batch 1, existing uniform-random K∈{1,2,4} with frozen B2/D/backbone/semantic/points/B, train U+shared spectral projector only. Start with the recovered LR if validated; otherwise explicitly use current default Adam LR0.001 in the successor manifest. Evaluate a frozen snapshot on the same 4 development subjects at predeclared checkpoints; never change routes using their targets during training. Log before/after parameter norms and dense corrections, K/location coverage and train/development paired gains. Stop this bootstrap if gradients/nonfinite states fail or corrections remain negligible at the bounded endpoint; then review optimization/input sensitivity before funding larger runs. A direct-vector/write-family capacity study is a separate future authorization, not silently added here.

### NEXT-2 — fair small R3 convergence pilot

**Training plus development evaluation:** 32 producer-fit subjects and 8 distinct development subjects, fixed common data order and matched optimizer-step cap, four B0/B1/B2/B-light arms. Use one paired initialization seed initially; match B1/B2 tensors and explicitly record common D initialization where possible. Train each for up to 20 passes (640 batch-one updates) with the same objective and optimizer settings; evaluate each saved checkpoint on all 8 development subjects with `static`, K0, no teacher/action/V. These bounds are a pilot, not a declaration of convergence. Record train curves, independent validation MAE/PSNR/SSIM/Charbonnier, parameter counts including dormant source weights, wall time/memory and checkpoint/source hashes.

If learning is still substantial at the cap, mark ranking unresolved and extend **only the competitive arms** after reviewing the curve. If the leading B1/B2 or simpler competitor is separated, verify the leading pair with another paired seed before choosing. Do not declare modality benefit from one seed or demand full-scale training before seeing pilot stability. A fair strong base is selected by held-out quality/convergence and practical cost, not by the earlier two-step loss. If all arms are poor, inspect data/prior/objective before routing.

### NEXT-3 — bootstrap U on selected Z0, freeze, then true headroom

**Training:** selected Z0/D checkpoint; freeze all except U+shared spectral projector; start with the same 32 producer-fit subjects, up to 320 S1 updates, existing random K∈{1,2,4}. Log actual K counts and coverage rather than assuming uniform finite-sample realization. Reassess on the 8 development subjects at fixed checkpoints. If there is no useful correction at this bound, stop and review U/decoder sensitivity; do not label convergence automatically. Any later U-only spectral-ownership comparison must branch from this same trained U/spectral snapshot; it is optional and not needed before the first headroom answer.

**Frozen evaluation:** on the same 8 development subjects initially, no-op K0; random K1 with 3 fixed seeds; sampled Oracle-1 with 32 candidates, fixed-Q1024 screening and exact-footprint winner confirmation; dense final metrics throughout. Add matched Random-2 and greedy Oracle-2 only if useful one-step correction is established or a specific interaction hypothesis is documented. Add Oracle-4 only if Oracle-2 continues to improve meaningfully. All controls share the same selected-base/U/D/spectral/point snapshot. Report both oracle−Z0 and oracle−random with paired subject effects, label/confirmation uncertainty, candidate coverage and exact telescoping checks where available. Greedy-K is not a global K-step optimum.

If the pilot suggests selection benefit, freeze the design and extend to **32 independent development subjects** before an R5 acceptance decision, consistent with current `ComparisonOptions`/runbook's default independent-subject floor. Repeated seeds/states are clustered within subjects. That floor is not power assurance: use paired intervals against a predeclared practical margin; wide intervals remain INCONCLUSIVE. The older synthesis proposes a 1% relative MAE contribution floor with positive signed Charbonnier gain and no material PSNR/SSIM degradation; adopt or revise that research margin **before** this expanded evaluation, never from the resulting effect size. Report outcome-based stopping, not a required positive route length. If Oracle≈Random and both improve, retain simple sparse correction; if neither improves with adequate precision and trained U, review or stop routing.

### NEXT-4 — only after accepted selection headroom

Then and only then run a bounded R5 bank pilot: 2 producer-fit subjects, at most 3 distinct states/subject and 32 measured candidates/state, fixed-Q chosen from the prior fidelity audit (Q1024 is a provisional starting value, not automatically reliable). All producers frozen; no optimizer; target-free behavior completed before labels; no oracle-derived states in MAIN. Required outputs are complete row/shard/index/scale identities, positive/negative/neutral gains, Q/SE, selection strata, replay artifacts and successful replay checks. This establishes bank mechanics on the accepted producer before scaling V work. Current evidence does not authorize this step now.

### Table F — uncertainty reduction and stopping

| Next experiment | Cost driver | Decision unlocked | Stop condition |
|---|---|---|---|
| NEXT-0 recovery/freeze | Server artifact retrieval and source comparison; no GPU training | Can experiments be attributed/reproduced? | Missing historical bytes must remain unreconstructed; successor requires explicit identity |
| NEXT-1 current-producer diagnostic; conditional small U bootstrap | Candidate footprint measurement, dense decode, then bounded S1 backward if needed | Is there any early action capacity, or an optimization/sensitivity blocker? | Invalid provenance/device/target boundary; noisy negative requires exact subset check; immature-U negative does not falsify family |
| NEXT-2 matched static pilot | Shared frozen prior traversal, static-head/D optimization and validation | Which Z0 is fair and strong enough? Does ordered conditioning help? | Nonfinite/leak; cap with unstable ranking → unresolved, extend only competitors |
| NEXT-3 selected-base U and headroom | S1 unroll/backward, frozen random/oracle evaluation, independent confirmation | Correction versus selection headroom; scientific go/no-go R5 | Precise negligible oracle gain → review/stop router; oracle≈random positive → simpler correction; wide intervals → INCONCLUSIVE |
| NEXT-4 conditional small bank | Frozen traces × measured candidates × Q; replay/storage | Is accepted-producer ValueBank ready for cached V fitting? | Missing accepted headroom/source, stale producer, noisy labels or replay mismatch |

The representative R2 timing check in section 6 can reuse saved NEXT-1/NEXT-3 diagnostic states; it should not delay a cheap existence-of-headroom test merely to chase the proposed speed target. It must precede claiming scalable teacher efficiency or committing large bank-generation resources. No wall-clock completion guarantee is inferred from these pilots.

## 17. Paper-level evidence status

### Table E — claims and required evidence

| Scientific claim | Current evidence | Status | Required next evidence |
|---|---|---|---|
| Strong modality-aware base | B2/B-light source paths preserve order; all tiny R3 optimizers execute; no held-out ranking | **PARTIALLY SUPPORTED** for architecture, unsupported for strength | Matched convergent-enough R3 validation pilot and stable selection |
| Useful sparse corrections | R4 has nonzero gradients/parameter movement; realized training effects near zero | **UNSUPPORTED** as a benefit claim | Frozen trained U improves common Z0 on independent development subjects |
| Selection headroom | No oracle execution exported | **NOT YET TESTED** | Independently confirmed Oracle−Random gap on common strong Z0/U |
| Footprint-teacher necessity in this MRI workload | Source cancellation theorem and prior synthetic local-positive/global-negative counterexample; no R0–R4 same-action local/full labels | **PARTIALLY SUPPORTED** mathematically, untested empirically | Same-action local/footprint/dense gains and decision regret/harm comparison on real development states |
| Efficient exact teacher | FP32 sampled sparse/full parity reported on CPU/CUDA; method-specific mean savings, no rows/stability/scaling | **PARTIALLY SUPPORTED** | Reconstruct source, exact-subset/dense and ranking checks, representative stable same-work throughput/resource evidence |
| Learned routing value | No V training/use or learned/random evaluation here | **NOT YET TESTED** | After accepted selection headroom, same-bank V evaluation then matched learned/random inference |

**Most important missing scientific result:** a reproducible, independently confirmed **Oracle−Random advantage that also improves Z0**, using a frozen useful U on the selected strong modality-aware base. It determines whether selecting point corrections offers a plausible contribution beyond a static model or simple sparse correction. A fast exact teacher is valuable engineering but cannot substitute for that result. No CVPR, SOTA, novelty, clinical or diagnostic-equivalence claim follows from this export.

## 18. Blocking issues

| Severity / issue | Expected versus actual | Consequence / bounded remedy |
|---|---|---|
| Critical for R5: no selection-headroom execution | Expected paired frozen no-op/random/oracle; only updater training exported | Do not start scientific ValueBank; execute section 16 after recovery |
| Critical provenance: dirty source unreconstructed | Expected source bytes and source/checkpoint hashes; only inaccessible SHA and dirty hashes | Recover historical snapshots; no invented patch or blanket rerun |
| Critical MAIN capability: official source unverified | Expected legitimate pretrained source chain; R2 flags official=false/synthetic_untrained=true | Review/verify actual file source before MAIN bank; do not bypass capability guard |
| High: base/U insufficiently trained | Expected stable strong Z0 and useful U; only tiny training histories | Small convergence/capacity sequence, not full automatic training |
| High: current-main CUDA correctness/flag handling | Expected model/sample/mask on effective requested device; mask normalization and service propagation defects remain | Recover/review fixes in separate code task; minimal affected verification afterward |
| High: incomplete export | Expected weights/config/roles/runtime/checkpoint hashes/benchmark rows/tracebacks; absent | Retrieve existing metadata; preserve old executions and failures |
| High for causal ablations: U-only cold projector | Expected shared trained spectral producer; both arms branch S0 | Reclassify current control; future warm-start matched branch only if needed |
| High plan drift: refiner config | Frozen plan small default versus current main sidecar width128 | Locate amendment or reconcile before new producer; actual-run setting currently unknown |
| Medium: misleading counters/role/W&B metadata | Outer validation role and route-only zeros can be misread as cohort/whole-work facts | Keep actual scope explicit; obtain resolved role and full resource records |
| Medium: benchmark acceptance incompletely measured | Missing rankings, raw timing dispersion/cold-warm/resource rows | Recover rows then one representative matched benchmark; below-target tiny ratio is not falsification |

These severities concern interpretation and the next decision. They do not claim that the recorded metrics are fabricated or that the entire implementation is scientifically invalid. No missing evidence has been synthesized.

## 19. Final recommendation

Keep the successful executions as **engineering progress with partial provenance**, retain both failures, and label the exported R4 as **S1 updater smoke / headroom INCONCLUSIVE**. Credit R2's reported sampled-query numerical agreement separately from label fidelity and runtime scalability. Credit R3/S1 gradient ownership separately from useful reconstruction.

Recover the exact producer/source first, make the cheap current-producer action-capacity diagnostic next, then run a modest fair static pilot and selected-base U/headroom sequence. The scientific R5 gate remains closed until a reproducible confirmed oracle advantage over random and Z0 exists on that selected producer, with adequate precision and legitimate MedicalNet provenance. If the advantage is absent, simplify or stop the routing claim; do not tune K for appearance.

### Audit verification and delivery boundary

Read-only checks in this audit: Git object/ref/history/source-scope inspection; parsing every exported JSON and inspecting YAML metadata; equality checks across duplicated metrics/service/parity/W&B records; recomputation of paired improvements, aggregate means and speed ratios; current-source parameter counting without forward/training; static inspection of relevant invariant tests. No test that trains a fixture, creates a run, reads MRI volumes or executes an oracle was run. Prior tests cited by the implementation review remain historical software evidence. Report formatting, local references, artifact immutability and report-only Git staging are checked before delivery.

For export identity, recursively sort regular files under `experiments_R0_R4`; for each take `(relative POSIX path, byte size, SHA256(file bytes))`; serialize the list using Python `json.dumps(items, separators=(",", ":"))` and SHA256 its UTF-8 bytes. The audited export identity is `d859e796474d8510a36f7ebc0867b2302e4e0c08a87fec0900d79db06469fd2c`. This is a reproducible integrity fingerprint of the supplied **incomplete export**, not a replacement for its absent source/checkpoints. Its per-run file ledger follows.

| Run directory | All retained files (relative to run) |
|---|---|
| `R0-20260908T115116Z-473456` | `receipt.json`; `wandb/run-20260908_185121-ujvokv7i/files/config.yaml`; `wandb/run-20260908_185121-ujvokv7i/files/wandb-metadata.json`; `wandb/run-20260908_185121-ujvokv7i/files/wandb-summary.json` |
| `R0-20260908T121953Z-473456` | `receipt.json`; `wandb/run-20260908_191957-cu68ijxm/files/config.yaml`; `wandb/run-20260908_191957-cu68ijxm/files/wandb-metadata.json`; `wandb/run-20260908_191957-cu68ijxm/files/wandb-summary.json` |
| `R1-real-20260908T115116Z-473456` | `receipt.json` |
| `R1-real-20260908T121953Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_192201-5crfliaj/files/config.yaml`; `wandb/run-20260908_192201-5crfliaj/files/wandb-metadata.json`; `wandb/run-20260908_192201-5crfliaj/files/wandb-summary.json` |
| `R2-20260908T121953Z-473456` | `benchmark.json`; `parity.json`; `receipt.json`; `service_receipt.json`; `wandb/run-20260908_195334-lnfjnkrc/files/config.yaml`; `wandb/run-20260908_195334-lnfjnkrc/files/wandb-metadata.json`; `wandb/run-20260908_195334-lnfjnkrc/files/wandb-summary.json` |
| `R2-gpu-20260908T133702Z-473456` | `receipt.json` |
| `R2-gpu-fixed-20260908T134203Z-473456` | `benchmark.json`; `parity.json`; `receipt.json`; `service_receipt.json`; `wandb/run-20260908_210149-w8h0fnpc/files/config.yaml`; `wandb/run-20260908_210149-w8h0fnpc/files/wandb-metadata.json`; `wandb/run-20260908_210149-w8h0fnpc/files/wandb-summary.json` |
| `R3-b0-20260908T140929Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_211431-khdjdvva/files/config.yaml`; `wandb/run-20260908_211431-khdjdvva/files/wandb-metadata.json`; `wandb/run-20260908_211431-khdjdvva/files/wandb-summary.json` |
| `R3-b1-20260908T140929Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_211249-2j9zk9hs/files/config.yaml`; `wandb/run-20260908_211249-2j9zk9hs/files/wandb-metadata.json`; `wandb/run-20260908_211249-2j9zk9hs/files/wandb-summary.json` |
| `R3-b2-20260908T140929Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_211108-cofa61n2/files/config.yaml`; `wandb/run-20260908_211108-cofa61n2/files/wandb-metadata.json`; `wandb/run-20260908_211108-cofa61n2/files/wandb-summary.json` |
| `R3-blight-20260908T140929Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_211612-xvev57r4/files/config.yaml`; `wandb/run-20260908_211612-xvev57r4/files/wandb-metadata.json`; `wandb/run-20260908_211612-xvev57r4/files/wandb-summary.json` |
| `R4-u-only-20260908T140929Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_213712-jynsup8x/files/config.yaml`; `wandb/run-20260908_213712-jynsup8x/files/wandb-metadata.json`; `wandb/run-20260908_213712-jynsup8x/files/wandb-summary.json` |
| `R4-updater-20260908T140929Z-473456` | `metrics.json`; `receipt.json`; `wandb/run-20260908_213340-zwbcsaly/files/config.yaml`; `wandb/run-20260908_213340-zwbcsaly/files/wandb-metadata.json`; `wandb/run-20260908_213340-zwbcsaly/files/wandb-summary.json` |
