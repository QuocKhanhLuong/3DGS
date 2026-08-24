# Ox Alpha Performance Forensics

Task: `task_b0b8cd8c7754` · Run: `run_7076b11588f3` · Dispatch: `ctx_f1459572f316` (continuation after a stalled prior Ox Alpha session; no benchmark was re-run).
Repo: `/Users/alvinluong/3DGS`, frozen main HEAD `0efeb94af72ffa067769e19afcd19ad358feefd2` (verified via `git rev-parse HEAD` at session start and re-verified before completion).
Status: **performance findings are recommendations only, not accepted fixes.** This report changes nothing in production code, tests, configs, plans, or existing docs.
Sol technical review: after reading the independent draft, Sol narrowed unsupported claims about physical filesystem opens, real-I/O verdicts, copy counts, transpose savings, and decoder reuse. Those editorial corrections changed only this report and added no new measured result.

Evidence provenance legend used below:

- **[P]** preserved numeric evidence measured by the prior stalled Ox Alpha session on this machine (synthetic gzip/raw arrays, CPU-only, under heavy host load); source scripts preserved at `/var/folders/.../T/opencode/bench_{data_path,dataloader,model,profile_load}.py`.
- **[S]** static source evidence recovered fresh in this continuation session from the frozen HEAD.
- **[D]** official primary documentation (links in §9).

---

## 1. Environment

| Item | Value |
|---|---|
| Host | Apple M1 (arm64, macOS/darwin), 8 logical CPUs, 16 GB RAM |
| Storage | APFS SSD; repo filesystem 96% full, ~9.3 GiB free — disk-pressure conditions during all measurements |
| GPU | No CUDA device (`torch.cuda.is_available() == False`). MPS is available but the training stack targets CUDA semantics; no MPS benchmark was run. |
| Python / PyTorch | Repo venv: Python 3.10.0, Torch 2.13.0, CUDA build false, MPS true, `torch.get_num_threads()=4`, interop 8; NumPy 2.2.6. Global Python 3.12 Torch install is broken and was not used. |
| Scientific deps absent locally | No nibabel, no scipy → real `.nii.gz` decode could not be exercised locally. No local BraTS21 dataset exists on this machine. |
| MCP tools actually available (OpenCode) | Only the local **Semble** MCP server is configured (`semble_search`, `semble_find_related`) plus generic MCP resource listing (`list_mcp_resources` returned empty; `list_mcp_resource_templates`). **No Serena, no docs MCP, no Morph, no external-search MCP was available.** The prior Ox Alpha session used `list_mcp_resources`, `list_mcp_resource_templates`, and `semble_search`; this continuation recovered all file-line evidence with the repo CodeGraph task entrypoint (`python scripts/codegraph.py --task server_pipeline`), Grep/Read, because the symbols were already precisely located. External research used direct fetches of official doc pages through the harness's normal network tooling (§9). Playwright/browser automation was not relevant and not used. |

Consequences for claim scope: **real NIfTI decode, real storage behavior, CUDA timing, H2D transfer, profiler overlap analysis, and GPU starvation are NOT YET PROVEN.** Everything numeric below is either synthetic-CPU [P] or static-source [S].

## 2. Actual data path

Full chain, with functions/files at frozen HEAD:

1. **Disk open + NIfTI proxy materialization** — `_read_nifti` (`src/smagm/data/brats21_point_guided.py:912-940`): `nib.load(str(path), mmap=True)` then immediately `data = np.asanyarray(image.dataobj)` (:920), shape/dtype/finiteness checks (:921-928), metadata extraction, final `np.asarray(data)` (:936). Per NiBabel docs [D], `mmap=True` only helps *uncompressed* files: "If file_like cannot be memory-mapped, ignore mmap value and read array from file" — `.nii.gz` cannot be memory-mapped, so each read fully decompresses and materializes the volume. The proxy is therefore never actually lazy here; there is no `get_fdata` caching benefit either since the array is pulled out of the proxy immediately.
2. **Per-sample orchestration** — `load_point_guided_subject` (:1278-1393): loops T1/T2/FLAIR calling `_read_nifti` once per modality (:1319-1325) plus geometry equality checks `_same_geometry` (:943-951, two `np.allclose` per extra volume), `np.stack` of the three XYZ volumes (:1327), union brain mask `derive_input_brain_mask` (:954-982, `np.abs(...)>thr` over `[3,X,Y,Z]` :979), transpose to DHW with an explicit contiguous copy `nifti_xyz_to_dhw` (:1335, defined at :221-235 as `np.ascontiguousarray(np.transpose(a,(2,1,0)))`), then `_normalize_masked` per modality (:1338-1349), then **T1ce target read+normalize** (:1351-1364) and **segmentation read + integral check + label validation** (:1366-1375), finally `torch.stack` of normalized observations into `[3,D,H,W]` float32 (:1386).
3. **Normalization hot spot** — `_normalize_masked` (:985-1045): casts the full volume to **float64** (:995), boolean-selects masked voxels (:999), float64 mean/std (:1007-1008), allocates a float64 output volume (:1009), masked z-score or percentile clip (`np.percentile` :1017-1018, `np.clip` :1024, scatter assignment :1025), cast to float32 (:1027), finiteness check (:1028), `torch.from_numpy(np.ascontiguousarray(...))` (:1045).
4. **Sample contract validation** — `BraTS21PointGuidedSample.__post_init__` (:1061-1092): full-volume `torch.isfinite(...).all()` on observations/target (:1066, :1073), `torch.unique` label validation on segmentation (:1078-1080) — duplicating the NumPy-side label check already done at :1372-1374.
5. **Collation** — `collate_point_guided_samples` (:1399-1447): pairwise affine `np.allclose` between samples (:1418), four `torch.stack`s over ~223 MB/sample tensors (:1422-1437), producing the frozen dataclass `PointGuidedBatch` (:1188) whose own `__post_init__` repeats full-batch `isfinite`/`unique`/`allclose` validations (:1200-1229).
6. **DataLoader** — `_make_loader` (`src/smagm/training/point_guided.py:1004-1037`): `num_workers=settings.num_workers` (**default 0**, :161), `pin_memory=context.device.type=="cuda"` (:1033), `persistent_workers=num_workers>0` (:1035), no explicit `prefetch_factor` (so None at workers=0, default 2 otherwise [D]).
7. **H2D + step** — `_prepare_batch` (:383-396) moves observations/mask with `non_blocking=True` (:387-388); the target-free frontend context is built first, and only then is the T1ce target (and segmentation) moved to device and passed to `compute_training_objective` (:696-709, timer `gate_e_loss` at :702). AMP autocast + CUDA `GradScaler` paths exist (:323-333).

Byte accounting per subject (240×240×155 = 8.926M voxels): int16 on-disk MRI ≈ 17.86 MB/volume; the five synthetic compressed payloads together were **50.8 MB** [P]. Resident torch tensors per sample ≈ **223.2 MB**: observations `[3,D,H,W]` fp32 107.1 MB, target fp32 35.7 MB, mask bool 8.9 MB, segmentation int64 71.4 MB [P]. Exact copy counts were not measured with a memory profiler. The source nevertheless establishes a lower bound of roughly **eight volume-equivalent writes per observation by the end of collation**: proxy materialization, contribution to the XYZ stack, contiguous DHW transpose, float64 cast, float64 normalized output, float32 cast, observation stack, and batch stack. At full shape this is about **305 MB written per observation** before counting union-mask temporaries, validation scans, allocator effects, or decompression buffers. These are sequential materializations/writes, not eight simultaneously resident copies.

## 3. Three-volume I/O findings

All numbers synthetic-gzip [P]; real NiBabel NIfTI NOT YET PROVEN.

- The source performs **5 logical volume-load calls** (T1, T2, FLAIR, T1ce, seg) every `__getitem__`, with **no application cache**; every epoch repeats those calls and their preprocessing [S] (`brats21_point_guided.py:1319-1375`, dataset `__getitem__` :1491 delegating to `load_point_guided_subject`). Exact OS-level `open`/`read` syscall counts were not measured because NiBabel and a real dataset were unavailable; NiBabel may perform more than one filesystem operation per logical load.
- One synthetic MRI gzip read+decompress (int16 240×240×155): median **71 ms**, p90 174.8 ms; synthetic segmentation 26.56 ms (p90 49.1). Implied decompressed-byte throughput was ~250 MB/s including zlib. This suggests raw reads were not dominant in this synthetic cached-host experiment, but it does **not** establish real BraTS storage or NiBabel decode performance.
- Union mask derivation: median **106.46 ms**, p90 324.4 ms.
- Transpose+contiguous copy (int16, full volume): median **32.50 ms**, p90 44.7 ms.
- `masked_robust_01` normalization starting from float64: median **954.63 ms**, p90 2453.95 ms — the single largest per-stage cost, ~13× the gzip decode.
- Full synthetic five-volume load: median **33.02 s**, p90 53.41 s under severe host saturation (load average >70 during parts of the run); bounded loader run at num_workers=0 gave **36.21 s for 2 subjects (18.11 s/sample)**. Worker 1-vs-2 comparison was terminated by saturation and is **inconclusive**.
- cProfile over three loads (52.63 s total): `_normalize_masked` **29.32 s** (12 calls); percentile/quantile/partition ≈ 13.7 s / 11.7 s combined attribution; dataclass `__post_init__` ≈ 11.0 s; `torch.unique` label validation 7.71 s; `np.allclose/isclose` 3.53 s; fake read/decompress ≈ 3.1 s; `ascontiguousarray` 2.82 s; `astype` 2.48 s; `torch.isfinite` 2.43 s; zlib decompress 2.21 s; flatten 2.02 s.
- Synthetic collate: B=1 median **3454.5 ms**, B=2 median **13684.6 ms**. The noisy two-point experiment grew by more than batch size alone; source and profile evidence identify `PointGuidedBatch.__post_init__` full-batch validation and stacking of ~223 MB/sample tensors as contributors, but the scaling curve needs a controlled rerun.

Interpretation: the "three-volume I/O" question splits into (a) real storage+NiBabel+gzip, which was not measurable here, and (b) the implemented CPU load pipeline around it, which was seconds-to-tens-of-seconds per synthetic sample due to float64 normalization, repeated materialization, duplicated validation, and zero caching. Within the synthetic experiment, (b) dominated the gzip-read proxy; whether that ordering holds on the server still requires measurement.

## 4. CPU bottleneck findings

Ranked by measured contribution (all [P]/[S]):

1. **Float64 masked normalization** — `_normalize_masked`: 29.32 s of a 52.63 s profile (≈56% of three-load wall time); stage median 954.63 ms/modality. Percentile mode adds `np.percentile` partition work (~13.7/11.7 s attributed). The float64 intermediate doubles memory traffic vs float32 and forces two more full-volume passes (zeros alloc + scatter + astype).
2. **Repeated full-volume materialization** — per-modality contiguous DHW transposes after the XYZ stack, a provably redundant `.copy()` on an already-contiguous mask (`:1335`), `np.stack` in XYZ, and `torch.stack` at sample and batch level; cumulative `ascontiguousarray` 2.82 s + `astype` 2.48 s in profile. Only the extra mask copy is proven removable by this audit; broader layout changes require equivalence benchmarking.
3. **Duplicated contract validation** — labels validated twice (NumPy `np.unique` at loader :1372 AND `torch.unique` in `__post_init__` :1078 AND again per batch :1214); affine `np.allclose` validated per modality pair, per sample pair in collate, and spacing-vs-affine again in `PointGuidedBatch.__post_init__` (:1223-1225); `torch.isfinite` over every tensor at sample and batch level (2.43 s profile). `__post_init__` alone ≈ 11.0 s of the profile.
4. **num_workers=0 serialization** — DataLoader defaults leave decompression+normalization inside the training process, blocking compute between steps [S config; D semantics]. The macOS spawn start method additionally makes worker startup expensive if raised naively [D].
5. **Union-mask pass over stacked observations** — 106.46 ms median, p90 324 ms; computed on int16 stack before any downcast.
6. **Collate/validation scaling** — 3.45 s @B=1 → 13.68 s @B=2 synthetic: per-batch validation is O(B × volume) with several full-tensor reductions per field.
7. **Host synchronization inside Gate E** — see §5.

Not isolatable locally: thread oversubscription effects beyond torch's 4 threads (no MKL/OMP control experiment survived host saturation), OS page-cache hit rates for repeat epochs, and any GPU-side queueing.

## 5. Gate E findings

Static trace [S]:

- Entry: `PointGuidedMRIModel.compute_training_objective` (`src/smagm/features/point_guided/model.py:419-438`) → `_compute_training_objective` (`src/smagm/features/point_guided/training_objective.py`) iterates **per trajectory step** (`for step_index, step in enumerate(trace.result.steps)` :342) and calls `counterfactual_reward_supervision` per active step (:358-374).
- Inside each call (`src/smagm/features/point_guided/reward_supervision.py:603-767`) a `torch.no_grad()` loop runs over **each counterfactual candidate slot** (:678), performing per slot: one `_counterfactual_transition` (dynamic state query + UpdateNet + write-back, :580-600) and **four decoder point-decodes** (local before/after :692-703, spill before/after when `spill_sample_count>0` :707-726). Total decoder/query invocations scale as `steps × candidates × (4 + transitions)`.
- Host syncs: implicit device→host coercions at `training_objective.py:344` (`bool(active.any())`), `:376` (`if reward_result.valid_count:`), `:392`, `:414`, `:429` (`bool(x.any())`), and explicit `.detach().cpu()` conversions at `:394`, `:421`, `:442`. Each forces a pipeline drain per step; on CUDA these become serialization points. Target-free boundary intact: target enters only as supervision values inside the objective; route decisions remain detached [S].
- Measured Gate-E share of the full step (CPU synthetic, noisy, small batch [P]):
  - k_max sweep (candidates=8 unless noted): kmax1 459.0 ms of 2309.9 ms (**19.9%**); kmax2 328.6/798.5 (**41.1%**); kmax4 819.2/1577.7 (**51.9%**); kmax8 1257.4/1807.2 (**69.6%**); kmax16 3951.8/5657.1 (**69.9%**).
  - Candidate-count sweep at fixed schedule: 8 candidates 3469.3/6812.9 ms (**50.9%**); 16 candidates 1738.9/2733.3 (**63.6%**); 32 candidates 2118.7/2673.9 (**79.2%**) — noisy absolute values, monotone share trend.
  - Economic stop at kmax16: Gate E collapses to 5.0 ms of a 305.2 ms step (**1.6%**) — stopping early, not cheaper kernels, removes almost all Gate-E cost.
  - Larger volume (48³, 256 points, kmax4, candidates=16): 841.8/1777.0 ms (**47.4%**).
- Mathematically necessary vs incidental: the measured counterfactual RewardNet targets, local/spill errors, monotonic hinge, and update-magnitude penalty are contract-required (Gate E E1–E9). The four local/spill, before/after decoder calls per slot are real, but this audit did **not** prove that any two calls have identical tensor inputs and can be cached safely. Proven incidental overhead is limited to per-step host synchronization and Python-loop scheduling; batching or reuse remains a hypothesis requiring input-identity and autograd-equivalence evidence. Any change must keep the locked reuse of one decoder/UpdateNet/write-back and must not alter route decisions.

## 6. Timing breakdown

Stage table. Medians/p90 from [P] synthetic CPU runs; % of step computed against matching-run totals. **CPU/GPU column: everything measured is CPU; no CUDA stage has ever been timed on this machine.**

| Stage | Median | p90/p95 | % of step | CPU/GPU | Status |
|---|---|---|---|---|---|
| T_disk (open+gzip decompress, per MRI volume, synthetic) | 71 ms (seg 26.6 ms) | p90 174.8 ms | n/a (loader side) | CPU/I-O | synthetic proxy only; real NIfTI **not proven** |
| T_nifti_decode (proxy→ndarray incl. checks) | included above; zlib share 2.21 s/3 loads | — | — | CPU | **not isolatable** separately from T_disk in our runs |
| T_numpy_preprocess (union mask) | 106.46 ms | p90 324.4 ms | — | CPU | synthetic |
| T_numpy_preprocess (masked_robust_01 normalize, per modality) | 954.63 ms | p90 2453.95 ms | dominant loader cost | CPU | synthetic |
| T_tensorize (transpose+contiguous int16) | 32.50 ms | p90 44.7 ms | — | CPU | synthetic |
| T_collate (direct collate B1 / B2 proxy) | 3454.5 / 13684.6 ms | — | — | CPU | synthetic direct-collate timing, not DataLoader wait |
| T_dataloader_wait | — | — | — | CPU | **UNAVAILABLE** independently; serialized end-to-end loader was 18.11 s/sample @workers=0 under load>70; worker scaling inconclusive |
| T_h2d | — | — | — | — | **NOT YET PROVEN** (no CUDA) |
| T_frontend (+Gate C+D context, kmax1/2/4/8/16) | 863.1 / 314.7 / 495.2 / 325.0 / 736.4 ms (noisy, non-monotone) | — | 12–40% of matching totals | CPU | includes Gate C+D context; **T_gate_c / T_gate_d individually UNAVAILABLE** (could not be isolated) |
| T_gate_e (kmax1/2/4/8/16) | 459.0 / 328.6 / 819.2 / 1257.4 / 3951.8 ms | — | 19.9 / 41.1 / 51.9 / 69.6 / 69.9% | CPU | synthetic; economic-stop variant 5.0 ms (**1.6%**) |
| T_gate_e (candidates 8/16/32) | 3469.3 / 1738.9 / 2118.7 ms | — | 50.9 / 63.6 / 79.2% | CPU | noisy absolutes |
| T_backward (kmax1/4/16) | 417.6 / 195.5 / 605.7 ms | — | 18.1 / 12.4 / 10.7% | CPU | synthetic |
| T_optimizer | 1.5–8.0 ms | — | <0.5% | CPU | synthetic |
| T_total_step (kmax1 baseline) | 2309.9 ms | — | 100% | CPU | larger 48³ config: 1777.0 ms |

Caveat: candidate-count runs were executed under fluctuating host load; their absolute medians disagree with the kmax-sweep totals at equal configuration. Only shares within a run should be compared, never absolute cross-run values.

## 7. GPU starvation assessment

**NOT YET PROVEN — and unprovable on this machine.** There is no CUDA device; MPS timing would answer nothing about the CUDA server path; NVIDIA Nsight Systems GPU metrics explicitly support only "Linux targets on x86-64 and aarch64, and Windows targets" [D], so the M1/macOS host cannot collect them even with a GPU present.

Structural evidence raises starvation risk for the eventual server run but does not prove it:

- With `num_workers=0`, official PyTorch semantics are that "data loading may block computing" in the training process [D]; an 18 s/sample loader path in front of a sub-second-to-second GPU step would starve any GPU.
- `pin_memory=True` is requested only on CUDA (`training/point_guided.py:1033`), but the custom frozen-dataclass `PointGuidedBatch` defines **no `pin_memory()` method** [S]; official DataLoader behavior is to return such custom batches **without pinning** [D]. The existing `non_blocking=True` calls therefore cannot rely on the documented pinned-memory path for transfer/compute overlap. Exact H2D behavior and impact remain unmeasured.
- Gate E contains ≥5 host-sync sites per trajectory step (§5), each draining the accelerator pipeline on CUDA [D: async execution + sync semantics].
- AMP + GradScaler infrastructure exists (:323-333) but its effect on server throughput is unmeasured.

Required server-side proof (when hardware allows): `torch.profiler` with `ProfilerActivity.CPU+CUDA` and wait/warmup/active schedule [D], CUDA-event bracketing with `torch.cuda.synchronize()` [D], and `nsys --gpu-metrics-devices` timelines reading SM utilization/gaps ("Is my GPU idle?" instrumentation [D]).

## 8. Root-cause ranking (measured above hypotheses)

1. **Float64 masked normalization** (`_normalize_masked`, `brats21_point_guided.py:985-1045`) — measured: 29.32 s/52.63 s profile; 954.63 ms median per modality. Largest single measured sink.
2. **Zero-cache, five-logical-load-per-sample path with repeated full-volume materialization and duplicated validation** (`load_point_guided_subject` :1278-1393; sample/batch `__post_init__`) — measured: 33 s synthetic five-volume load, 18.1 s/sample end-to-end at workers=0; `__post_init__` ≈11 s, unique 7.7 s, allclose 3.5 s, isfinite 2.4 s, ascontiguousarray 2.8 s, astype 2.5 s. OS-level filesystem operation counts were not measured.
3. **Gate E per-step × per-candidate loop with four decoder queries per slot and per-step host syncs** (`training_objective.py:342-442`, `reward_supervision.py:677-726`) — measured: 19.9%→79.2% of step as k_max/candidates grow; 1.6% with economic stop. Grows with schedule depth, unlike fixed-cost stages.
4. **Single-process DataLoader (num_workers=0)** (`training/point_guided.py:161,1032`) — measured indirectly (everything serialized in-process); worker-scaling magnitude inconclusive due to host saturation.
5. **Collate-time full-batch re-validation and stacking** (`collate_point_guided_samples`, `PointGuidedBatch.__post_init__`) — measured: 3.45 s @B=1, 13.68 s @B=2.
6. **Ineffective pinning of the custom batch type** (`PointGuidedBatch` lacks `pin_memory()`; `pin_memory=cuda-only` at :1033) — doc-derived [D]+static [S]; performance impact **NOT YET PROVEN** (needs CUDA).
7. Hypotheses demoted for lack of measurement: OS page-cache thrash from 96%-full disk (plausible, unmeasured), thread oversubscription (torch pinned at 4 threads; untested alternatives), gzip level choice (files are given, not generated by us).

## 9. External research (primary citations/links)

All primary sources, fetched directly from vendor domains during this continuation:

1. **PyTorch 2.13 `torch.utils.data` (DataLoader)** — https://docs.pytorch.org/docs/2.13/data.html — num_workers default 0; single-process loading blocks computing under the GIL; multi-process loading runs dataset access + IO + transforms + `collate_fn` in workers; `prefetch_factor` default None at workers=0 else 2; `persistent_workers` keeps datasets alive across epochs; **memory pinning recognizes only Tensors/maps/iterables — a custom collate return type is returned unpinned unless it defines a `pin_memory()` method**; macOS uses the spawn start method (pickled dataset/collate requirements).
2. **PyTorch 2.13 CUDA semantics note** — https://docs.pytorch.org/docs/2.13/notes/cuda.html — GPU ops asynchronous by default; `to()`/`copy_()` accept `non_blocking` to skip unnecessary sync; accurate timing requires `torch.cuda.synchronize()` or `torch.cuda.Event`; "Use pinned memory buffers": H2D copies are much faster from pinned (page-locked) memory and `non_blocking=True` can overlap transfers with computation; warning that pinning is expensive and overuse causes problems when RAM is low (relevant: 16 GB host).
3. **PyTorch 2.13 `torch.profiler`** — https://docs.pytorch.org/docs/2.13/profiler.html — CUPTI-based CUDA kernel tracing via `ProfilerActivity.CUDA`, `schedule(wait, warmup, active)` for steady-state windows, chrome-trace export.
4. **NiBabel ArrayProxy API** — https://nipy.org/nibabel/reference/nibabel.arrayproxy.html — `ArrayProxy(mmap=True)` behaves like `mmap='c'`; "**If file_like cannot be memory-mapped, ignore mmap value and read array from file**" (gzip streams qualify), so `.nii.gz` always pays full decompression; `keep_file_open` controls handle reuse.
5. **NiBabel manual: Images and memory** — https://nipy.org/nibabel/images_and_memory.html — proxy images defer loading until array access; `get_fdata(caching='fill')` caches by default while `np.asarray(img.dataobj)` avoids filling the image cache; proxy slicing reads only needed bytes.
6. **NumPy 2.5 `numpy.memmap`** — https://numpy.org/doc/stable/reference/generated/numpy.memmap.html — memmaps "access small segments of large files on disk, without reading the entire file into memory" (applies to uncompressed `.npy`/raw, not gzip).
7. **NVIDIA Nsight Systems User Guide v2026.4** — https://docs.nvidia.com/nsight-systems/UserGuide/index.html — GPU Metrics exist to answer "Is my GPU idle? Is my GPU full? Am I blocked on IO?" via SM utilization/warp occupancy timelines; supported only on Linux x86-64/aarch64 and Windows targets (hence unavailable for this M1/macOS forensic host); focused-profiling/NVTX guidance for region-limited collection.

Community sources were not needed; no claim rests on secondary material.

## 10. Recommended performance experiments

Grouped by required change class. Nothing here is an accepted fix; each needs its stated validation.

### Class 1 — obvious behavior-preserving inefficiencies (low risk; bitwise-identical outputs expected)

- **PERF-001 · C1. Remove the provably redundant contiguous copy** of the brain mask: `torch.from_numpy(nifti_xyz_to_dhw(brain_mask_xyz).copy())` (`brats21_point_guided.py:1335`) — `nifti_xyz_to_dhw` already returns `np.ascontiguousarray`; the extra `.copy()` duplicates ~8.9 MB per sample. Validation: assert `torch.equal(old_sample.brain_mask, new_sample.brain_mask)` over fixtures; microbench copy count.
- **PERF-002 · C1. Stop re-validating what was just validated in the same call chain** where the second check cannot see different data (e.g., batch-level label set after sample-level validation with unchanged tensors). Validation: property test that invalid inputs still raise at the retained check; byte-identical batches.

### Class 2 — requires benchmark/equivalence testing (possible numerical or ordering surfaces)

- **PERF-003 · C2. Float32 statistics path or chunked percentile computation in `_normalize_masked`.** Problem: 954.63 ms median, 29.3 s/52.63 s profile share in float64. Change: keep exact policy semantics but reduce dtype/passes. Expected: largest loader opportunity; Amdahl's-law ceiling from this profile is about **2.27× end-to-end** even if normalization cost vanished, while the realistic gain is unmeasured. Risk: bit-level differences in mean/std/percentiles propagate to every normalized voxel. Validation: max-abs-diff bound vs current implementation on real server volumes; downstream loss-equivalence tolerance agreed before adoption.
- **PERF-004 · C2. Worker count / prefetch tuning on real storage (1/2/4/8)** with `persistent_workers=True` already wired. Expected: overlap decompression+preprocess with compute. Risk: 16 GB RAM × workers × ~223 MB/sample resident tensors + spawn overhead on macOS; watch swap. Validation: subjects/sec and RSS per worker; determinism of sample order preserved (seeded sampler unaffected).
- **PERF-005 · C2. Implement `PointGuidedBatch.pin_memory()`** so DataLoader pinning actually applies to the custom batch [D]. Expected: faster/overlapped H2D with existing `non_blocking=True`. Risk: pinning allocation cost and RAM pressure (official warning); no numerical change. Validation: CUDA-server A/B of H2D time and step gap; `is_pinned()` assertions.
- **PERF-006 · C2. Reduce Gate-E host syncs**: accumulate per-step counts as device tensors and convert once after the step loop (:344/:376/:392/:414/:429 booleans; :394/:421/:442 `.cpu()`). Expected: fewer pipeline drains per step; biggest effect at high k_max/candidates on CUDA. Risk: none to values if conversions move verbatim; control-flow must remain semantically identical (some branches gate tensor work). Validation: bit-equal `TrainingObjectiveResult` on fixed-seed fixtures CPU and server.
- **PERF-007 · C2. Investigate candidate batching or decoder reuse only after proving input identity.** The audit observed four decoder calls per candidate slot but did not establish duplicate inputs. Expected benefit is therefore unquantified. Risk: changed generator order, floating-point association, aliasing, or autograd behavior. Validation: first log/hash decoder inputs in a read-only server profile; only then design a bit-equivalent optimization.
- **PERF-008 · C2. AMP on the CUDA server** (infrastructure exists, `settings.amp`). Numerical change by design. Validation: metric deltas within agreed tolerance; stability over full train.
- **PERF-009 · C2. `torch.compile` trial on frontend/decoder modules**. Compatibility with grid_sample-heavy, generator-using code unknown. Validation: numerical equivalence suite + compile-time cost; fall back cleanly.
- **PERF-010 · C2. Batch-level validation cost cap** (fuse isfinite/unique into fewer passes using fused reductions). Expected: cut part of the 3.45→13.68 s collate growth. Validation: same accept/reject decisions on adversarial fixtures.

### Class 3 — architecture/protocol change (requires Human Gate)

- **PERF-011 · C3. Provenance-bound preprocessing cache (preconverted `.npy`/memmap/tensor store) keyed by source-file hash + preprocessing-version + cohort split.** Official docs support memmap partial reads [D] and uncompressed proxies honor `mmap=True` [D]; expected to collapse T_disk+decode+preprocess to near-page-fault cost. Risks: cache invalidation correctness, cohort/provenance guarantees, disk budget (96%-full filesystem), double-storage footprint. Requires plan-level protocol decision; do not implement unilaterally.
- **PERF-012 · C3. Delayed/lazy target materialization** (defer T1ce read until after the target-free context, or serve targets from the cache store). Source already keeps target use strictly post-inference [S]; changing *when bytes are read* alters data-path protocol and audit story. Cautious assessment only; requires Human Gate.
- **PERF-013 · C3. Restructuring Gate-E scheduling** (vectorizing slots across candidates/steps, fusing transitions) touches execution order of generator draws and floating-point association; preserving locked sampling semantics needs an explicit design decision, not an optimization PR.
- **PERF-014 · C3. Economic-stop policy changes** (the 1.6%-share result comes from the existing stop mechanism; altering stop criteria is a routing-protocol change, out of optimization scope).

## 11. Suggested optimizations

Condensed problem→change→impact→risk→validation matrix (classes per §10):

| # | Current bottleneck | Change | Expected impact | Risk | Validation | Class |
|---|---|---|---|---|---|---|
| 1 | float64 normalize 954.6 ms/modality | float32/chunked stats path | largest loader opportunity; ≤2.27× synthetic end-to-end ceiling, actual unknown | numeric drift | max-abs-diff bound + loss equivalence | 2 |
| 2 | 5 uncached logical loads/sample; repeated materializations | drop redundant mask copy; remove only demonstrably duplicate validations | removes known extra writes/scans | low if retained checks preserve rejection behavior | fixture equality + adversarial validation tests + microbench | 1 |
| 3 | num_workers=0 serialization | workers 1–4 + persistent + prefetch on server | overlaps ~seconds/sample with compute | RAM ×223 MB/sample, spawn cost | subjects/sec + RSS A/B | 2 |
| 4 | unpinned custom batch | add `PointGuidedBatch.pin_memory()` | overlapped H2D | RAM pressure from pinning | CUDA A/B + `is_pinned()` | 2 |
| 5 | Gate-E syncs + four decoder calls/slot | deferred count conversion; profile candidate batching/reuse only after proving identical inputs | fewer proven drains; decoder-call benefit unquantified | control-flow, RNG, and numerical equivalence | bit-equal objective on seeds plus decoder-input identity evidence | 2 |
| 6 | gzip decode + preprocess every epoch | hashed npy/memmap cache store | near-elimination of T_disk+preprocess | invalidation, disk 96% full, protocol | Human Gate first; then hash-invalidation tests | 3 |
| 7 | eager T1ce read per sample | delayed target read post-context | small I/O shift; audit-relevant | target-free boundary optics | protocol review | 3 |
| 8 | fp32 CUDA step cost | benchmark existing AMP flag | impact unknown until server measurement | numerics | tolerance-gated server A/B | 2 |

Determinism, split/provenance, locked point behavior, final-Z-only decoder, and the target-free inference boundary are treated as hard constraints in every row.

## 12. Verdict

- **Is loading 3 volumes an I/O bottleneck? NOT YET PROVEN.**
  The synthetic gzip proxy was modest (median 71 ms per MRI volume), while the surrounding CPU pipeline was much larger: five logical load calls per sample with no cache, 954.63 ms median float64 normalization per modality, repeated full-volume materializations, duplicated validation, and 18.1 s/sample end-to-end at `num_workers=0` under severe host load. This is evidence that CPU preprocessing is a serious candidate bottleneck, not proof of the real server's storage/NiBabel share. Exact physical filesystem operations, real `.nii.gz` decode, page-cache state, and server storage latency remain unmeasured; the NiBabel proxy behavior only establishes why gzip cannot use memmap.
- **Is CPU preprocessing starving the GPU? NOT YET PROVEN.**
  No CUDA device exists locally; Nsight GPU metrics do not run on macOS [D]; no H2D/profiler overlap was measurable. Structural evidence (single-process loader blocking compute by documented PyTorch semantics; a custom batch that does not reach the documented pinned-memory overlap path; seconds-scale synthetic loader vs sub-second model stages; per-step Gate-E host syncs) predicts starvation risk on a CUDA server, but that prediction requires server-side `torch.profiler`/Nsight confirmation before being treated as fact.
- **Is Gate E a major training bottleneck? YES (on measured CPU-synthetic evidence), with scope caveats.**
  Gate E's share of the step rises monotonically with schedule depth and candidate count — 19.9% (kmax1) → 69.6–69.9% (kmax8/16) → 79.2% (32 candidates) — driven by the per-step × per-candidate counterfactual loop with four decoder queries per slot and multiple host syncs per step (`training_objective.py:342-442`; `reward_supervision.py:677-726`). The same measurements show it is schedule-sensitive rather than intrinsic: with economic stop at kmax16, Gate E falls to 1.6% of the step. Caveats: all timings are CPU-only, small-batch, noisy under host saturation, and GPU-side cost distribution is unproven; absolute candidate-count medians must not be compared across runs. No finding weakens the target-after-inference boundary or authorizes remediation without the class-appropriate gates in §10.

---

*End of report. Investigated and drafted by the independent Ox Alpha forensics worker (task_b0b8cd8c7754, dispatch ctx_f1459572f316), then evidence-bounded by Sol as noted above. Only this report was created or edited for the performance-forensics lane; all other dirty/untracked repository state predates it.*
