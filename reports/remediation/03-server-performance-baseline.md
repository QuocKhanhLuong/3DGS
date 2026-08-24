# Server Performance Baseline

Audit base: `0efeb94af72ffa067769e19afcd19ad358feefd2`  
Working branch: `remediation/main-0efeb94`  
Result: **ENVIRONMENT_BLOCKED — no server baseline was collected.**

## Required environment check

| Requirement | Observed | Result |
| --- | --- | --- |
| Actual training server | No saved/connected Orca remote environment | unavailable |
| CUDA / NVIDIA tooling | Apple M1; `torch.cuda.is_available() == False`; zero CUDA devices; no `nvidia-smi` | unavailable |
| Real BraTS-style volumes | No `.nii` or `.nii.gz` volumes found in the repository or the bounded user-directory search | unavailable |
| NiBabel | Import fails with `ModuleNotFoundError` in the repository venv | unavailable |
| Target storage | Local APFS only, 228 GiB total, 11 GiB available, 95% used | not the target server |
| OpenCode harness | OpenCode 1.18.21 installed | available |
| Ox Alpha | Not listed by `opencode models opencode`; `opencode/ox-alpha` returns `Model not found` | requested independent worker unavailable |

The local runtime is Python 3.10.0, PyTorch 2.13.0, NumPy 2.2.6 on
macOS arm64 with 16 GiB RAM. MPS is available, but it is not evidence for the
required CUDA server path and was not substituted.

Ox Alpha unavailability prevented the requested independent OpenCode worker,
but it is not the reason measurements were impossible. Even with another
worker, the absence of the actual server, CUDA, NiBabel, real volumes, and
target storage independently blocks the required benchmark.

## MCP enumeration

OpenCode itself exposed exactly one connected MCP server:

- `semble` via `uvx --from semble[mcp]==0.5.5 semble`

No Serena, Morph, documentation, or external-research MCP was configured in
OpenCode. The surrounding Codex harness exposed Semble search/find-related and
a CodeGraph tool, but this repository has no `.codegraph/` index, so CodeGraph
was not used for the performance gate. Generic MCP resource templates were
empty. Playwright was not relevant.

## Required baseline metrics

No numeric value is reported because the required data, decoder dependency,
CUDA device, storage, and Ox Alpha worker were unavailable.

| Metric | Result |
| --- | --- |
| `T_disk` | NOT MEASURED |
| `T_nifti_decode` | NOT MEASURED |
| `T_numpy_preprocess` | NOT MEASURED |
| `T_validation` | NOT MEASURED |
| `T_collate` | NOT MEASURED |
| `T_dataloader_wait` | NOT MEASURED |
| `T_h2d` | NOT MEASURED |
| `T_frontend` | NOT MEASURED |
| `T_gate_c` | NOT MEASURED |
| `T_gate_d` | NOT MEASURED |
| `T_gate_e` | NOT MEASURED |
| `T_backward` | NOT MEASURED |
| `T_optimizer` | NOT MEASURED |
| `T_total_step` | NOT MEASURED |
| samples/sec | NOT MEASURED |
| GPU utilization / idle | NOT MEASURED |
| host / worker RSS | NOT MEASURED |

## Evidence boundary

The accepted audit report
`reports/audit/04-ox-alpha-performance-forensics.md` contains static tracing
and CPU-synthetic measurements from the same non-CUDA Mac. It remains useful
investigation evidence, but it is not promoted to real-server storage, CUDA,
H2D, DataLoader-overlap, or GPU-starvation evidence.

## Verdict

- Is storage itself limiting throughput? **NOT YET PROVEN**.
- Is CPU preprocessing starving the GPU? **NOT YET PROVEN**.
- Is Gate E a CUDA training bottleneck? **NOT YET PROVEN**.

The server performance gate must be rerun in the configured CUDA/BraTS
environment before any throughput claim or production performance adoption.
