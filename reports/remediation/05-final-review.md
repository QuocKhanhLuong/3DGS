# Final Remediation Review

Audit base: `0efeb94af72ffa067769e19afcd19ad358feefd2`  
Branch: `remediation/main-0efeb94`  
Review result: **CORRECTNESS READY; SERVER PERFORMANCE EVIDENCE BLOCKED.**

## Sol integration verdict

All accepted correctness issues are implemented and the final post-edit suite
passes. MAIN-013 remains deferred. MAIN-PROB-001 was not reproduced and was
not changed. MAIN-PROB-002/003/004 were reproduced and narrowly remediated.

The first MAIN-012 implementation passed local tests but broke the supported
AMP contract. The independent system review identified the interaction:
autocast lowers dynamic state while RAS coordinates remain FP32. Remediation
was corrected to preserve FP32 geometry, explicitly promote only latent state
at sampling, and reject unsupported pairs before sampling or the decoder MLP.
The independent re-review returned PASS.

## Independent system review

DeepSeek was not configured. The Codex `reviewer` role (`gpt-5.6-sol`, high)
was used as the disclosed strongest available independent adversarial
substitute. It reviewed provenance, launchers, artifacts, checkpoint/resume
behavior, target/model boundaries, module modes, dtypes, routing/docs, and
PERF-001.

Final result after the MAIN-012 continuation: **PASS**. No other supported
cross-fix regression was found.

## Claude evidence-quality pass

Claude completed the final proofread and returned PASS on substance. Its ten
editorial corrections were incorporated: PERF-009 is explicitly unauthorized,
the experiment adoption rule no longer applies retroactively to PERF-001,
CUDA RNG handling and the manual DDP timing are properly scoped, reviewer and
orchestration provenance are named, and stale/deferred status language is
corrected. Claude made no code changes and introduced no new finding.

## Verification

- Final repository suite: `678 passed, 42 skipped, 26 subtests passed`.
- Warning: one tensor-with-grad to scalar warning in `tests/test_torch_compat.py`.
- Focused MAIN-012 integration: `75 passed, 2 skipped`.
- Shell syntax: all `scripts/point_guided_*.sh` passed `bash -n`.
- Task routing: every declared entrypoint/read path validated.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.

Skipped tests and the performance block are not treated as CUDA/server proof.
The retained evidence gap for MAIN-PROB-001 is that the bounded Gloo/elastic
failure injection was manual rather than a committed regression test.
Rank-indexed CUDA RNG handling is implemented but was not executed on this
CUDA-less host; only Python, NumPy, CPU Torch, and DataLoader worker streams
were runtime-verified.

## Performance review

The actual-server gate could not run: no connected server, CUDA, NVIDIA tools,
NiBabel, or real volumes were available, and OpenCode did not expose Ox Alpha.
Therefore samples/sec, loader wait, GPU idle/utilization, Gate-E CUDA share,
and dominant server CPU/GPU stages remain unmeasured.

Only PERF-001 is adopted. PERF-002 is rejected because the retained scan is a
public trust-boundary validation. PERF-003/004/005/006/007/008/010 need real
server evidence. PERF-011/012/013/014 remain Human Gate decisions.

## Readiness

- Software correctness remediation: **READY FOR HUMAN REVIEW**.
- Gate F3/F4 execution: **PENDING**.
- Real trained checkpoint and held-out Gate G evidence: **PENDING**.
- Server performance claims/optimizations: **PENDING REAL-SERVER EVIDENCE**.
- Protocol-changing performance work: **BLOCKED ON SECOND HUMAN GATE**.
