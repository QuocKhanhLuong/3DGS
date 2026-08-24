# Claude Editorial Review — Sol-High Technical Report

## Review basis

- Frozen HEAD: `0efeb94af72ffa067769e19afcd19ad358feefd2`
- Subject: `reports/audit/04-sol-technical-review.md` (consolidated draft)
- Corroboration inputs: `01-sol-plan.md`, `03-deepseek-system-review.md`, spot-checked
  live code for every evidence citation that was auditable without execution.
- Mutation boundary: this file only.

## Overall verdict

**READY WITH FOUR MINOR AMENDMENTS.** The Sol draft is structurally sound. The
evidence citations are specific and spot-checkable; severity demotions are
justified; false-positive and design-decision callouts are well-reasoned; the
GPT-OSS substitution is disclosed and its overreach is explicitly rejected; the
probable/confirmed distinction is maintained with epistemic discipline. The four
issues below are targeted edits, not structural changes.

---

## Issue 1 — Incomplete demotion parenthetical in MAIN-PROB-004

**Location:** Line that reads `Source: AGY-D-FIND-006 (severity demoted).`

**Problem:** Two other demotion parentheticals in the same document specify the
source severity: `AGY-B-FIND-001 (worker severity demoted from P1)` and
`AGY-A-FIND-001 (worker severity demoted from P2)`. The MAIN-PROB-004
parenthetical omits the source level. The GPT-OSS review placed SYS-DEF-012
(the resource-cleanup finding) at P2; the Sol demotes it to P3 and labels it
probable. A reader cannot trace the demotion chain without consulting the
DeepSeek report.

**Recommended fix:** Change `(severity demoted)` to `(worker severity demoted
from P2)`.

---

## Issue 2 — MAIN-001 evidence does not name the three config files

**Location:** MAIN-001 evidence block, third bullet.

**Problem:** The text says "all three checked-in server profiles set
`offset_hidden_channels` to 128" but the evidence line names only authority
documents (`AGENTS.md:200-207`, `PLAN.md:1476-1483`, `baseline_training.py:118`,
and the test). The actual config files are not cited by path. Independent
verification confirms the value at:

- `configs/training/point_guided_brats21_4070.json:16`
- `configs/training/point_guided_brats21_2xa4000.json:16`
- `configs/training/point_guided_brats21_overfit.json:16`

Without these paths, a remediation author cannot confirm which files to change.
The DeepSeek review (Section 4.1) named them; the Sol should carry them
forward.

**Recommended fix:** Add those three config paths to the MAIN-001 evidence
bullet. No other change needed.

---

## Issue 3 — DistributedSampler disagreement with GPT-OSS is implicit

**Location:** "Design decisions and rejected hypotheses" section, first
DESIGN_DECISION_REQUIRED bullet.

**Problem:** The Sol correctly designates DistributedSampler padding as a
design decision rather than a P2 defect. GPT-OSS (SYS-DEF-011) rated it P2
Major and recommended `drop_last=True`. The Sol's reasoning — "standard DDP
behavior that keeps collective counts equal" — is correct. But the disagreement
is invisible to a reader who has not also read the DeepSeek report. Because the
plan grants GPT-OSS second-opinion status (not authority), the Sol's
disagreement should be legible without reference to the other document.

**Recommended fix:** Add one sentence: "GPT-OSS classified this as P2 (SYS-DEF-011)
and recommended `drop_last=True`; that recommendation is rejected because
`drop_last` discards subjects and is not a governance-approved change."

---

## Issue 4 — MAIN-008/MAIN-PROB-003 cross-reference is forward-looking

**Location:** MAIN-008 evidence, last sentence: "`AGY-G-003` is a probable
artifact subcase."

**Problem:** A reader scanning MAIN-008 encounters a subcase reference but must
search forward to find MAIN-PROB-003 to understand it. The linkage exists but
is one-directional. MAIN-PROB-003 in turn says only `Source: AGY-G-003` without
noting that MAIN-008 treats it as a parent. The round-trip lookup is minor but
adds friction for the remediation author.

**Recommended fix:** Change the MAIN-008 sentence to "AGY-G-003 is a probable
artifact subcase (see MAIN-PROB-003)." No change needed in MAIN-PROB-003 itself.

---

## Confirmations — no change required

The following judgments are correct and should be kept verbatim.

**GPT-OSS substitution disclosure.** The review-basis statement that DeepSeek
was unavailable, GPT-OSS 120B was substituted, and it is treated as a second
opinion not an authority is properly placed and worded. The disclosure is
present at the start of the document where a reader naturally looks.

**"Mathematically sound" rejection.** GPT-OSS Section 8 concludes the
architecture is "mathematically sound." The Sol correctly places this in the
UNSUPPORTED CLAIM REJECTED section. Code inspection cannot establish mathematical
soundness. The rejection language is proportionate and does not overreach in the
opposite direction.

**AGY-B-FIND-001 demotion from P1 to P2.** The justification — CLI normally
exits; public-API risk is real but bounded — is sound and includes a
reproduction. Demotion is appropriate.

**AGY-A-FIND-001 demotion from P2 to P3.** Phantom CODEGRAPH paths have zero
runtime effect. P3 is correct.

**AGY-D-FIND-005 (DistributedSampler) as DESIGN_DECISION_REQUIRED.** Standard
DDP behavior. Not a defect absent an explicit governance decision to accept
subject weighting change. Correct classification (subject to the wording fix in
Issue 3).

**Incomplete-export claim rejected.** AGY-A-FIND-003 / SYS-DOC-001 (incomplete
`__init__.py` exports) correctly classified as DESIGN_DECISION_REQUIRED rather
than a defect. Deep submodule imports are the intentional pattern; the export
gap is not a runtime bug.

**AGY-E-FIND-003 and AGY-E-FIND-004 as NOT A DEFECT.** Single-pass evaluation
without per-subject resume is a future capability, not a present contract
failure. Dice 1.0 on two empty masks is the documented convention. Both
classifications are correct.

**AGY-C qualified pass.** The current-flow table accurately captures AGY-C's
finding: NIfTI affine, target decoupling, and atomic replacement are aligned,
but the split hash check is opaque string validation. The row correctly says
"Aligned in software; real NIfTI execution was unavailable."

**Probable findings labeled PROBABLE.** All four MAIN-PROB entries include a
"Needed reproduction" subsection and a "Status: PROBABLE" sentence. No probable
finding is presented as confirmed. Correct epistemic posture.

**MAIN-004 collapse of AGY-D-FIND-002 and AGY-F-FIND-003.** Both findings share
the same root cause (non-exclusive run directories and fixed atomic-temp names);
collapsing them is justified and the evidence from both workers is retained.

**The eight `8-thread/40-write` reproduction evidence for MAIN-004.** Concrete
and specific. Keep.

**Remediation order.** The sequencing (locked model config → provenance →
launcher contract → DDP teardown → namespace → checkpoint/resume → ownership
seams → hygiene) is logically sound and matches severity.

**Human Gate stop.** "No remediation is authorized before the Human Gate" is
correctly placed at the end of the document and should remain.

---

## Checklist summary

| Dimension | Status |
| --- | --- |
| Technical clarity | Pass |
| Duplicate findings | Pass — MAIN-013 dedup explicit |
| Unsupported claims | Pass — GPT-OSS overreach rejected |
| Severity inflation | Pass with Issue 1 (missing demotion source) |
| Contradictions | Pass |
| Missing evidence | Issue 2 (config file paths absent from MAIN-001) |
| Reproduction clarity | Pass — PROB entries have explicit reproduction steps |
| Stable IDs | Pass — MAIN-### scheme consistent throughout |
| Current-flow readability | Pass |
| Docs-vs-code status | Pass — MAIN-DOC-001 scoped as drift, not code defect |
| DeepSeek-unavailable / GPT-OSS substitution disclosure | Pass |

No new findings are introduced. No existing findings are promoted or demoted
beyond the four targeted amendments above.
