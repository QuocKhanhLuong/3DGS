from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_FILES = [ROOT / name for name in ("README.md", "AGENTS.md", "CODEBASE.md")]
for directory in ("docs", "quality"):
    MARKDOWN_FILES.extend((ROOT / directory).rglob("*.md"))

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]*)\)")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
STALE_STATUS_PHRASES = (
    "T1-B remains blocked",
    "T1-B Human Gate is still pending",
    "T1-A remains blocked",
    "NO ACTIVE RUN",
    "no executable package",
)
ACTIVE_STATUS_FILES = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "CODEBASE.md",
    ROOT / "docs" / "codex" / "README.md",
    ROOT / "docs" / "codex" / "T1A_EXECUTABLE_REFERENCE.md",
    ROOT / "docs" / "codex" / "T1B_TEACHER_FREE_ENCODER.md",
    ROOT / "docs" / "strategies" / "2026-07-29-isbi-realignment.md",
    ROOT / "docs" / "strategies" / "2026-07-31-execution-status-addendum.md",
    ROOT / "docs" / "strategies" / "2026-07-31-t1b-human-gate-decision.md",
    ROOT / "docs" / "reconstruction" / "README.md",
    ROOT / "docs" / "reconstruction" / "FULL_FLOW.md",
    ROOT / "quality" / "README.md",
)
RETIRED_PATHS = (
    "docs/strategies/2026-07-29-cvpr-priorities.md",
    "docs/research/2026-07-29-cvpr-external-signals.md",
    "docs/designs/2026-07-29-cvpr-internal-health.md",
    "docs/meetings/20260729T104239Z-deepdive.md",
    "docs/reconstruction/PROOFREAD_NOTES.md",
    "docs/plans/2026-07-29-t05-t1-teacher-free-encoder-fixed-gaussian-baseline.md",
    "docs/designs/2026-07-29-t05-t1-isbi-design-delta.md",
)


def _local_link_targets(path: Path) -> list[tuple[int, Path]]:
    """Return relative Markdown destinations outside fenced code blocks."""

    targets: list[tuple[int, Path]] = []
    in_fence = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK_PATTERN.finditer(line):
            destination = match.group(1).strip()
            if destination.startswith("<"):
                closing = destination.find(">")
                if closing < 0:
                    continue
                destination = destination[1:closing]
            else:
                destination = destination.split(maxsplit=1)[0]
            if not destination or destination.startswith("#"):
                continue
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", destination):
                continue
            destination = destination.split("#", 1)[0]
            if not destination:
                continue
            targets.append((line_number, (path.parent / destination).resolve()))
    return targets


def _unresolved_links(path: Path) -> list[str]:
    return [
        f"{path.relative_to(ROOT)}:{line_number} -> {target}"
        for line_number, target in _local_link_targets(path)
        if not target.exists()
    ]


def test_relative_markdown_links_resolve_and_code_blocks_are_ignored() -> None:
    failures = [failure for path in MARKDOWN_FILES for failure in _unresolved_links(path)]
    assert not failures, "\n".join(failures)


def test_link_parser_ignores_urls_mailto_anchors_and_fenced_code(tmp_path: Path) -> None:
    (tmp_path / "exists.md").write_text("# valid\n", encoding="utf-8")
    sample = tmp_path / "sample.md"
    sample.write_text(
        "[web](https://example.com) [mail](mailto:test@example.com) [anchor](#part)\n"
        "```\n[code](missing.md)\n```\n"
        "[local](exists.md)\n",
        encoding="utf-8",
    )
    targets = _local_link_targets(sample)
    assert targets == [(5, (tmp_path / "exists.md").resolve())]


def test_docs_index_exists_and_all_active_map_links_resolve() -> None:
    index = ROOT / "docs" / "README.md"
    assert index.is_file()
    assert _unresolved_links(index) == []


def test_retired_documents_are_absent() -> None:
    assert all(not (ROOT / path).exists() for path in RETIRED_PATHS)


def test_active_status_documents_have_no_stale_phase_claims() -> None:
    for path in ACTIVE_STATUS_FILES:
        contents = path.read_text(encoding="utf-8")
        for phrase in STALE_STATUS_PHRASES:
            assert phrase not in contents, f"{phrase!r} remains in {path.relative_to(ROOT)}"


def test_historical_and_immutable_records_remain_distinct_from_status_checks() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    gate_record = (ROOT / "docs/strategies/2026-07-31-t1b-human-gate-decision.md").read_text(
        encoding="utf-8"
    )
    assert "Historical Unreleased Notes" in changelog
    assert "T1-B Human Gate: `PASS`" in gate_record
