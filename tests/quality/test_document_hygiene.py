from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_FILES = [ROOT / name for name in ("README.md", "AGENTS.md", "CODEBASE.md", "CHANGELOG.md")]
for directory in ("docs", "quality"):
    MARKDOWN_FILES.extend((ROOT / directory).rglob("*.md"))

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]*)\)")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")


def _local_link_targets(path: Path) -> list[tuple[int, Path]]:
    targets: list[tuple[int, Path]] = []
    in_fence = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK_PATTERN.finditer(line):
            destination = match.group(1).strip().split("#", 1)[0]
            if not destination or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", destination):
                continue
            targets.append((line_number, (path.parent / destination).resolve()))
    return targets


def test_relative_markdown_links_resolve() -> None:
    failures = [
        f"{path.relative_to(ROOT)}:{line_number} -> {target}"
        for path in MARKDOWN_FILES
        for line_number, target in _local_link_targets(path)
        if not target.exists()
    ]
    assert not failures, "\n".join(failures)


def test_new_documentation_index_and_codegraph_exist() -> None:
    assert (ROOT / "docs" / "README.md").is_file()
    graph = json.loads((ROOT / "CODEGRAPH.json").read_text(encoding="utf-8"))
    assert graph["schema_version"] == 1
    assert {"frontend", "medicalnet", "data-boundary-audit", "tests", "quality"} <= set(graph["tasks"])


def test_active_docs_do_not_describe_the_retired_gaussian_direction() -> None:
    active = "\n".join(path.read_text(encoding="utf-8") for path in MARKDOWN_FILES)
    assert "Sparse Support-Anchor Gaussian Reconstruction" not in active
    assert "T1-B Human Gate" not in active
