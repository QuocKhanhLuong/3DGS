from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "quality" / "checklists.json"
RUNNER = ROOT / "scripts" / "check_phase.py"
QUALITY_README = ROOT / "quality" / "README.md"


def _load_runner():
    spec = importlib.util.spec_from_file_location("point_guided_gate_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_metadata() -> dict[str, object]:
    return {"commit": "abc123", "dirty": False, "dirty_entries": []}


def test_catalog_has_one_minimal_point_guided_gate() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)

    assert set(catalog) == {"schema_version", "gate"}
    assert catalog["schema_version"] == 1
    assert "phases" not in catalog
    assert "global_checks" not in catalog
    gate = catalog["gate"]
    assert gate["id"] == "POINT_GUIDED_FRONTEND"
    assert gate["human_gate"]["status"] == "pending"
    assert len(gate["automated_checks"]) == 3
    assert {check["id"] for check in gate["automated_checks"]} == {
        "PGF-CPU-TESTS-001",
        "PGF-COMPILE-001",
        "PGF-DIFF-001",
    }
    assert "docs/" not in json.dumps(catalog)


def test_catalog_rejects_any_attempt_to_approve_the_human_gate() -> None:
    runner = _load_runner()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog = deepcopy(catalog)
    catalog["gate"]["human_gate"]["status"] = "passed"

    with pytest.raises(runner.CatalogError, match="immutable pending"):
        runner._validate_catalog(catalog)


def test_runner_lists_the_single_gate_without_executing_checks() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--list"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "POINT_GUIDED_FRONTEND" in completed.stdout
    assert "pending" in completed.stdout


def test_dry_run_keeps_automated_evidence_unrun_and_human_gate_pending() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)

    report = runner._build_report(catalog, run=False)

    assert report["automated_verdict"] == "NOT_RUN"
    assert report["gate_verdict"] == "NOT_RUN"
    assert all(result["status"] == "NOT_RUN" for result in report["results"])
    assert report["human_gate"]["status"] == "pending"
    assert report["human_gate"]["decision"] is None
    assert report["human_gate"]["decided_by"] is None


def test_automated_pass_still_requires_a_human_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    calls: list[list[str]] = []

    def fake_run(command: list[str], timeout_seconds: int) -> dict[str, object]:
        calls.append(command)
        return {
            "status": "PASS",
            "command": command,
            "return_code": 0,
            "duration_seconds": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(runner, "_git_metadata", _clean_metadata)
    monkeypatch.setattr(runner, "_run_process", fake_run)
    report = runner._build_report(catalog, run=True)

    assert len(calls) == len(catalog["gate"]["automated_checks"])
    assert report["automated_verdict"] == "PASS"
    assert report["gate_verdict"] == "PENDING_HUMAN_GATE"
    assert report["human_gate"]["status"] == "pending"
    assert report["human_gate"]["decision"] is None
    assert report["human_gate"]["decided_by"] is None
    assert "Decision: not recorded by the automated runner." in runner._render_markdown(report)


def test_dirty_tree_blocks_execution_without_allow_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    dirty = {"commit": "abc123", "dirty": True, "dirty_entries": [" M src/example.py"]}

    def should_not_run(command: list[str], timeout_seconds: int) -> dict[str, object]:
        pytest.fail(f"dirty-tree run executed {command}")

    monkeypatch.setattr(runner, "_git_metadata", lambda: dirty)
    monkeypatch.setattr(runner, "_run_process", should_not_run)
    report = runner._build_report(catalog, run=True)

    assert report["automated_verdict"] == "BLOCKED"
    assert report["gate_verdict"] == "BLOCKED"
    assert report["repository"]["dirty_execution_allowed"] is False
    assert any(result["id"] == "PGF-CLEAN-TREE-001" for result in report["results"])


def test_allow_dirty_executes_development_evidence_without_approving(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    dirty = {"commit": "abc123", "dirty": True, "dirty_entries": [" M src/example.py"]}

    def fake_run(command: list[str], timeout_seconds: int) -> dict[str, object]:
        return {
            "status": "PASS",
            "command": command,
            "return_code": 0,
            "duration_seconds": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(runner, "_git_metadata", lambda: dirty)
    monkeypatch.setattr(runner, "_run_process", fake_run)
    report = runner._build_report(catalog, run=True, allow_dirty=True)

    assert report["automated_verdict"] == "PASS"
    assert report["repository"]["dirty_execution_allowed"] is True
    assert report["gate_verdict"] == "PENDING_HUMAN_GATE"
    assert report["human_gate"]["decision"] is None


def test_runner_uses_the_invoking_python_for_catalogued_python_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    seen: dict[str, list[str]] = {}

    def fake_run(command: list[str], timeout_seconds: int) -> dict[str, object]:
        seen["command"] = command
        return {"status": "PASS"}

    monkeypatch.setattr(runner, "_run_process", fake_run)
    result = runner._evaluate_check(
        {
            "id": "PGF-TEST-001",
            "description": "test",
            "command": ["python", "-m", "pytest", "-q", "tests/features/point_guided"],
            "timeout_seconds": 10,
        },
        run=True,
        execution_blocked=False,
    )

    assert result["status"] == "PASS"
    assert seen["command"][0] == sys.executable


def test_quality_readme_describes_only_the_current_gate() -> None:
    text = QUALITY_README.read_text(encoding="utf-8")
    assert "POINT_GUIDED_FRONTEND" in text
    assert "docs/" not in text
