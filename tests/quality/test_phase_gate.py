from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "quality" / "checklists.json"
RUNNER = ROOT / "scripts" / "check_phase.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("phase_gate_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_checks(catalog: dict) -> list[dict]:
    checks = list(catalog["global_checks"])
    for phase in catalog["phases"].values():
        checks.extend(phase["checks"])
    return checks


def test_catalog_contains_all_phases_and_exact_status_pairs() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    assert list(catalog["phases"]) == ["T0", "T05", "T1A", "T1B", "T1C", "T2", "T3", "T4", "T5"]

    for phase_name in ("T0", "T05", "T1A"):
        phase = catalog["phases"][phase_name]
        assert phase["implementation_status"] == "implemented"
        assert phase["human_gate_status"] == "retrospective_unrecorded"
    assert catalog["phases"]["T1B"]["implementation_status"] == "implemented"
    assert catalog["phases"]["T1B"]["human_gate_status"] == "passed"
    assert catalog["phases"]["T1B"]["human_gate_record"]
    for phase_name in ("T1C", "T2", "T3", "T4", "T5"):
        phase = catalog["phases"][phase_name]
        assert phase["implementation_status"] == "planned"
        assert phase["human_gate_status"] == "blocked"
        assert any(check["mode"] == "planned" for check in phase["checks"])


def test_status_and_verdict_vocabularies_are_separate_and_valid() -> None:
    runner = _load_runner()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert set(catalog["allowed_implementation_statuses"]) == runner.IMPLEMENTATION_STATUSES
    assert set(catalog["allowed_human_gate_statuses"]) == runner.HUMAN_GATE_STATUSES
    assert set(catalog["allowed_automated_verdicts"]) == runner.AUTOMATED_VERDICTS
    assert set(catalog["allowed_phase_verdicts"]) == runner.PHASE_VERDICTS


def test_every_reviewer_role_maps_to_a_configured_codex_agent() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    configured = {path.stem for path in (ROOT / ".codex" / "agents").glob("*.toml")}
    reviewer_roles = {
        check["reviewer_role"]
        for check in _all_checks(catalog)
        if check["mode"] == "human"
    }
    assert reviewer_roles <= configured
    assert {"pm", "reviewer", "qa", "reproducibility_auditor"} <= reviewer_roles


def test_blocker_failure_cannot_be_scored_away() -> None:
    runner = _load_runner()
    results = [
        {"mode": "command", "severity": "blocker", "status": "FAIL"},
        *({"mode": "command", "severity": "minor", "status": "PASS"} for _ in range(20)),
    ]
    assert runner._automated_verdict("implemented", results, run=True) == "FAIL"


def test_aliases_are_stable() -> None:
    runner = _load_runner()
    assert runner._normalize_phase("T0.5") == "T05"
    assert runner._normalize_phase("t1-a") == "T1A"
    assert runner._normalize_phase("T1-B") == "T1B"
    assert runner._normalize_phase("t5") == "T5"


def test_runner_lists_catalog_without_executing_phase_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--list"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "T1B" in completed.stdout
    assert "implemented" in completed.stdout
    assert "passed" in completed.stdout
    assert "T5" in completed.stdout


def test_t1b_dry_run_does_not_rerun_automated_checks() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    report = runner._build_report(catalog, "T1B", run=False)
    assert report["automated_verdict"] == "NOT_RUN"
    assert report["phase_verdict"] == "NOT_RUN"
    assert report["human_gate_status"] == "passed"
    assert report["human_gate_record"]
    assert report["final_human_gate"]["decision"] is None


def test_t1b_automated_pass_respects_recorded_human_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)

    def fake_evaluate(check: dict, run: bool, execution_blocked: bool = False) -> dict:
        status = "PENDING_HUMAN" if check["mode"] == "human" else "PLANNED" if check["mode"] == "planned" else "PASS"
        return {"id": check["id"], "category": check["category"], "severity": check["severity"], "mode": check["mode"], "description": check["description"], "status": status, **({"reviewer_role": check["reviewer_role"]} if check["mode"] == "human" else {})}

    monkeypatch.setattr(runner, "_evaluate_check", fake_evaluate)
    report = runner._build_report(catalog, "T1B", run=True, allow_dirty=True)
    assert report["automated_verdict"] == "PASS"
    assert report["phase_verdict"] == "PASS"


def test_planned_phase_reports_blocked_without_claiming_failure() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    report = runner._build_report(catalog, "T5", run=False)
    assert report["automated_verdict"] == "BLOCKED"
    assert report["phase_verdict"] == "BLOCKED"
    assert report["human_gate_status"] == "blocked"
    assert all(item["status"] != "PASS" for item in report["results"] if item["mode"] == "planned")


def test_dirty_tree_blocks_run_unless_allow_dirty_is_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    dirty = {"commit": "abc", "dirty": True, "dirty_entries": [" M file"]}
    monkeypatch.setattr(runner, "_git_metadata", lambda: dirty)
    blocked = runner._build_report(catalog, "T1B", run=True)
    assert blocked["automated_verdict"] == "BLOCKED"
    assert blocked["repository"]["dirty_execution_allowed"] is False
    assert any(item["id"] == "G-REPO-CLEAN-001" for item in blocked["results"])


def test_allow_dirty_is_explicit_development_only(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    dirty = {"commit": "abc", "dirty": True, "dirty_entries": [" M file"]}
    monkeypatch.setattr(runner, "_git_metadata", lambda: dirty)

    def fake_evaluate(check: dict, run: bool, execution_blocked: bool = False) -> dict:
        status = "PENDING_HUMAN" if check["mode"] == "human" else "PLANNED" if check["mode"] == "planned" else "PASS"
        return {"id": check["id"], "category": check["category"], "severity": check["severity"], "mode": check["mode"], "description": check["description"], "status": status, **({"reviewer_role": check["reviewer_role"]} if check["mode"] == "human" else {})}

    monkeypatch.setattr(runner, "_evaluate_check", fake_evaluate)
    report = runner._build_report(catalog, "T1B", run=True, allow_dirty=True)
    assert report["automated_verdict"] == "PASS"
    assert report["repository"]["dirty_execution_allowed"] is True


def test_report_captures_pre_and_post_git_state(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    states = iter(
        [
            {"commit": "before", "dirty": False, "dirty_entries": []},
            {"commit": "after", "dirty": False, "dirty_entries": []},
        ]
    )
    monkeypatch.setattr(runner, "_git_metadata", lambda: next(states))
    report = runner._build_report(catalog, "T1B", run=False)
    assert report["repository"]["before"]["commit"] == "before"
    assert report["repository"]["after"]["commit"] == "after"
    assert report["repository"]["dirty_before"] is False
    assert report["repository"]["dirty_after"] is False


def test_staged_and_unstaged_diff_checks_are_catalogued() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    ids = {check["id"] for check in catalog["global_checks"]}
    assert {"G-DIFF-001", "G-DIFF-002"} <= ids
    commands = {check["id"]: check.get("command") for check in catalog["global_checks"]}
    assert commands["G-DIFF-001"] == ["git", "diff", "--check"]
    assert commands["G-DIFF-002"] == ["git", "diff", "--cached", "--check"]


def test_runner_uses_current_python_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    seen: dict[str, list[str]] = {}

    def fake_run(command: list[str], timeout_seconds: int) -> dict[str, str]:
        seen["command"] = command
        return {"status": "PASS"}

    monkeypatch.setattr(runner, "_run_process", fake_run)
    result = runner._evaluate_check(
        {"id": "X", "category": "test", "severity": "blocker", "mode": "pytest", "description": "x", "target": "tests/quality"},
        run=True,
    )
    assert result["status"] == "PASS"
    assert seen["command"][0] == sys.executable


def test_automated_report_does_not_write_a_decision_field() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    report = runner._build_report(catalog, "T1B", run=False)
    rendered = runner._render_markdown(report)
    assert report["final_human_gate"]["decision"] is None
    assert report["final_human_gate"]["status"] == "passed"
    assert "Decision: not recorded by the automated runner." in rendered
    assert "Decision: `PASS`" not in rendered


def test_generated_reports_are_ignored_by_git() -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "quality/reports/T1B-example.json"],
        cwd=ROOT,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
