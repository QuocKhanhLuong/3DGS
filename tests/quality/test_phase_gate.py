from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "quality" / "checklists.json"
RUNNER = ROOT / "scripts" / "check_phase.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("phase_gate_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_is_valid_and_covers_every_phase_through_t5() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    assert list(catalog["phases"]) == ["T0", "T05", "T1A", "T1B", "T1C", "T2", "T3", "T4", "T5"]
    assert catalog["phases"]["T1B"]["implementation_status"] == "active"
    for phase_name in ("T1C", "T2", "T3", "T4", "T5"):
        assert catalog["phases"][phase_name]["implementation_status"] == "planned"
        assert any(check["mode"] == "planned" for check in catalog["phases"][phase_name]["checks"])


def test_all_check_ids_are_unique_and_blockers_cannot_be_scored_away() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    checks = list(catalog["global_checks"])
    for phase in catalog["phases"].values():
        checks.extend(phase["checks"])
    identifiers = [check["id"] for check in checks]
    assert len(identifiers) == len(set(identifiers))
    assert "no_score_compensation" in catalog["human_gate_policy"]
    assert all(check["severity"] in catalog["allowed_severities"] for check in checks)


def test_every_phase_has_human_review_pass_criteria_and_non_claims() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for phase in catalog["phases"].values():
        assert phase["human_review_questions"]
        assert phase["pass_criteria"]
        assert phase["non_claims"]
        assert all(isinstance(item, str) and item.strip() for item in phase["human_review_questions"])


def test_phase_aliases_are_stable() -> None:
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
    assert "T5" in completed.stdout


def test_planned_phase_is_reported_blocked_without_claiming_failure() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    report = runner._build_report(catalog, "T5", run=False)
    assert report["automated_verdict"] == "BLOCKED"
    assert report["phase_verdict"] == "BLOCKED"
    assert report["final_human_gate"]["status"] == "PENDING"
    assert all(item["status"] != "PASS" for item in report["results"] if item["mode"] == "planned")


def test_active_phase_dry_run_keeps_human_gate_pending() -> None:
    runner = _load_runner()
    catalog = runner._load_catalog(CATALOG)
    report = runner._build_report(catalog, "T1B", run=False)
    assert report["automated_verdict"] == "NOT_RUN"
    assert report["phase_verdict"] == "NOT_RUN"
    assert any(item["status"] == "PENDING_HUMAN" for item in report["results"])
