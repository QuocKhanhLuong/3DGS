#!/usr/bin/env python3
"""Validate and execute the repository phase-gate checklist.

The runner collects software evidence only. It never writes a final Human Gate
verdict.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "quality" / "checklists.json"
PHASE_ALIASES = {"T0.5": "T05", "T0_5": "T05", "T1-A": "T1A", "T1-B": "T1B", "T1-C": "T1C"}


class CatalogError(ValueError):
    """Raised when the machine-readable gate catalog is malformed."""


def _load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"checklist catalog not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"invalid checklist JSON: {exc}") from exc
    _validate_catalog(data)
    return data


def _validate_catalog(data: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "allowed_verdicts",
        "allowed_bindings",
        "allowed_severities",
        "allowed_modes",
        "global_checks",
        "phases",
        "human_gate_policy",
    }
    missing = required - data.keys()
    if missing:
        raise CatalogError(f"catalog missing keys: {sorted(missing)}")
    if data["schema_version"] != 1:
        raise CatalogError("unsupported schema_version")
    if not isinstance(data["phases"], dict) or not data["phases"]:
        raise CatalogError("phases must be a non-empty object")

    bindings = set(data["allowed_bindings"])
    severities = set(data["allowed_severities"])
    modes = set(data["allowed_modes"])
    seen_ids: set[str] = set()

    def validate_check(check: Any, owner: str) -> None:
        if not isinstance(check, dict):
            raise CatalogError(f"{owner} contains a non-object check")
        for key in ("id", "category", "binding", "severity", "mode", "description"):
            if not isinstance(check.get(key), str) or not check[key].strip():
                raise CatalogError(f"{owner} check missing non-empty {key}")
        check_id = check["id"]
        if check_id in seen_ids:
            raise CatalogError(f"duplicate check id: {check_id}")
        seen_ids.add(check_id)
        if check["binding"] not in bindings:
            raise CatalogError(f"{check_id} has unsupported binding")
        if check["severity"] not in severities:
            raise CatalogError(f"{check_id} has unsupported severity")
        if check["mode"] not in modes:
            raise CatalogError(f"{check_id} has unsupported mode")
        if check["mode"] == "command":
            command = check.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
                raise CatalogError(f"{check_id} command must be a non-empty string list")
        elif check["mode"] == "pytest":
            if not isinstance(check.get("target"), str) or not check["target"]:
                raise CatalogError(f"{check_id} pytest check requires target")
        elif check["mode"] == "human":
            if not isinstance(check.get("reviewer_role"), str) or not check["reviewer_role"]:
                raise CatalogError(f"{check_id} human check requires reviewer_role")
        elif check["mode"] == "file":
            if not isinstance(check.get("path"), str) or not check["path"]:
                raise CatalogError(f"{check_id} file check requires path")

    if not isinstance(data["global_checks"], list):
        raise CatalogError("global_checks must be a list")
    for check in data["global_checks"]:
        validate_check(check, "global_checks")

    phase_names = set(data["phases"])
    for phase_name, phase in data["phases"].items():
        if not isinstance(phase, dict):
            raise CatalogError(f"phase {phase_name} must be an object")
        for key in (
            "title",
            "implementation_status",
            "prerequisites",
            "authoritative_documents",
            "checks",
            "human_review_questions",
            "pass_criteria",
            "non_claims",
        ):
            if key not in phase:
                raise CatalogError(f"phase {phase_name} missing {key}")
        if phase["implementation_status"] not in {"retrospective", "active", "planned"}:
            raise CatalogError(f"phase {phase_name} has invalid implementation_status")
        if not isinstance(phase["prerequisites"], list):
            raise CatalogError(f"phase {phase_name} prerequisites must be a list")
        unknown = set(phase["prerequisites"]) - phase_names
        if unknown:
            raise CatalogError(f"phase {phase_name} has unknown prerequisites: {sorted(unknown)}")
        for list_key in ("authoritative_documents", "checks", "human_review_questions", "pass_criteria", "non_claims"):
            if not isinstance(phase[list_key], list):
                raise CatalogError(f"phase {phase_name} {list_key} must be a list")
        for check in phase["checks"]:
            validate_check(check, phase_name)


def _normalize_phase(value: str) -> str:
    normalized = value.strip().upper()
    return PHASE_ALIASES.get(normalized, normalized.replace("-", ""))


def _git_metadata() -> dict[str, Any]:
    def output(command: list[str]) -> str:
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=30)
        return completed.stdout.strip()

    commit = output(["git", "rev-parse", "HEAD"]) or "unknown"
    dirty_lines = output(["git", "status", "--porcelain"]).splitlines()
    return {"commit": commit, "dirty": bool(dirty_lines), "dirty_entries": dirty_lines[:50]}


def _run_process(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        status = "PASS" if completed.returncode == 0 else "FAIL"
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        status = "FAIL"
        return_code = None
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout_seconds}s"
    ended = dt.datetime.now(dt.timezone.utc)
    return {
        "status": status,
        "command": command,
        "return_code": return_code,
        "duration_seconds": round((ended - started).total_seconds(), 3),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _evaluate_check(check: dict[str, Any], run: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": check["id"],
        "category": check["category"],
        "severity": check["severity"],
        "mode": check["mode"],
        "description": check["description"],
    }
    mode = check["mode"]
    if mode == "human":
        result.update(status="PENDING_HUMAN", reviewer_role=check["reviewer_role"])
    elif mode == "planned":
        result.update(status="PLANNED", planned_owner=check.get("planned_owner"))
    elif mode == "file":
        exists = (ROOT / check["path"]).exists()
        result.update(status="PASS" if exists else "FAIL", path=check["path"])
    elif not run:
        result.update(status="NOT_RUN")
    else:
        if mode == "pytest":
            command = ["python", "-m", "pytest", "-q", check["target"], "--tb=short"]
        else:
            command = list(check["command"])
        result.update(_run_process(command, int(check.get("timeout_seconds", 300))))
    return result


def _automated_verdict(phase_status: str, results: list[dict[str, Any]], run: bool) -> str:
    if phase_status == "planned":
        return "BLOCKED"
    automated = [item for item in results if item["mode"] not in {"human", "planned"}]
    if not run:
        return "NOT_RUN"
    if any(item["severity"] == "blocker" and item["status"] == "FAIL" for item in automated):
        return "FAIL"
    if any(item["status"] == "NOT_RUN" for item in automated):
        return "INCOMPLETE"
    return "PASS"


def _build_report(catalog: dict[str, Any], phase_name: str, run: bool) -> dict[str, Any]:
    phase = catalog["phases"][phase_name]
    checks = list(catalog["global_checks"]) + list(phase["checks"])
    results = [_evaluate_check(check, run=run) for check in checks]
    automated = _automated_verdict(phase["implementation_status"], results, run)
    human_pending = any(item["status"] == "PENDING_HUMAN" for item in results)
    phase_verdict = "PENDING_HUMAN_GATE" if automated == "PASS" and human_pending else automated
    return {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": phase_name,
        "title": phase["title"],
        "implementation_status": phase["implementation_status"],
        "repository": _git_metadata(),
        "automated_verdict": automated,
        "phase_verdict": phase_verdict,
        "results": results,
        "human_review_questions": phase["human_review_questions"],
        "pass_criteria": phase["pass_criteria"],
        "non_claims": phase["non_claims"],
        "final_human_gate": {
            "status": "PENDING",
            "decision": None,
            "decided_by": None,
            "conditions": [],
            "notes": None,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Phase Gate Report — {report['phase']}",
        "",
        f"- Title: {report['title']}",
        f"- Commit: `{report['repository']['commit']}`",
        f"- Dirty tree: `{report['repository']['dirty']}`",
        f"- Automated verdict: **{report['automated_verdict']}**",
        f"- Phase verdict: **{report['phase_verdict']}**",
        "",
        "## Checks",
        "",
        "| Status | ID | Category | Description |",
        "|---|---|---|---|",
    ]
    for item in report["results"]:
        description = item["description"].replace("|", "\\|")
        lines.append(f"| {item['status']} | `{item['id']}` | {item['category']} | {description} |")
    lines.extend(["", "## Human review questions", ""])
    lines.extend(f"- [ ] {question}" for question in report["human_review_questions"])
    lines.extend(["", "## Pass criteria", ""])
    lines.extend(f"- {item}" for item in report["pass_criteria"])
    lines.extend(["", "## Non-claims", ""])
    lines.extend(f"- {item}" for item in report["non_claims"])
    lines.extend(["", "## Final Human Gate", "", "Status: `PENDING`", ""])
    return "\n".join(lines)


def _write_report(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    commit = report["repository"]["commit"][:12]
    stem = f"{report['phase']}-{commit}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _print_catalog(catalog: dict[str, Any]) -> None:
    for name, phase in catalog["phases"].items():
        print(f"{name:4}  {phase['implementation_status']:13}  {phase['title']}")


def _print_report(report: dict[str, Any]) -> None:
    print(f"Phase: {report['phase']} — {report['title']}")
    print(f"Commit: {report['repository']['commit']}")
    print(f"Dirty tree: {report['repository']['dirty']}")
    print()
    for item in report["results"]:
        print(f"[{item['status']:<13}] {item['id']}: {item['description']}")
    print()
    print(f"AUTOMATED VERDICT: {report['automated_verdict']}")
    print(f"PHASE VERDICT: {report['phase_verdict']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", help="Phase name, for example T1B or T0.5")
    parser.add_argument("--list", action="store_true", help="List available phases")
    parser.add_argument("--run", action="store_true", help="Execute active command and pytest checks")
    parser.add_argument("--report-dir", type=Path, help="Write JSON and Markdown evidence reports")
    args = parser.parse_args(argv)

    try:
        catalog = _load_catalog()
    except CatalogError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        _print_catalog(catalog)
        return 0
    if not args.phase:
        parser.error("phase is required unless --list is used")
    phase_name = _normalize_phase(args.phase)
    if phase_name not in catalog["phases"]:
        print(f"unknown phase: {args.phase}", file=sys.stderr)
        return 2

    report = _build_report(catalog, phase_name, run=args.run)
    _print_report(report)
    if args.report_dir:
        json_path, md_path = _write_report(report, args.report_dir)
        print(f"Reports: {json_path.relative_to(ROOT)}, {md_path.relative_to(ROOT)}")

    if args.run and report["automated_verdict"] in {"FAIL", "INCOMPLETE"}:
        return 1
    if args.run and report["automated_verdict"] == "BLOCKED":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
