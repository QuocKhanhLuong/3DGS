#!/usr/bin/env python3
"""Validate and execute repository phase-gate evidence.

The runner distinguishes implementation state from Human Gate state. It can
collect evidence and write reports, but it never records a Human Gate PASS.
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
AGENT_DIR = ROOT / ".codex" / "agents"
PHASE_ALIASES = {
    "T0.5": "T05",
    "T0_5": "T05",
    "T1-A": "T1A",
    "T1-B": "T1B",
    "T1-C": "T1C",
}
IMPLEMENTATION_STATUSES = {"implemented", "active", "planned"}
HUMAN_GATE_STATUSES = {
    "pending",
    "blocked",
    "retrospective_unrecorded",
    "passed",
    "passed_with_conditions",
    "failed",
}
AUTOMATED_VERDICTS = {"PASS", "FAIL", "INCOMPLETE", "BLOCKED", "NOT_RUN"}
PHASE_VERDICTS = {
    "PASS",
    "PASS_WITH_CONDITIONS",
    "REWORK",
    "FAIL",
    "BLOCKED",
    "PENDING_HUMAN_GATE",
    "NOT_RUN",
}


class CatalogError(ValueError):
    """Raised when the machine-readable gate catalog is malformed."""


def _configured_role_ids() -> set[str]:
    return {path.stem for path in AGENT_DIR.glob("*.toml")}


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
        "allowed_implementation_statuses",
        "allowed_human_gate_statuses",
        "allowed_automated_verdicts",
        "allowed_phase_verdicts",
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
    if data["schema_version"] != 2:
        raise CatalogError("unsupported schema_version")
    if set(data["allowed_implementation_statuses"]) != IMPLEMENTATION_STATUSES:
        raise CatalogError("implementation status vocabulary is not exact")
    if set(data["allowed_human_gate_statuses"]) != HUMAN_GATE_STATUSES:
        raise CatalogError("Human Gate status vocabulary is not exact")
    if set(data["allowed_automated_verdicts"]) != AUTOMATED_VERDICTS:
        raise CatalogError("automated verdict vocabulary is not exact")
    if set(data["allowed_phase_verdicts"]) != PHASE_VERDICTS:
        raise CatalogError("phase verdict vocabulary is not exact")
    if not isinstance(data["phases"], dict) or not data["phases"]:
        raise CatalogError("phases must be a non-empty object")

    expected_phases = {"T0", "T05", "T1A", "T1B", "T1C", "T2", "T3", "T4", "T5"}
    if set(data["phases"]) != expected_phases:
        raise CatalogError(f"catalog phases must be exactly {sorted(expected_phases)}")

    bindings = set(data["allowed_bindings"])
    severities = set(data["allowed_severities"])
    modes = set(data["allowed_modes"])
    configured_roles = _configured_role_ids()
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
            if not isinstance(command, list) or not command or not all(
                isinstance(part, str) and part for part in command
            ):
                raise CatalogError(f"{check_id} command must be a non-empty string list")
            dirty_argument = check.get("development_allow_dirty_argument")
            if dirty_argument is not None and (not isinstance(dirty_argument, str) or not dirty_argument):
                raise CatalogError(f"{check_id} development_allow_dirty_argument must be a non-empty string")
        elif check["mode"] == "pytest":
            if not isinstance(check.get("target"), str) or not check["target"]:
                raise CatalogError(f"{check_id} pytest check requires target")
        elif check["mode"] == "human":
            role = check.get("reviewer_role")
            if not isinstance(role, str) or not role:
                raise CatalogError(f"{check_id} human check requires reviewer_role")
            if role not in configured_roles:
                raise CatalogError(
                    f"{check_id} reviewer_role {role!r} has no .codex/agents/{role}.toml"
                )
        elif check["mode"] == "planned":
            if not isinstance(check.get("planned_owner"), str) or not check["planned_owner"]:
                raise CatalogError(f"{check_id} planned check requires planned_owner")
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
            "human_gate_status",
            "prerequisites",
            "authoritative_documents",
            "checks",
            "human_review_questions",
            "pass_criteria",
            "non_claims",
        ):
            if key not in phase:
                raise CatalogError(f"phase {phase_name} missing {key}")
        if phase["implementation_status"] not in IMPLEMENTATION_STATUSES:
            raise CatalogError(f"phase {phase_name} has invalid implementation_status")
        if phase["human_gate_status"] not in HUMAN_GATE_STATUSES:
            raise CatalogError(f"phase {phase_name} has invalid human_gate_status")
        if phase["implementation_status"] == "planned" and phase["human_gate_status"] != "blocked":
            raise CatalogError(f"planned phase {phase_name} must have a blocked Human Gate")
        if phase["implementation_status"] == "implemented" and phase["human_gate_status"] == "blocked":
            raise CatalogError(f"implemented phase {phase_name} cannot have a blocked Human Gate")
        if phase["human_gate_status"] in {"passed", "passed_with_conditions", "failed"}:
            record = phase.get("human_gate_record")
            if not isinstance(record, str) or not record or not (ROOT / record).is_file():
                raise CatalogError(f"phase {phase_name} requires a committed human_gate_record")
        if not isinstance(phase["prerequisites"], list):
            raise CatalogError(f"phase {phase_name} prerequisites must be a list")
        unknown = set(phase["prerequisites"]) - phase_names
        if unknown:
            raise CatalogError(f"phase {phase_name} has unknown prerequisites: {sorted(unknown)}")
        for list_key in (
            "authoritative_documents",
            "checks",
            "human_review_questions",
            "pass_criteria",
            "non_claims",
        ):
            if not isinstance(phase[list_key], list):
                raise CatalogError(f"phase {phase_name} {list_key} must be a list")
        for check in phase["checks"]:
            validate_check(check, phase_name)


def _normalize_phase(value: str) -> str:
    normalized = value.strip().upper()
    return PHASE_ALIASES.get(normalized, normalized.replace("-", ""))


def _git_metadata() -> dict[str, Any]:
    def output(command: list[str]) -> str:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip()

    commit = output(["git", "rev-parse", "HEAD"]) or "unknown"
    dirty_entries = output(["git", "status", "--porcelain=v1", "--untracked-files=all"]).splitlines()
    return {
        "commit": commit,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries[:50],
    }


def _resolve_command(command: list[str]) -> list[str]:
    """Use the interpreter running this process for Python commands."""

    if command and Path(command[0]).name in {"python", "python3"}:
        return [sys.executable, *command[1:]]
    return command


def _run_process(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    resolved = _resolve_command(command)
    started = dt.datetime.now(dt.timezone.utc)
    try:
        completed = subprocess.run(
            resolved,
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
        "command": resolved,
        "return_code": return_code,
        "duration_seconds": round((ended - started).total_seconds(), 3),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _evaluate_check(check: dict[str, Any], run: bool, execution_blocked: bool = False) -> dict[str, Any]:
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
    elif execution_blocked:
        result.update(status="BLOCKED", reason="dirty tree; rerun with --allow-dirty for development evidence")
    elif not run:
        result.update(status="NOT_RUN")
    else:
        if mode == "pytest":
            command = [sys.executable, "-m", "pytest", "-q", check["target"], "--tb=short"]
        else:
            command = list(check["command"])
        result.update(_run_process(command, int(check.get("timeout_seconds", 300))))
    return result


def _automated_verdict(
    phase_status: str,
    results: list[dict[str, Any]],
    run: bool,
    execution_blocked: bool = False,
) -> str:
    if phase_status == "planned":
        return "BLOCKED"
    if execution_blocked:
        return "BLOCKED"
    automated = [item for item in results if item["mode"] not in {"human", "planned"}]
    if not run:
        return "NOT_RUN"
    if any(item["status"] == "FAIL" for item in automated):
        return "FAIL"
    if any(item["status"] in {"NOT_RUN", "BLOCKED"} for item in automated):
        return "INCOMPLETE"
    return "PASS"


def _phase_verdict(phase: dict[str, Any], automated: str) -> str:
    if phase["implementation_status"] == "planned" or phase["human_gate_status"] == "blocked":
        return "BLOCKED"
    if phase["human_gate_status"] in {"pending", "retrospective_unrecorded"}:
        if automated in {"PASS", "NOT_RUN"}:
            return "PENDING_HUMAN_GATE"
        return automated
    return automated


def _build_report(
    catalog: dict[str, Any],
    phase_name: str,
    run: bool,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    phase = catalog["phases"][phase_name]
    before = _git_metadata()
    execution_blocked = run and before["dirty"] and not allow_dirty
    checks = list(catalog["global_checks"]) + list(phase["checks"])
    if allow_dirty:
        resolved_checks = []
        for check in checks:
            dirty_argument = check.get("development_allow_dirty_argument")
            if check["mode"] == "command" and dirty_argument:
                check = dict(check)
                check["command"] = [*check["command"], dirty_argument]
            resolved_checks.append(check)
        checks = resolved_checks
    results = []
    execution_cache: dict[tuple[object, ...], dict[str, Any]] = {}
    identity_keys = {"id", "category", "severity", "mode", "description"}
    for check in checks:
        cache_key: tuple[object, ...] | None = None
        if check["mode"] == "pytest":
            cache_key = ("pytest", check["target"])
        elif check["mode"] == "command":
            cache_key = ("command", *check["command"])
        if cache_key is not None and cache_key in execution_cache:
            shared = execution_cache[cache_key]
            result = {
                "id": check["id"],
                "category": check["category"],
                "severity": check["severity"],
                "mode": check["mode"],
                "description": check["description"],
                **{key: value for key, value in shared.items() if key not in identity_keys},
            }
        else:
            result = _evaluate_check(check, run=run, execution_blocked=execution_blocked)
            if cache_key is not None:
                execution_cache[cache_key] = result
        results.append(result)
    if phase["human_gate_status"] in {"passed", "passed_with_conditions", "failed"}:
        for item in results:
            if item["status"] == "PENDING_HUMAN":
                item["status"] = "RECORDED_HUMAN"
                item["human_gate_record"] = phase["human_gate_record"]
    after = _git_metadata()

    if execution_blocked:
        results.append(
            {
                "id": "G-REPO-CLEAN-001",
                "category": "repository",
                "severity": "blocker",
                "mode": "command",
                "description": "--run requires a clean tree unless --allow-dirty is explicitly supplied.",
                "status": "BLOCKED",
                "reason": "dirty-before and --allow-dirty was not supplied",
            }
        )
    if run and not before["dirty"] and after["dirty"]:
        results.append(
            {
                "id": "G-REPO-STABLE-001",
                "category": "repository",
                "severity": "blocker",
                "mode": "command",
                "description": "Phase checks must not turn a clean checkout dirty.",
                "status": "FAIL",
                "reason": "dirty-after became true",
            }
        )

    automated = _automated_verdict(
        phase["implementation_status"],
        results,
        run,
        execution_blocked=execution_blocked,
    )
    phase_verdict = _phase_verdict(phase, automated)
    pending_roles = sorted(
        {
            item["reviewer_role"]
            for item in results
            if item["status"] == "PENDING_HUMAN" and "reviewer_role" in item
        }
    )
    return {
        "schema_version": 2,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": phase_name,
        "title": phase["title"],
        "implementation_status": phase["implementation_status"],
        "human_gate_status": phase["human_gate_status"],
        "human_gate_record": phase.get("human_gate_record"),
        "repository": {
            "commit": after["commit"],
            "before": before,
            "after": after,
            "dirty_before": before["dirty"],
            "dirty_after": after["dirty"],
            "dirty_execution_allowed": allow_dirty,
        },
        "automated_verdict": automated,
        "phase_verdict": phase_verdict,
        "pending_reviewer_roles": pending_roles,
        "results": results,
        "human_review_questions": phase["human_review_questions"],
        "pass_criteria": phase["pass_criteria"],
        "non_claims": phase["non_claims"],
        "final_human_gate": {
            "status": phase["human_gate_status"],
            "record": phase.get("human_gate_record"),
            "decision": None,
            "decided_by": None,
            "conditions": [],
            "notes": None,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    repository = report["repository"]
    lines = [
        f"# Phase Gate Report — {report['phase']}",
        "",
        f"- Title: {report['title']}",
        f"- Implementation status: `{report['implementation_status']}`",
        f"- Human Gate status: `{report['human_gate_status']}`",
        f"- Human Gate record: `{report['human_gate_record'] or 'none'}`",
        f"- Commit: `{repository['commit']}`",
        f"- Dirty before: `{repository['dirty_before']}`",
        f"- Dirty after: `{repository['dirty_after']}`",
        f"- Dirty execution allowed: `{repository['dirty_execution_allowed']}`",
        f"- Automated verdict: **{report['automated_verdict']}**",
        f"- Phase verdict: **{report['phase_verdict']}**",
        "",
        "## Pending reviewer roles",
        "",
    ]
    lines.extend(f"- `{role}`" for role in report["pending_reviewer_roles"])
    if not report["pending_reviewer_roles"]:
        lines.append("- None")
    lines.extend(["", "## Checks", "", "| Status | ID | Category | Description |", "|---|---|---|---|"])
    for item in report["results"]:
        description = item["description"].replace("|", "\\|")
        lines.append(f"| {item['status']} | `{item['id']}` | {item['category']} | {description} |")
    lines.extend(["", "## Human review questions", ""])
    lines.extend(f"- [ ] {question}" for question in report["human_review_questions"])
    lines.extend(["", "## Pass criteria", ""])
    lines.extend(f"- {item}" for item in report["pass_criteria"])
    lines.extend(["", "## Non-claims", ""])
    lines.extend(f"- {item}" for item in report["non_claims"])
    lines.extend(
        [
            "",
            "## Final Human Gate",
            "",
            f"Status: `{report['final_human_gate']['status']}`",
            f"Record: `{report['final_human_gate']['record'] or 'none'}`",
            "Decision: not recorded by the automated runner.",
            "",
        ]
    )
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
    print("Phase  Implementation  Human Gate                 Title")
    for name, phase in catalog["phases"].items():
        print(
            f"{name:5}  {phase['implementation_status']:13}  "
            f"{phase['human_gate_status']:25}  {phase['title']}"
        )


def _print_report(report: dict[str, Any]) -> None:
    repository = report["repository"]
    print(f"Phase: {report['phase']} — {report['title']}")
    print(f"Implementation status: {report['implementation_status']}")
    print(f"Human Gate status: {report['human_gate_status']}")
    print(f"Human Gate record: {report['human_gate_record'] or 'none'}")
    print(f"Commit: {repository['commit']}")
    print(f"Dirty before: {repository['dirty_before']}")
    print(f"Dirty after: {repository['dirty_after']}")
    print(f"Dirty execution allowed: {repository['dirty_execution_allowed']}")
    print()
    for item in report["results"]:
        print(f"[{item['status']:<13}] {item['id']}: {item['description']}")
    print()
    print(f"Pending reviewer roles: {', '.join(report['pending_reviewer_roles']) or 'none'}")
    print(f"AUTOMATED VERDICT: {report['automated_verdict']}")
    print(f"PHASE VERDICT: {report['phase_verdict']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", help="Phase name, for example T1B or T0.5")
    parser.add_argument("--list", action="store_true", help="List available phases")
    parser.add_argument("--run", action="store_true", help="Execute active command and pytest checks")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow --run on a dirty tree for development-only evidence",
    )
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

    report = _build_report(catalog, phase_name, run=args.run, allow_dirty=args.allow_dirty)
    _print_report(report)
    if args.report_dir:
        json_path, md_path = _write_report(report, args.report_dir)
        try:
            json_label = json_path.relative_to(ROOT)
            md_label = md_path.relative_to(ROOT)
        except ValueError:
            json_label = json_path
            md_label = md_path
        print(f"Reports: {json_label}, {md_label}")

    if args.run and report["automated_verdict"] == "FAIL":
        return 1
    if args.run and report["automated_verdict"] in {"BLOCKED", "INCOMPLETE"}:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
