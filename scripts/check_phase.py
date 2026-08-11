#!/usr/bin/env python3
"""Validate the locked point-guided MRI frontend quality gate.

The runner records automated software evidence.  Its Human Gate is deliberately
pending: this program has no approval path and cannot make a human decision.
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
GATE_ID = "POINT_GUIDED_FRONTEND"


class CatalogError(ValueError):
    """Raised when the machine-readable quality-gate catalog is malformed."""


def _load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"checklist catalog not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"invalid checklist JSON: {exc}") from exc
    _validate_catalog(catalog)
    return catalog


def _validate_string_list(value: Any, label: str, *, nonempty: bool = True) -> None:
    if not isinstance(value, list) or (nonempty and not value):
        raise CatalogError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise CatalogError(f"{label} must contain non-empty strings")


def _validate_catalog(catalog: Any) -> None:
    if not isinstance(catalog, dict):
        raise CatalogError("catalog must be an object")
    if set(catalog) != {"schema_version", "gate"}:
        raise CatalogError("catalog must contain only schema_version and gate")
    if catalog["schema_version"] != 1:
        raise CatalogError("unsupported schema_version")

    gate = catalog["gate"]
    if not isinstance(gate, dict):
        raise CatalogError("gate must be an object")
    required_gate_keys = {"id", "title", "automated_checks", "human_gate", "non_claims"}
    if set(gate) != required_gate_keys:
        raise CatalogError(f"gate keys must be exactly {sorted(required_gate_keys)}")
    if gate["id"] != GATE_ID:
        raise CatalogError(f"gate id must be {GATE_ID}")
    if not isinstance(gate["title"], str) or not gate["title"].strip():
        raise CatalogError("gate title must be a non-empty string")
    _validate_string_list(gate["non_claims"], "gate non_claims")

    human_gate = gate["human_gate"]
    if not isinstance(human_gate, dict):
        raise CatalogError("human_gate must be an object")
    if set(human_gate) != {"status", "policy", "questions"}:
        raise CatalogError("human_gate keys must be exactly status, policy, and questions")
    if human_gate["status"] != "pending":
        raise CatalogError("the Human Gate status is immutable pending in this runner")
    if not isinstance(human_gate["policy"], str) or not human_gate["policy"].strip():
        raise CatalogError("human_gate policy must be a non-empty string")
    _validate_string_list(human_gate["questions"], "human_gate questions")

    checks = gate["automated_checks"]
    if not isinstance(checks, list) or not checks:
        raise CatalogError("automated_checks must be a non-empty list")
    seen_ids: set[str] = set()
    required_check_keys = {"id", "description", "command", "timeout_seconds"}
    for check in checks:
        if not isinstance(check, dict) or set(check) != required_check_keys:
            raise CatalogError(f"each automated check must have exactly {sorted(required_check_keys)}")
        check_id = check["id"]
        if not isinstance(check_id, str) or not check_id.startswith("PGF-"):
            raise CatalogError("automated check id must start with PGF-")
        if check_id in seen_ids:
            raise CatalogError(f"duplicate automated check id: {check_id}")
        seen_ids.add(check_id)
        if not isinstance(check["description"], str) or not check["description"].strip():
            raise CatalogError(f"{check_id} description must be a non-empty string")
        _validate_string_list(check["command"], f"{check_id} command")
        timeout = check["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise CatalogError(f"{check_id} timeout_seconds must be a positive integer")


def _normalize_gate(value: str) -> str:
    return value.strip().upper()


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
    """Run catalogued Python commands with the interpreter running this script."""

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


def _evaluate_check(check: dict[str, Any], *, run: bool, execution_blocked: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": check["id"],
        "description": check["description"],
        "command": _resolve_command(list(check["command"])),
    }
    if execution_blocked:
        result.update(
            status="BLOCKED",
            reason="dirty tree; rerun with --allow-dirty for development evidence",
        )
    elif not run:
        result.update(status="NOT_RUN")
    else:
        result.update(_run_process(_resolve_command(list(check["command"])), check["timeout_seconds"]))
    return result


def _automated_verdict(results: list[dict[str, Any]], *, run: bool, execution_blocked: bool) -> str:
    if execution_blocked:
        return "BLOCKED"
    if not run:
        return "NOT_RUN"
    if any(result["status"] == "FAIL" for result in results):
        return "FAIL"
    if any(result["status"] != "PASS" for result in results):
        return "BLOCKED"
    return "PASS"


def _gate_verdict(automated_verdict: str) -> str:
    if automated_verdict == "PASS":
        return "PENDING_HUMAN_GATE"
    return automated_verdict


def _build_report(catalog: dict[str, Any], *, run: bool, allow_dirty: bool = False) -> dict[str, Any]:
    gate = catalog["gate"]
    before = _git_metadata()
    execution_blocked = run and before["dirty"] and not allow_dirty
    results = [
        _evaluate_check(check, run=run, execution_blocked=execution_blocked)
        for check in gate["automated_checks"]
    ]
    if execution_blocked:
        results.append(
            {
                "id": "PGF-CLEAN-TREE-001",
                "description": "--run requires a clean tree unless --allow-dirty is explicitly supplied.",
                "status": "BLOCKED",
                "reason": "dirty-before and --allow-dirty was not supplied",
            }
        )
    after = _git_metadata()
    automated_verdict = _automated_verdict(
        results,
        run=run,
        execution_blocked=execution_blocked,
    )
    human_gate = gate["human_gate"]
    return {
        "schema_version": catalog["schema_version"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gate": {"id": gate["id"], "title": gate["title"]},
        "repository": {
            "commit": after["commit"],
            "before": before,
            "after": after,
            "dirty_before": before["dirty"],
            "dirty_after": after["dirty"],
            "dirty_execution_allowed": allow_dirty,
        },
        "automated_verdict": automated_verdict,
        "gate_verdict": _gate_verdict(automated_verdict),
        "results": results,
        "human_gate": {
            "status": human_gate["status"],
            "policy": human_gate["policy"],
            "questions": human_gate["questions"],
            "decision": None,
            "decided_by": None,
        },
        "non_claims": gate["non_claims"],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    repository = report["repository"]
    gate = report["gate"]
    lines = [
        f"# Quality Gate Report — {gate['id']}",
        "",
        f"- Title: {gate['title']}",
        f"- Commit: `{repository['commit']}`",
        f"- Dirty before: `{repository['dirty_before']}`",
        f"- Dirty after: `{repository['dirty_after']}`",
        f"- Dirty execution allowed: `{repository['dirty_execution_allowed']}`",
        f"- Automated verdict: **{report['automated_verdict']}**",
        f"- Gate verdict: **{report['gate_verdict']}**",
        "",
        "## Automated checks",
        "",
        "| Status | ID | Description |",
        "|---|---|---|",
    ]
    for result in report["results"]:
        description = result["description"].replace("|", "\\|")
        lines.append(f"| {result['status']} | `{result['id']}` | {description} |")
    human_gate = report["human_gate"]
    lines.extend(["", "## Human Gate", ""])
    lines.extend(
        [
            f"Status: `{human_gate['status']}`",
            human_gate["policy"],
            "Decision: not recorded by the automated runner.",
            "",
            "### Review questions",
            "",
        ]
    )
    lines.extend(f"- [ ] {question}" for question in human_gate["questions"])
    lines.extend(["", "## Non-claims", ""])
    lines.extend(f"- {item}" for item in report["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _write_report(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    commit = report["repository"]["commit"][:12]
    stem = f"{report['gate']['id']}-{commit}"
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _print_catalog(catalog: dict[str, Any]) -> None:
    gate = catalog["gate"]
    print("Gate                     Human Gate  Title")
    print(f"{gate['id']:23}  {gate['human_gate']['status']:10}  {gate['title']}")


def _print_report(report: dict[str, Any]) -> None:
    repository = report["repository"]
    gate = report["gate"]
    print(f"Gate: {gate['id']} — {gate['title']}")
    print(f"Human Gate status: {report['human_gate']['status']}")
    print(f"Commit: {repository['commit']}")
    print(f"Dirty before: {repository['dirty_before']}")
    print(f"Dirty after: {repository['dirty_after']}")
    print(f"Dirty execution allowed: {repository['dirty_execution_allowed']}")
    print()
    for result in report["results"]:
        print(f"[{result['status']:<9}] {result['id']}: {result['description']}")
    print()
    print(f"AUTOMATED VERDICT: {report['automated_verdict']}")
    print(f"GATE VERDICT: {report['gate_verdict']}")
    print("HUMAN GATE: pending; no decision is recorded by this runner.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", nargs="?", help=f"Gate identifier ({GATE_ID})")
    parser.add_argument("--list", action="store_true", help="List the available quality gate")
    parser.add_argument("--run", action="store_true", help="Execute automated checks")
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
        if args.gate:
            parser.error("--list does not accept a gate identifier")
        _print_catalog(catalog)
        return 0
    if not args.gate:
        parser.error("gate is required unless --list is used")
    if args.allow_dirty and not args.run:
        parser.error("--allow-dirty requires --run")
    if _normalize_gate(args.gate) != GATE_ID:
        print(f"unknown gate: {args.gate}", file=sys.stderr)
        return 2

    report = _build_report(catalog, run=args.run, allow_dirty=args.allow_dirty)
    _print_report(report)
    if args.report_dir:
        json_path, markdown_path = _write_report(report, args.report_dir)
        try:
            json_label = json_path.relative_to(ROOT)
            markdown_label = markdown_path.relative_to(ROOT)
        except ValueError:
            json_label = json_path
            markdown_label = markdown_path
        print(f"Reports: {json_label}, {markdown_label}")

    if args.run and report["automated_verdict"] == "FAIL":
        return 1
    if args.run and report["automated_verdict"] == "BLOCKED":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
