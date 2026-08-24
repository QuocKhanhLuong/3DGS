#!/usr/bin/env python3
"""Resolve the repository's task-scoped code navigation policy.

This small local helper never opens source files and refuses paths outside the
repository. It tells a contributor which paths are in scope before inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "CODEGRAPH.json"
DECLARED_PATH_FIELDS = ("entrypoints", "read_paths")


def _load() -> dict[str, Any]:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    if graph.get("schema_version") != 1 or not isinstance(graph.get("tasks"), dict):
        raise ValueError("CODEGRAPH.json must use schema_version 1 with a tasks object")
    return graph


def _matches(path: str, patterns: list[str]) -> bool:
    candidate = PurePosixPath(path)
    return any(candidate.match(pattern) or path == pattern for pattern in patterns)


def _relative(path: str) -> str:
    candidate = (ROOT / path).resolve()
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside this repository: {path}") from exc


def _declared_path_exists(pattern: str) -> bool:
    """Return whether a manifest path or glob resolves inside the repository."""

    candidate = PurePosixPath(pattern)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    try:
        matches = ROOT.glob(pattern) if any(char in pattern for char in "*?[") else [ROOT / pattern]
    except (NotImplementedError, ValueError):
        return False
    return any(match.exists() for match in matches)


def _missing_declared_paths(
    tasks: dict[str, Any], task_names: list[str] | None = None
) -> list[str]:
    """Collect missing entrypoint/read declarations without opening source files."""

    names = sorted(tasks) if task_names is None else task_names
    missing: list[str] = []
    for task_name in names:
        task = tasks[task_name]
        for field in DECLARED_PATH_FIELDS:
            paths = task.get(field)
            if not isinstance(paths, list):
                missing.append(f"{task_name}:{field}:<missing list>")
                continue
            for pattern in paths:
                if not isinstance(pattern, str) or not _declared_path_exists(pattern):
                    missing.append(f"{task_name}:{field}:{pattern}")
    return missing


def _render(task_name: str, task: dict[str, Any]) -> str:
    lines = [f"task: {task_name}", f"purpose: {task['purpose']}", "entrypoints:"]
    lines.extend(f"  - {item}" for item in task["entrypoints"])
    lines.append("read paths:")
    lines.extend(f"  - {item}" for item in task["read_paths"])
    lines.append("write paths:")
    lines.extend(f"  - {item}" for item in task["write_paths"])
    lines.append("blocked paths:")
    lines.extend(f"  - {item}" for item in task["blocked_paths"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list task names without selecting one")
    parser.add_argument("--task", help="task name from CODEGRAPH.json")
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="verify that every declared entrypoint and read path exists",
    )
    parser.add_argument("--check", nargs="*", metavar="PATH", help="verify proposed paths are allowed to read")
    args = parser.parse_args(argv)
    graph = _load()
    tasks = graph["tasks"]
    if args.list and args.validate_paths:
        parser.error("--list and --validate-paths cannot be used together")
    if args.task and args.task not in tasks:
        parser.error(f"unknown task {args.task!r}; choose one of: {', '.join(sorted(tasks))}")
    if args.validate_paths:
        selected_tasks = [args.task] if args.task else None
        missing = _missing_declared_paths(tasks, selected_tasks)
        if missing:
            print("missing declared paths:", *missing, sep="\n  - ", file=sys.stderr)
            return 2
        scope = args.task if args.task else "all tasks"
        print(f"validated declared entrypoints and read paths for {scope}")
        return 0
    if args.list:
        print("\n".join(sorted(tasks)))
        return 0
    if not args.task:
        parser.error("--task is required unless --list is used")
    task = tasks[args.task]
    if args.check is not None:
        rejected: list[str] = []
        for supplied in args.check:
            relative = _relative(supplied)
            if _matches(relative, task["blocked_paths"]) or not _matches(relative, task["read_paths"]):
                rejected.append(relative)
        if rejected:
            print("denied read paths:", *rejected, sep="\n  - ", file=sys.stderr)
            return 2
    print(_render(args.task, task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
