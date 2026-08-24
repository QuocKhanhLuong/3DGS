"""Best-effort provenance helpers for the server-ready point-guided pipeline."""

from __future__ import annotations

from pathlib import Path
import subprocess


def best_effort_git_head(repository_root: str | Path | None = None) -> str | None:
    """Return the repository HEAD, or ``None`` when Git is unavailable.

    Git provenance is useful metadata, but it must not make an otherwise
    valid training or evaluation run fail.  ``repository_root`` is optional so
    callers can resolve a specific worktree while tests can exercise both the
    repository and unavailable-Git cases without changing process state.
    """

    try:
        head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=None if repository_root is None else Path(repository_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head or None


__all__ = ["best_effort_git_head"]
