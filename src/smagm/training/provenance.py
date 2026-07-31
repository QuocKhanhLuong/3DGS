"""Immutable run and checkpoint provenance helpers for T1-C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
from typing import Mapping

import torch


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def module_state_hash(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{module_index}:{name}:{value.dtype}:{tuple(value.shape)}".encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RunProvenance:
    commit: str
    dirty: bool
    config_hash: str
    manifest_hash: str
    split_registry_hash: str
    assignment_schedule_hash: str
    seed: int
    environment: tuple[tuple[str, str], ...]
    environment_hash: str
    checkpoint_hash: str
    artifact_hashes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.commit or not isinstance(self.dirty, bool) or not isinstance(self.seed, int):
            raise ValueError("provenance requires commit, bool dirty state, and integer seed")
        for name in (
            "config_hash",
            "manifest_hash",
            "split_registry_hash",
            "assignment_schedule_hash",
            "environment_hash",
            "checkpoint_hash",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.environment or any(not key or not value for key, value in self.environment):
            raise ValueError("environment provenance must be explicit and non-empty")
        for name, digest in self.artifact_hashes:
            if not name or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("artifact hashes must contain names and SHA-256 digests")

    @property
    def record_hash(self) -> str:
        return canonical_hash(self.__dict__)


def capture_run_provenance(
    *,
    repository_root: str | Path,
    config_hash: str,
    manifest_hash: str,
    split_registry_hash: str,
    assignment_schedule_hash: str,
    seed: int,
    checkpoint_hash: str,
    artifact_hashes: Mapping[str, str] | None = None,
    allow_dirty: bool = False,
) -> RunProvenance:
    root = Path(repository_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty_entries = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty = bool(dirty_entries)
    if dirty and not allow_dirty:
        raise RuntimeError("gate-quality provenance requires a clean repository; allow_dirty is development-only")
    environment = {
        "cuda_available": str(torch.cuda.is_available()),
        "machine": platform.machine() or "unknown",
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    return RunProvenance(
        commit=commit,
        dirty=dirty,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        split_registry_hash=split_registry_hash,
        assignment_schedule_hash=assignment_schedule_hash,
        seed=seed,
        environment=tuple(sorted(environment.items())),
        environment_hash=canonical_hash(environment),
        checkpoint_hash=checkpoint_hash,
        artifact_hashes=tuple(sorted((artifact_hashes or {}).items())),
    )
