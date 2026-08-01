"""Immutable run and checkpoint provenance helpers for T1-C."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    modality_mapping_hash: str
    preprocessing_policy_hash: str
    preprocessing_record_hash: str
    opened_file_ledger_hash: str
    dependency_manifest_hash: str
    encoder_variant: str
    encoder_config_hash: str
    encoder_state_hash: str
    gaussian_head_initialization_hash: str
    renderer_config_hash: str
    amplitude_gauge_hash: str
    frozen_patient_state_schema_version: str
    device: str
    parameter_count: int
    run_started_at: str
    run_ended_at: str
    artifact_hashes: tuple[tuple[str, str], ...]

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
            "modality_mapping_hash",
            "preprocessing_policy_hash",
            "preprocessing_record_hash",
            "opened_file_ledger_hash",
            "dependency_manifest_hash",
            "encoder_config_hash",
            "encoder_state_hash",
            "gaussian_head_initialization_hash",
            "renderer_config_hash",
            "amplitude_gauge_hash",
        ):
            value = getattr(self, name)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
                raise ValueError(f"{name} must be a non-placeholder SHA-256 digest")
        if not self.environment or any(not key or not value for key, value in self.environment):
            raise ValueError("environment provenance must be explicit and non-empty")
        if (
            self.encoder_variant == "unknown"
            or self.device == "unknown"
            or not self.frozen_patient_state_schema_version
            or self.parameter_count <= 0
            or self.run_started_at == "unknown"
            or self.run_ended_at == "unknown"
        ):
            raise ValueError("provenance requires encoder, device, and parameter metadata")
        if not self.artifact_hashes:
            raise ValueError("provenance requires at least one immutable artifact digest")
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
    artifact_hashes: Mapping[str, str],
    modality_mapping_hash: str,
    preprocessing_policy_hash: str,
    preprocessing_record_hash: str,
    opened_file_ledger_hash: str,
    dependency_manifest_hash: str,
    encoder_variant: str,
    encoder_config_hash: str,
    encoder_state_hash: str,
    gaussian_head_initialization_hash: str,
    renderer_config_hash: str,
    amplitude_gauge_hash: str,
    frozen_patient_state_schema_version: str,
    device: str,
    parameter_count: int,
    run_started_at: str | None = None,
    run_ended_at: str | None = None,
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
    try:
        processor = platform.processor() or "unknown"
    except Exception:
        processor = "unknown"
    environment = {
        "cuda_available": str(torch.cuda.is_available()),
        "machine": platform.machine() or "unknown",
        "cpu_count": str(__import__("os").cpu_count() or 0),
        "platform": platform.platform(),
        "processor": processor,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_interop_threads": str(torch.get_num_interop_threads()),
        "torch_num_threads": str(torch.get_num_threads()),
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
        modality_mapping_hash=modality_mapping_hash,
        preprocessing_policy_hash=preprocessing_policy_hash,
        preprocessing_record_hash=preprocessing_record_hash,
        opened_file_ledger_hash=opened_file_ledger_hash,
        dependency_manifest_hash=dependency_manifest_hash,
        encoder_variant=encoder_variant,
        encoder_config_hash=encoder_config_hash,
        encoder_state_hash=encoder_state_hash,
        gaussian_head_initialization_hash=gaussian_head_initialization_hash,
        renderer_config_hash=renderer_config_hash,
        amplitude_gauge_hash=amplitude_gauge_hash,
        frozen_patient_state_schema_version=frozen_patient_state_schema_version,
        device=device,
        parameter_count=parameter_count,
        run_started_at=run_started_at or datetime.now(timezone.utc).isoformat(),
        run_ended_at=run_ended_at or datetime.now(timezone.utc).isoformat(),
        artifact_hashes=tuple(sorted(artifact_hashes.items())),
    )
