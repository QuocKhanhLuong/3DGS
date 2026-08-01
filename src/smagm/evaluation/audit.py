"""Isolated serialized-prediction audit barrier.

This module deliberately imports neither patient state, model, trainer, nor
reconstruction generation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..contracts.coordinates import TargetGrid
from ..contracts.outputs import ReconstructionPackage, VolumeReconstruction
from .metrics import ReconstructionMetrics, compute_reconstruction_metrics


@dataclass(frozen=True)
class FreezeRecord:
    package_hash: str
    config_hash: str
    split_hash: str
    architecture_frozen: bool
    analysis_plan_frozen: bool


@dataclass(frozen=True)
class SerializedPredictions:
    package: ReconstructionPackage
    volumes: tuple[VolumeReconstruction, ...]
    source_directory: Path


@dataclass(frozen=True)
class AuditTarget:
    patient_id: str
    split_hash: str
    modality_id: str
    grid: TargetGrid
    values: torch.Tensor
    valid_mask: torch.Tensor


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package(raw: dict[str, object]) -> ReconstructionPackage:
    return ReconstructionPackage(
        patient_id=raw["patient_id"], repository_commit=raw["repository_commit"], config_hash=raw["config_hash"],
        manifest_hash=raw["manifest_hash"], split_hash=raw["split_hash"], assignment_hash=raw["assignment_hash"],
        patient_state_version=raw["patient_state_version"], encoder_identity=raw["encoder_identity"], field_identity=raw["field_identity"],
        gaussian_identity=raw["gaussian_identity"], propagation_identity=raw["propagation_identity"],
        modality_mapping=tuple(tuple(v) for v in raw["modality_mapping"]), output_artifacts=tuple(tuple(v) for v in raw["output_artifacts"]),
        execution_status=raw["execution_status"], runtime_seconds=float(raw["runtime_seconds"]), environment_hash=raw["environment_hash"],
        non_claims=tuple(raw["non_claims"]), package_hash=raw["package_hash"],
    )


def _volume(payload: dict[str, object]) -> VolumeReconstruction:
    raw_grid = payload["grid"]
    grid = TargetGrid(raw_grid["index_to_ras_mm"], raw_grid["shape_dhw"], raw_grid["modality_ids"], raw_grid["normalization_records"])
    return VolumeReconstruction(
        payload["patient_id"], payload["modality_id"], grid, payload["intensity"], payload["support_mass"], payload["unsupported_mask"],
        payload["support_uncertainty"], int(payload["depth_chunk_size"]), payload["renderer_config_hash"], payload["patient_state_version"], payload["artifact_hash"],
    )


def open_serialized_predictions(directory: str | Path) -> SerializedPredictions:
    root = Path(directory); manifest = json.loads((root / "package.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "smagm-reconstruction-package-v1":
        raise ValueError("unsupported serialized prediction schema")
    for name, digest in manifest["file_hashes"]:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("serialized prediction inventory permits sibling filenames only")
        if _hash_file(root / name) != digest:
            raise ValueError(f"serialized prediction artifact is corrupt: {name}")
    package = _package(manifest["package"])
    for item in manifest["volumes"]:
        if not isinstance(item.get("pt"), str) or Path(item["pt"]).name != item["pt"]:
            raise ValueError("serialized volume inventory permits sibling filenames only")
    volumes = tuple(_volume(torch.load(root / item["pt"], map_location="cpu", weights_only=True)) for item in manifest["volumes"])
    expected_artifacts = dict(package.output_artifacts)
    actual_artifacts = {f"volume:{volume.modality_id}": volume.artifact_hash for volume in volumes}
    if actual_artifacts != expected_artifacts:
        raise ValueError("serialized volumes do not match package artifact identities")
    return SerializedPredictions(package, volumes, root)


def open_serialized_audit_targets(path: str | Path) -> tuple[AuditTarget, ...]:
    """Open an immutable tensor-only audit target file after prediction freeze."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "smagm-audit-targets-v1":
        raise ValueError("audit targets use an unsupported safe schema")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("audit target file must contain a non-empty target list")
    targets = []
    for raw in raw_targets:
        grid_raw = raw["grid"]
        grid = TargetGrid(
            grid_raw["index_to_ras_mm"], grid_raw["shape_dhw"],
            grid_raw["modality_ids"], grid_raw["normalization_records"],
        )
        targets.append(AuditTarget(
            str(raw["patient_id"]), str(raw["split_hash"]), str(raw["modality_id"]),
            grid, raw["values"], raw["valid_mask"],
        ))
    return tuple(targets)


def evaluate_audit_targets(predictions: SerializedPredictions, targets: tuple[AuditTarget, ...], *, freeze_record: FreezeRecord) -> tuple[ReconstructionMetrics, ...]:
    if not isinstance(predictions, SerializedPredictions):
        raise TypeError("audit evaluation accepts serialized predictions only")
    if not freeze_record.architecture_frozen or not freeze_record.analysis_plan_frozen:
        raise PermissionError("sealed audit requires architecture and analysis-plan freeze")
    package = predictions.package
    if (freeze_record.package_hash, freeze_record.config_hash, freeze_record.split_hash) != (package.package_hash, package.config_hash, package.split_hash):
        raise ValueError("freeze record does not match prediction package")
    by_modality = {volume.modality_id: volume for volume in predictions.volumes}
    results = []
    for target in targets:
        if target.patient_id != package.patient_id or target.split_hash != package.split_hash or target.modality_id not in by_modality:
            raise ValueError("audit target identity does not match prediction package")
        prediction = by_modality[target.modality_id]
        if target.grid.canonical_json() != prediction.grid.canonical_json():
            raise ValueError("audit target physical grid does not match prediction")
        results.append(compute_reconstruction_metrics(prediction, target.values, target.valid_mask))
    return tuple(results)
