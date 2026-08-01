"""Atomic PT/JSON/NIfTI export with affine and artifact verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np
import torch

from ..contracts.coordinates import TargetGrid
from ..contracts.outputs import ReconstructionPackage, VolumeReconstruction


EXPORT_SCHEMA = "smagm-reconstruction-package-v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary); temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _write_nifti_float32(volume: VolumeReconstruction, path: Path) -> None:
    """Write a minimal single-file NIfTI-1 image without a new dependency."""
    data = volume.intensity.detach().cpu().to(torch.float32).contiguous().numpy()
    d, h, w = data.shape
    affine = np.asarray(volume.grid.index_to_ras_mm, dtype=np.float32)
    spacing = [float(np.linalg.norm(affine[:3, index])) for index in range(3)]
    header = bytearray(348)
    struct.pack_into("<I", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, w, h, d, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)  # float32
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 1.0, spacing[0], spacing[1], spacing[2], 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 252, 0); struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, *affine[0]); struct.pack_into("<4f", header, 296, *affine[1]); struct.pack_into("<4f", header, 312, *affine[2])
    header[344:348] = b"n+1\0"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(header); stream.write(b"\0\0\0\0"); stream.write(data.tobytes(order="C"))
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _volume_payload(volume: VolumeReconstruction) -> dict[str, Any]:
    frozen = lambda value: value.detach().cpu().contiguous()
    return {
        "schema": "smagm-volume-reconstruction-v1", "patient_id": volume.patient_id,
        "modality_id": volume.modality_id, "grid": volume.grid.to_canonical_dict(),
        "intensity": frozen(volume.intensity), "support_mass": frozen(volume.support_mass),
        "unsupported_mask": frozen(volume.unsupported_mask), "support_uncertainty": frozen(volume.support_uncertainty),
        "depth_chunk_size": volume.depth_chunk_size, "renderer_config_hash": volume.renderer_config_hash,
        "patient_state_version": volume.patient_state_version, "artifact_hash": volume.artifact_hash,
    }


def export_reconstruction_package(
    package: ReconstructionPackage, volumes: tuple[VolumeReconstruction, ...], output_dir: str | Path,
    *, overwrite: bool = False, write_nifti: bool = True,
) -> Path:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError("reconstruction output directory is non-empty")
    destination.mkdir(parents=True, exist_ok=True)
    if tuple((f"volume:{v.modality_id}", v.artifact_hash) for v in volumes) != package.output_artifacts:
        raise ValueError("package artifact inventory does not match supplied volumes")
    file_hashes: list[tuple[str, str]] = []
    volume_records = []
    for volume in volumes:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", volume.modality_id) is None:
            raise ValueError("modality_id must be a safe portable filename component")
        pt_name = f"volume_{volume.modality_id}.pt"; pt_path = destination / pt_name
        _atomic_torch_save(_volume_payload(volume), pt_path); file_hashes.append((pt_name, _file_hash(pt_path)))
        record: dict[str, object] = {"modality_id": volume.modality_id, "pt": pt_name, "artifact_hash": volume.artifact_hash}
        if write_nifti:
            nii_name = f"volume_{volume.modality_id}.nii"; nii_path = destination / nii_name
            _write_nifti_float32(volume, nii_path); file_hashes.append((nii_name, _file_hash(nii_path))); record["nifti"] = nii_name
        volume_records.append(record)
    manifest = {
        "schema": EXPORT_SCHEMA, "package": package.__dict__, "volumes": volume_records,
        "file_hashes": tuple(file_hashes),
    }
    _atomic_json(manifest, destination / "package.json")
    return destination


def _restore_volume(payload: dict[str, Any]) -> VolumeReconstruction:
    if payload.get("schema") != "smagm-volume-reconstruction-v1":
        raise ValueError("unsupported volume schema")
    raw_grid = payload["grid"]
    grid = TargetGrid(raw_grid["index_to_ras_mm"], raw_grid["shape_dhw"], raw_grid["modality_ids"], raw_grid["normalization_records"])
    return VolumeReconstruction(
        payload["patient_id"], payload["modality_id"], grid, payload["intensity"], payload["support_mass"],
        payload["unsupported_mask"], payload["support_uncertainty"], int(payload["depth_chunk_size"]),
        payload["renderer_config_hash"], payload["patient_state_version"], payload["artifact_hash"],
    )


def load_reconstruction_package(path: str | Path) -> tuple[ReconstructionPackage, tuple[VolumeReconstruction, ...]]:
    root = Path(path); manifest = json.loads((root / "package.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported reconstruction package schema")
    for name, digest in manifest["file_hashes"]:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("artifact inventory permits sibling filenames only")
        if _file_hash(root / name) != digest:
            raise ValueError(f"artifact hash mismatch for {name}")
    raw = manifest["package"]
    package = ReconstructionPackage(
        patient_id=raw["patient_id"], repository_commit=raw["repository_commit"], config_hash=raw["config_hash"],
        manifest_hash=raw["manifest_hash"], split_hash=raw["split_hash"], assignment_hash=raw["assignment_hash"],
        patient_state_version=raw["patient_state_version"], encoder_identity=raw["encoder_identity"],
        field_identity=raw["field_identity"], gaussian_identity=raw["gaussian_identity"],
        propagation_identity=raw["propagation_identity"], modality_mapping=tuple(tuple(v) for v in raw["modality_mapping"]),
        output_artifacts=tuple(tuple(v) for v in raw["output_artifacts"]), execution_status=raw["execution_status"],
        runtime_seconds=float(raw["runtime_seconds"]), environment_hash=raw["environment_hash"],
        non_claims=tuple(raw["non_claims"]), package_hash=raw["package_hash"],
    )
    for item in manifest["volumes"]:
        if not isinstance(item.get("pt"), str) or Path(item["pt"]).name != item["pt"]:
            raise ValueError("volume inventory permits sibling filenames only")
    volumes = tuple(_restore_volume(torch.load(root / item["pt"], map_location="cpu", weights_only=True)) for item in manifest["volumes"])
    if tuple((f"volume:{volume.modality_id}", volume.artifact_hash) for volume in volumes) != package.output_artifacts:
        raise ValueError("restored volume artifacts do not match package inventory")
    return package, volumes
