"""Immutable reconstruction outputs with physical geometry and artifact hashes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import torch

from .coordinates import PhysicalPlane, TargetGrid


def _tensor_digest(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{item.dtype}:{tuple(item.shape)}".encode())
    digest.update(item.to(torch.uint8).numpy().tobytes() if item.dtype is torch.bool else item.numpy().tobytes())
    return digest.hexdigest()


def _artifact_hash(metadata: dict[str, object], tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for tensor in tensors:
        digest.update(_tensor_digest(tensor).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class PlaneReconstruction:
    patient_id: str
    modality_id: str
    plane: PhysicalPlane
    intensity: torch.Tensor  # [H,W], NaN where unsupported
    support_mass: torch.Tensor  # [H,W]
    unsupported_mask: torch.Tensor  # [H,W], bool
    support_uncertainty: torch.Tensor  # [H,W], NaN where unsupported
    renderer_version: str
    renderer_config_hash: str
    patient_state_version: str
    artifact_hash: str

    def __post_init__(self) -> None:
        shape = tuple(self.plane.shape_hw)
        if not self.patient_id or not self.modality_id or self.intensity.shape != shape:
            raise ValueError("plane reconstruction identity/shape is invalid")
        if self.support_mass.shape != shape or self.unsupported_mask.shape != shape or self.support_uncertainty.shape != shape:
            raise ValueError("plane reconstruction diagnostics must share plane shape")
        if self.unsupported_mask.dtype is not torch.bool or not bool(torch.isfinite(self.support_mass).all()) or bool((self.support_mass < 0).any()):
            raise ValueError("support mass must be finite/non-negative and mask bool")
        if bool(torch.isfinite(self.intensity[self.unsupported_mask]).any()) or bool(torch.isfinite(self.support_uncertainty[self.unsupported_mask]).any()):
            raise ValueError("unsupported intensity and uncertainty must remain explicit NaN")
        if not bool(torch.isfinite(self.intensity[~self.unsupported_mask]).all() and torch.isfinite(self.support_uncertainty[~self.unsupported_mask]).all()):
            raise ValueError("supported reconstruction values must be finite")
        expected = plane_artifact_hash(self)
        if self.artifact_hash != expected:
            raise ValueError("plane artifact hash mismatch")


def plane_artifact_hash(value: PlaneReconstruction) -> str:
    return plane_output_hash(
        patient_id=value.patient_id, modality_id=value.modality_id, plane=value.plane,
        renderer_version=value.renderer_version, renderer_config_hash=value.renderer_config_hash,
        patient_state_version=value.patient_state_version, intensity=value.intensity,
        support_mass=value.support_mass, unsupported_mask=value.unsupported_mask,
        support_uncertainty=value.support_uncertainty,
    )


def plane_output_hash(
    *, patient_id: str, modality_id: str, plane: PhysicalPlane, renderer_version: str,
    renderer_config_hash: str, patient_state_version: str, intensity: torch.Tensor,
    support_mass: torch.Tensor, unsupported_mask: torch.Tensor, support_uncertainty: torch.Tensor,
) -> str:
    metadata = {
        "patient_id": patient_id, "modality_id": modality_id, "plane": plane.canonical_json(),
        "renderer_version": renderer_version, "renderer_config_hash": renderer_config_hash,
        "patient_state_version": patient_state_version,
    }
    return _artifact_hash(metadata, (intensity, support_mass, unsupported_mask, support_uncertainty))


@dataclass(frozen=True)
class VolumeReconstruction:
    patient_id: str
    modality_id: str
    grid: TargetGrid
    intensity: torch.Tensor  # [D,H,W], NaN where unsupported
    support_mass: torch.Tensor
    unsupported_mask: torch.Tensor
    support_uncertainty: torch.Tensor
    depth_chunk_size: int
    renderer_config_hash: str
    patient_state_version: str
    artifact_hash: str

    def __post_init__(self) -> None:
        shape = tuple(self.grid.shape_dhw)
        if not self.patient_id or not self.modality_id or self.depth_chunk_size <= 0:
            raise ValueError("volume reconstruction identity/chunking is invalid")
        if any(value.shape != shape for value in (self.intensity, self.support_mass, self.unsupported_mask, self.support_uncertainty)):
            raise ValueError("volume tensors must match TargetGrid shape_dhw")
        if self.unsupported_mask.dtype is not torch.bool or not bool(torch.isfinite(self.support_mass).all()) or bool((self.support_mass < 0).any()):
            raise ValueError("volume support diagnostics are invalid")
        if bool(torch.isfinite(self.intensity[self.unsupported_mask]).any()) or bool(torch.isfinite(self.support_uncertainty[self.unsupported_mask]).any()):
            raise ValueError("unsupported voxels must stay explicit")
        if not bool(torch.isfinite(self.intensity[~self.unsupported_mask]).all() and torch.isfinite(self.support_uncertainty[~self.unsupported_mask]).all()):
            raise ValueError("supported volume values must be finite")
        if self.artifact_hash != volume_artifact_hash(self):
            raise ValueError("volume artifact hash mismatch")


def volume_artifact_hash(value: VolumeReconstruction) -> str:
    return volume_output_hash(
        patient_id=value.patient_id, modality_id=value.modality_id, grid=value.grid,
        depth_chunk_size=value.depth_chunk_size, renderer_config_hash=value.renderer_config_hash,
        patient_state_version=value.patient_state_version, intensity=value.intensity,
        support_mass=value.support_mass, unsupported_mask=value.unsupported_mask,
        support_uncertainty=value.support_uncertainty,
    )


def volume_output_hash(
    *, patient_id: str, modality_id: str, grid: TargetGrid, depth_chunk_size: int,
    renderer_config_hash: str, patient_state_version: str, intensity: torch.Tensor,
    support_mass: torch.Tensor, unsupported_mask: torch.Tensor, support_uncertainty: torch.Tensor,
) -> str:
    metadata = {
        "patient_id": patient_id, "modality_id": modality_id, "grid": grid.canonical_json(),
        "depth_chunk_size": depth_chunk_size, "renderer_config_hash": renderer_config_hash,
        "patient_state_version": patient_state_version,
    }
    return _artifact_hash(metadata, (intensity, support_mass, unsupported_mask, support_uncertainty))


@dataclass(frozen=True)
class ReconstructionPackage:
    patient_id: str
    repository_commit: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    assignment_hash: str
    patient_state_version: str
    encoder_identity: str
    field_identity: str
    gaussian_identity: str
    propagation_identity: str
    modality_mapping: tuple[tuple[str, int], ...]
    output_artifacts: tuple[tuple[str, str], ...]
    execution_status: str
    runtime_seconds: float
    environment_hash: str
    non_claims: tuple[str, ...]
    package_hash: str

    def __post_init__(self) -> None:
        if not self.patient_id or self.execution_status not in ("COMPLETE", "INSUFFICIENTLY_OBSERVED"):
            raise ValueError("reconstruction package identity/status is invalid")
        if re.fullmatch(r"[0-9a-f]{40}", self.repository_commit) is None or len(set(self.repository_commit)) == 1:
            raise ValueError("package requires exact repository commit")
        for name in ("config_hash", "manifest_hash", "split_hash", "assignment_hash", "patient_state_version", "environment_hash"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.output_artifacts or not self.non_claims or self.runtime_seconds < 0:
            raise ValueError("package requires artifacts, non-claims, and non-negative runtime")
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for _, digest in self.output_artifacts):
            raise ValueError("output artifact inventory requires SHA-256 digests")
        if self.package_hash != reconstruction_package_hash(self):
            raise ValueError("package_hash mismatch")


def reconstruction_package_hash(value: ReconstructionPackage) -> str:
    payload = value.__dict__.copy(); payload.pop("package_hash", None)
    return reconstruction_package_payload_hash(payload)


def reconstruction_package_payload_hash(payload: dict[str, object]) -> str:
    clean = dict(payload); clean.pop("package_hash", None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
