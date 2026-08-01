"""Dual-bank patient-specific Gaussian memory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json

import torch

from ..gaussians import GaussianBatch


class PrimitiveKind(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    VOLUMETRIC = "VOLUMETRIC"


@dataclass(frozen=True)
class PrimitiveObservability:
    evidence_count: torch.Tensor  # [N,1]
    coverage: torch.Tensor  # [N,1]
    disagreement: torch.Tensor  # [N,1]
    uncertainty: torch.Tensor  # [N,1]
    propagation_depth: torch.Tensor  # [N,1], int64
    update_round: torch.Tensor  # [N,1], int64

    def __post_init__(self) -> None:
        count = self.evidence_count.shape[0]
        for name in ("evidence_count", "coverage", "disagreement", "uncertainty"):
            value = getattr(self, name)
            if value.shape != (count, 1) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"{name} must be finite, non-negative, and [N,1]")
        for name in ("propagation_depth", "update_round"):
            value = getattr(self, name)
            if value.shape != (count, 1) or value.dtype is not torch.int64 or bool((value < 0).any()):
                raise ValueError(f"{name} must be non-negative int64 [N,1]")


@dataclass(frozen=True)
class GaussianMemoryBank:
    kind: PrimitiveKind
    gaussians: GaussianBatch
    anchor_ids: tuple[str, ...]
    parent_primitive_ids: tuple[str | None, ...]
    provenance_hashes: tuple[str, ...]
    observability: PrimitiveObservability

    def __post_init__(self) -> None:
        count = self.gaussians.count
        if any(len(values) != count for values in (self.anchor_ids, self.parent_primitive_ids, self.provenance_hashes)):
            raise ValueError("memory metadata must match Gaussian count")
        if self.gaussians.primitive_kind != (self.kind.value,) * count:
            raise ValueError("Gaussian primitive kinds must match their memory bank")
        if self.observability.evidence_count.shape[0] != count:
            raise ValueError("observability must match Gaussian count")


@dataclass(frozen=True)
class GaussianMemory:
    structural: GaussianMemoryBank
    volumetric: GaussianMemoryBank
    modality_ids: tuple[str, ...]
    memory_hash: str

    def __post_init__(self) -> None:
        if self.structural.kind is not PrimitiveKind.STRUCTURAL or self.volumetric.kind is not PrimitiveKind.VOLUMETRIC:
            raise ValueError("GaussianMemory requires distinct structural and volumetric banks")
        if self.structural.gaussians.appearance_channels != len(self.modality_ids) or self.volumetric.gaussians.appearance_channels != len(self.modality_ids):
            raise ValueError("memory appearance channels must match modality IDs")
        if self.memory_hash != gaussian_memory_hash(self.structural, self.volumetric, self.modality_ids):
            raise ValueError("memory_hash does not bind the exact dual-bank memory")

    @property
    def primitive_count(self) -> int:
        return self.structural.gaussians.count + self.volumetric.gaussians.count


def _hash_tensor(digest: "hashlib._Hash", value: torch.Tensor) -> None:
    item = value.detach().cpu().contiguous()
    digest.update(f"{item.dtype}:{tuple(item.shape)}".encode())
    digest.update(item.numpy().tobytes())


def gaussian_memory_hash(structural: GaussianMemoryBank, volumetric: GaussianMemoryBank, modality_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256(json.dumps({
        "modality_ids": modality_ids,
        "structural_ids": structural.gaussians.primitive_id,
        "structural_parents": structural.parent_primitive_ids,
        "volumetric_ids": volumetric.gaussians.primitive_id,
        "volumetric_parents": volumetric.parent_primitive_ids,
    }, sort_keys=True, separators=(",", ":")).encode())
    for bank in (structural, volumetric):
        for tensor in (
            bank.gaussians.centers_ras_mm, bank.gaussians.covariance_factor,
            bank.gaussians.log_support_amplitude, bank.gaussians.appearance,
            bank.gaussians.appearance_valid.to(torch.uint8), bank.observability.uncertainty,
            bank.observability.propagation_depth,
        ):
            _hash_tensor(digest, tensor)
    return digest.hexdigest()
