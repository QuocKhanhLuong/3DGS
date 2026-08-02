"""Typed patient-specific anchor contracts in canonical RAS millimetres."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import torch

from ..features.contracts import FeatureGridToPlaneTransform


def _tensor_hash(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{item.dtype}:{tuple(item.shape)}".encode("utf-8"))
    digest.update(item.to(torch.uint8).numpy().tobytes() if item.dtype is torch.bool else item.numpy().tobytes())
    return digest.hexdigest()


def _geometry_hash_payload(geometry: "AnchorGeometryBatch") -> dict[str, object]:
    """Bind all physical frame and provenance fields into the anchor digest."""

    return {
        "anchor_ids": geometry.anchor_ids,
        "centers_ras_mm": _tensor_hash(geometry.centers_ras_mm),
        "frame_axes_ras": _tensor_hash(geometry.frame_axes_ras),
        "frame_validity": _tensor_hash(geometry.frame_validity),
        "support_scales_mm": _tensor_hash(geometry.support_scales_mm),
        "geometry_confidence": _tensor_hash(geometry.geometry_confidence),
        "disagreement": _tensor_hash(geometry.disagreement),
        "contributing_observation_ids": geometry.contributing_observation_ids,
        "contributing_plane_hashes": geometry.contributing_plane_hashes,
        "provenance_hashes": geometry.provenance_hashes,
    }


def _anchor_hash_payload(
    *, patient_id: str, geometry: "AnchorGeometryBatch", evidence: torch.Tensor,
    appearance: torch.Tensor, appearance_valid: torch.Tensor, observability: torch.Tensor,
) -> dict[str, object]:
    return {
        "appearance": _tensor_hash(appearance),
        "appearance_valid": _tensor_hash(appearance_valid),
        "evidence": _tensor_hash(evidence),
        "geometry": _geometry_hash_payload(geometry),
        "observability": _tensor_hash(observability),
        "patient_id": patient_id,
    }


@dataclass(frozen=True)
class StructuralCandidateBatch:
    """Sparse candidates on one legal context feature grid.

    ``feature_indices_vu`` uses integer ``[v, u]`` indices. Scores remain live
    tensors, but deterministic selection itself is intentionally discrete.
    """

    observation_id: str
    modality_id: str
    feature_indices_vu: torch.Tensor  # [K, 2], int64
    score: torch.Tensor  # [K, 1]
    structural_score: torch.Tensor  # [K, 1]
    reliability_score: torch.Tensor  # [K, 1]
    transform: FeatureGridToPlaneTransform
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.feature_indices_vu.shape[0]
        if not self.observation_id or not self.modality_id or count <= 0:
            raise ValueError("candidate batch requires observation, modality, and candidates")
        if self.feature_indices_vu.shape != (count, 2) or self.feature_indices_vu.dtype is not torch.int64:
            raise ValueError("feature_indices_vu must have shape [K, 2] and dtype int64")
        for name in ("score", "structural_score", "reliability_score"):
            value = getattr(self, name)
            if value.shape != (count, 1) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite with shape [K, 1]")
        if len(self.candidate_ids) != count or len(set(self.candidate_ids)) != count:
            raise ValueError("candidate_ids must be unique and match candidate count")
        if self.transform.input_plane is None or self.transform.input_plane.observation_id != self.observation_id:
            raise ValueError("candidate transform must bind the source observation")


@dataclass(frozen=True)
class LiftedCandidateBatch:
    """Feature-grid candidates lifted into canonical RAS millimetres."""

    candidate_ids: tuple[str, ...]
    centers_ras_mm: torch.Tensor  # [K, 3]
    score: torch.Tensor  # [K, 1]
    modality_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    plane_hashes: tuple[str, ...]
    plane_axes_ras: torch.Tensor  # [K, 3, 3], columns (u, v, signed normal)

    def __post_init__(self) -> None:
        count = self.centers_ras_mm.shape[0]
        if count <= 0 or self.centers_ras_mm.shape != (count, 3) or self.score.shape != (count, 1):
            raise ValueError("lifted candidates require [K,3] centres and [K,1] scores")
        if any(len(values) != count for values in (self.candidate_ids, self.modality_ids, self.observation_ids, self.plane_hashes)):
            raise ValueError("lifted candidate metadata must match candidate count")
        if self.plane_axes_ras.shape != (count, 3, 3):
            raise ValueError("plane_axes_ras must have shape [K,3,3]")
        if not bool(torch.isfinite(self.centers_ras_mm).all() and torch.isfinite(self.score).all() and torch.isfinite(self.plane_axes_ras).all()):
            raise ValueError("lifted candidate tensors must be finite")

    def select(self, indices: torch.Tensor) -> "LiftedCandidateBatch":
        if indices.dtype is not torch.int64 or indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("indices must be a non-empty int64 vector")
        chosen = tuple(int(value) for value in indices.tolist())
        return LiftedCandidateBatch(
            candidate_ids=tuple(self.candidate_ids[i] for i in chosen),
            centers_ras_mm=self.centers_ras_mm[indices],
            score=self.score[indices],
            modality_ids=tuple(self.modality_ids[i] for i in chosen),
            observation_ids=tuple(self.observation_ids[i] for i in chosen),
            plane_hashes=tuple(self.plane_hashes[i] for i in chosen),
            plane_axes_ras=self.plane_axes_ras[indices],
        )


@dataclass(frozen=True)
class AnchorGeometryBatch:
    """Consolidated physical anchor geometry before evidence aggregation."""

    anchor_ids: tuple[str, ...]
    centers_ras_mm: torch.Tensor  # [N, 3]
    frame_axes_ras: torch.Tensor  # [N, 3, 3], columns local x/y/z
    frame_validity: torch.Tensor  # [N, 3], bool
    support_scales_mm: torch.Tensor  # [N, 3]
    geometry_confidence: torch.Tensor  # [N, 1]
    disagreement: torch.Tensor  # [N, 1]
    contributing_observation_ids: tuple[tuple[str, ...], ...]
    contributing_plane_hashes: tuple[tuple[str, ...], ...]
    provenance_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.centers_ras_mm.shape[0]
        if count <= 0 or len(self.anchor_ids) != count or len(set(self.anchor_ids)) != count:
            raise ValueError("anchor geometry requires unique non-empty anchor IDs")
        expected = {
            "centers_ras_mm": (count, 3), "frame_axes_ras": (count, 3, 3),
            "frame_validity": (count, 3), "support_scales_mm": (count, 3),
            "geometry_confidence": (count, 1), "disagreement": (count, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype is not torch.bool and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if self.frame_validity.dtype is not torch.bool or not bool((self.support_scales_mm > 0).all()):
            raise ValueError("frame validity must be bool and support scales positive")
        gram = self.frame_axes_ras.transpose(-1, -2) @ self.frame_axes_ras
        identity = torch.eye(3, dtype=gram.dtype, device=gram.device).expand_as(gram)
        if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5) or not bool((torch.linalg.det(self.frame_axes_ras) > 0).all()):
            raise ValueError("anchor frames must be right-handed and orthonormal")
        if any(len(values) != count for values in (
            self.contributing_observation_ids, self.contributing_plane_hashes, self.provenance_hashes
        )):
            raise ValueError("anchor provenance metadata must match anchor count")
        for digest in self.provenance_hashes:
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("anchor provenance hashes must be SHA-256 digests")


@dataclass(frozen=True)
class AnchorBatch:
    """Patient-bound anchors with compact evidence and modality appearance."""

    patient_id: str
    geometry: AnchorGeometryBatch
    evidence: torch.Tensor  # [N, C]
    appearance: torch.Tensor  # [N, M]
    appearance_valid: torch.Tensor  # [N, M], bool
    observability: torch.Tensor  # [N, O]
    modality_ids: tuple[str, ...]
    evidence_hash: str

    def __post_init__(self) -> None:
        count = len(self.geometry.anchor_ids)
        if not self.patient_id or self.evidence.ndim != 2 or self.evidence.shape[0] != count or self.evidence.shape[1] <= 0:
            raise ValueError("anchor evidence must have shape [N,C] with C > 0")
        if self.appearance.ndim != 2 or self.appearance.shape[0] != count or self.appearance.shape[1] <= 0:
            raise ValueError("anchor appearance must have shape [N,M] with M > 0")
        if self.appearance_valid.shape != self.appearance.shape or self.appearance_valid.dtype is not torch.bool:
            raise ValueError("appearance_valid must be bool and match appearance")
        if self.observability.ndim != 2 or self.observability.shape[0] != count:
            raise ValueError("observability must have shape [N,O]")
        for tensor in (self.evidence, self.appearance, self.observability):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("anchor evidence tensors must be finite")
        if len(self.modality_ids) != self.appearance.shape[1] or len(set(self.modality_ids)) != len(self.modality_ids):
            raise ValueError("modality_ids must identify each appearance channel")
        actual = hashlib.sha256(json.dumps(_anchor_hash_payload(
            patient_id=self.patient_id, geometry=self.geometry, evidence=self.evidence,
            appearance=self.appearance, appearance_valid=self.appearance_valid,
            observability=self.observability,
        ), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if self.evidence_hash != actual:
            raise ValueError("evidence_hash does not bind the exact anchor evidence")

    @property
    def count(self) -> int:
        return len(self.geometry.anchor_ids)

    @property
    def anchor_ids(self) -> tuple[str, ...]:
        return self.geometry.anchor_ids

    @property
    def centers_ras_mm(self) -> torch.Tensor:
        return self.geometry.centers_ras_mm

    @property
    def frame_axes_ras(self) -> torch.Tensor:
        return self.geometry.frame_axes_ras

    @property
    def support_scales_mm(self) -> torch.Tensor:
        return self.geometry.support_scales_mm


def anchor_evidence_hash(
    *, patient_id: str, geometry: AnchorGeometryBatch, evidence: torch.Tensor,
    appearance: torch.Tensor, appearance_valid: torch.Tensor, observability: torch.Tensor,
) -> str:
    payload = _anchor_hash_payload(
        patient_id=patient_id, geometry=geometry, evidence=evidence,
        appearance=appearance, appearance_valid=appearance_valid,
        observability=observability,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
