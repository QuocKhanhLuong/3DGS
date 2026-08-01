"""Transparent cache-only evidence aggregation for physical anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..features.contracts import EncoderFeatureMaps
from .contracts import AnchorBatch, AnchorGeometryBatch, anchor_evidence_hash


class EmptyAnchorEvidenceError(RuntimeError):
    """Raised when an anchor has no legal registered context contributor."""


@dataclass(frozen=True)
class CachedPlaneEvidence:
    observation_id: str
    modality_id: str
    features: EncoderFeatureMaps
    cache_key_hash: str
    registration_id: str
    normalized_image: torch.Tensor | None = None  # [1,1,H,W]
    valid_image_mask: torch.Tensor | None = None  # same, bool
    registration_confidence: float = 1.0
    context_only: bool = True

    def __post_init__(self) -> None:
        if not self.context_only:
            raise PermissionError("cached plane evidence must be declared context-only")
        if not isinstance(self.registration_id, str) or not self.registration_id.strip():
            raise ValueError("cached evidence requires an explicit canonical-RAS registration identity")
        if self.features.batch_size != 1 or self.features.modality_ids != (self.modality_id,):
            raise ValueError("cached evidence must bind exactly one modality feature map")
        plane = self.features.grid_to_plane.input_plane
        if plane is None or plane.observation_id != self.observation_id:
            raise ValueError("cached evidence identity must match its bound physical plane")
        if not math.isfinite(self.registration_confidence) or not 0 < self.registration_confidence <= 1:
            raise ValueError("registration confidence must be in (0,1]")
        if self.normalized_image is not None:
            if self.normalized_image.ndim != 4 or self.normalized_image.shape[:2] != (1, 1):
                raise ValueError("normalized_image must have shape [1,1,H,W]")
            if self.valid_image_mask is None or self.valid_image_mask.shape != self.normalized_image.shape or self.valid_image_mask.dtype is not torch.bool:
                raise ValueError("valid_image_mask must be bool and match normalized_image")


@dataclass(frozen=True)
class AggregationConfig:
    maximum_plane_distance_mm: float = 4.0
    distance_sigma_mm: float = 2.0
    minimum_total_weight: float = 1e-6

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError("aggregation distances and weights must be positive finite values")


def _sample(map_tensor: torch.Tensor, grid: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    return F.grid_sample(map_tensor, grid.view(1, 1, -1, 2), mode=mode, padding_mode="zeros", align_corners=False)[0, :, 0].transpose(0, 1)


def aggregate_anchor_evidence(
    geometry: AnchorGeometryBatch, evidence: tuple[CachedPlaneEvidence, ...], *,
    patient_id: str, modality_ids: tuple[str, ...], config: AggregationConfig | None = None,
) -> AnchorBatch:
    config = config or AggregationConfig()
    if not evidence or not modality_ids or len(set(modality_ids)) != len(modality_ids):
        raise ValueError("aggregation requires cached evidence and unique modality IDs")
    if len({item.registration_id for item in evidence}) != 1:
        raise PermissionError("cross-observation aggregation requires one declared common registration identity")
    device, dtype = geometry.centers_ras_mm.device, geometry.centers_ras_mm.dtype
    count = geometry.centers_ras_mm.shape[0]
    tokens: list[list[torch.Tensor]] = [[] for _ in range(count)]
    token_weights: list[list[torch.Tensor]] = [[] for _ in range(count)]
    modality_values: list[list[list[torch.Tensor]]] = [[[] for _ in modality_ids] for _ in range(count)]
    modality_weights: list[list[list[torch.Tensor]]] = [[[] for _ in modality_ids] for _ in range(count)]
    for item in evidence:
        transform = item.features.grid_to_plane
        plane = transform.input_plane
        assert plane is not None
        centers = geometry.centers_ras_mm
        origin = torch.as_tensor(plane.pixel_center_origin_ras_mm, dtype=dtype, device=device)
        normal = torch.as_tensor(plane.signed_normal_ras, dtype=dtype, device=device)
        signed_distance = ((centers - origin) * normal).sum(dim=1)
        grid = transform.grid_sample_coordinates(centers)
        in_grid = (grid.abs() <= 1.0).all(dim=1) & (signed_distance.abs() <= config.maximum_plane_distance_mm)
        sampled_valid = _sample(item.features.valid_feature_mask.to(dtype=dtype), grid, mode="nearest")[:, 0] > 0.5
        legal = in_grid & sampled_valid
        structural = _sample(item.features.structural, grid)
        appearance_features = _sample(item.features.appearance, grid)
        reliability = _sample(item.features.reliability, grid)[:, 0].clamp(0, 1)
        distance_weight = torch.exp(-0.5 * (signed_distance / config.distance_sigma_mm).square())
        weights = reliability * distance_weight * item.registration_confidence
        for index in torch.nonzero(legal, as_tuple=False).flatten().tolist():
            token = torch.cat((structural[index], appearance_features[index], reliability[index:index+1], signed_distance[index:index+1].abs()))
            tokens[index].append(token); token_weights[index].append(weights[index])
        if item.normalized_image is not None:
            plane_grid = transform.grid_sample_coordinates(centers)
            image_value = _sample(item.normalized_image.to(device=device, dtype=dtype), plane_grid)[:, 0]
            image_valid = _sample(item.valid_image_mask.to(device=device, dtype=dtype), plane_grid, mode="nearest")[:, 0] > 0.5
            modality_index = modality_ids.index(item.modality_id) if item.modality_id in modality_ids else -1
            if modality_index >= 0:
                for index in torch.nonzero(legal & image_valid, as_tuple=False).flatten().tolist():
                    modality_values[index][modality_index].append(image_value[index])
                    modality_weights[index][modality_index].append(weights[index])
    aggregated, observable = [], []
    appearance = torch.zeros((count, len(modality_ids)), dtype=dtype, device=device)
    appearance_valid = torch.zeros_like(appearance, dtype=torch.bool)
    for index in range(count):
        if not tokens[index]:
            raise EmptyAnchorEvidenceError(f"anchor {geometry.anchor_ids[index]} has no legal cached contributor")
        values = torch.stack(tokens[index]); weights = torch.stack(token_weights[index])
        total = weights.sum()
        if float(total.detach()) < config.minimum_total_weight:
            raise EmptyAnchorEvidenceError("anchor evidence has insufficient legal weight")
        normalized = weights / total
        mean = (normalized[:, None] * values).sum(dim=0)
        variance = (normalized[:, None] * (values - mean).square()).sum(dim=0)
        epsilon = torch.finfo(variance.dtype).eps
        # ``sqrt(0)`` has an infinite derivative.  Preserve an exact zero
        # diagnostic while keeping the live single-contributor path finite.
        dispersion = torch.sqrt(variance + epsilon) - math.sqrt(epsilon)
        aggregated.append(torch.cat((mean, dispersion)))
        observable.append(torch.stack((torch.tensor(float(values.shape[0]), dtype=dtype, device=device), total, dispersion.mean())))
        for modality_index in range(len(modality_ids)):
            if modality_values[index][modality_index]:
                values_m = torch.stack(modality_values[index][modality_index])
                weights_m = torch.stack(modality_weights[index][modality_index])
                appearance[index, modality_index] = (values_m * weights_m).sum() / weights_m.sum().clamp_min(config.minimum_total_weight)
                appearance_valid[index, modality_index] = True
    evidence_tensor = torch.stack(aggregated)
    observability = torch.stack(observable)
    digest = anchor_evidence_hash(
        patient_id=patient_id, geometry=geometry, evidence=evidence_tensor, appearance=appearance,
        appearance_valid=appearance_valid, observability=observability,
    )
    return AnchorBatch(
        patient_id=patient_id, geometry=geometry, evidence=evidence_tensor, appearance=appearance,
        appearance_valid=appearance_valid, observability=observability, modality_ids=modality_ids,
        evidence_hash=digest,
    )
