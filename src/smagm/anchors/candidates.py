"""Deterministic structural candidates from legal cached context features."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..features.contracts import EncoderFeatureMaps
from .contracts import StructuralCandidateBatch


@dataclass(frozen=True)
class CandidateSelectionConfig:
    maximum_candidates: int = 64
    minimum_score: float = 0.0
    structural_weight: float = 1.0
    reliability_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.maximum_candidates <= 0 or not all(math.isfinite(v) and v >= 0 for v in (
            self.minimum_score, self.structural_weight, self.reliability_weight
        )):
            raise ValueError("candidate budgets and score weights must be finite and non-negative")
        if self.structural_weight + self.reliability_weight <= 0:
            raise ValueError("at least one candidate score component must be active")


def select_structural_candidates(
    features: EncoderFeatureMaps, *, batch_index: int = 0,
    config: CandidateSelectionConfig | None = None,
) -> StructuralCandidateBatch:
    """Select legal feature centres with stable row-major tie breaking."""

    if not isinstance(features, EncoderFeatureMaps) or not 0 <= batch_index < features.batch_size:
        raise ValueError("features and an in-range batch_index are required")
    config = config or CandidateSelectionConfig()
    transform = features.grid_to_planes[batch_index]
    plane = transform.input_plane
    assert plane is not None
    if not plane.observation_id:
        raise ValueError("candidate source plane requires an observation_id")
    structural = features.structural[batch_index].square().mean(dim=0).sqrt()
    reliability = features.reliability[batch_index, 0]
    combined = config.structural_weight * structural + config.reliability_weight * reliability
    valid = features.valid_feature_mask[batch_index, 0] & (combined >= config.minimum_score)
    flat_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    if flat_indices.numel() == 0:
        raise ValueError("candidate selection produced no legal feature centres")
    flat_scores = combined.reshape(-1)[flat_indices]
    order = torch.argsort(flat_scores, descending=True, stable=True)[: config.maximum_candidates]
    selected_flat = flat_indices[order]
    width = combined.shape[1]
    indices = torch.stack((selected_flat // width, selected_flat % width), dim=1).to(torch.int64)
    score = combined.reshape(-1)[selected_flat, None]
    structural_score = structural.reshape(-1)[selected_flat, None]
    reliability_score = reliability.reshape(-1)[selected_flat, None]
    ids = tuple(f"{plane.observation_id}:v{int(v)}:u{int(u)}" for v, u in indices.tolist())
    return StructuralCandidateBatch(
        observation_id=plane.observation_id, modality_id=features.modality_ids[batch_index],
        feature_indices_vu=indices, score=score, structural_score=structural_score,
        reliability_score=reliability_score, transform=transform, candidate_ids=ids,
    )
