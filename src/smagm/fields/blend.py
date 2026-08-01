"""Stable compact-support blending for anchor-local field values."""

from __future__ import annotations

import math

import torch

from .contracts import StructuralFieldOutput


def compact_support_weights(local_coordinates: torch.Tensor, neighbor_valid: torch.Tensor) -> torch.Tensor:
    if local_coordinates.shape[:-1] != neighbor_valid.shape or neighbor_valid.dtype is not torch.bool:
        raise ValueError("local coordinates and neighbor validity must align")
    radius = torch.linalg.vector_norm(local_coordinates, dim=-1)
    weights = (1.0 - radius.square()).clamp_min(0).square()
    return (weights * neighbor_valid.to(dtype=weights.dtype)).unsqueeze(-1)


def blend_local_fields(
    local_values: torch.Tensor, local_coordinates: torch.Tensor, neighbor_valid: torch.Tensor,
    *, minimum_total_weight: float = 1e-6,
) -> StructuralFieldOutput:
    if local_values.shape[:-1] != local_coordinates.shape[:-1] or local_values.shape[-1] != 1:
        raise ValueError("local values must match [Q,K] local coordinates")
    if minimum_total_weight <= 0:
        raise ValueError("minimum_total_weight must be positive")
    weights = compact_support_weights(local_coordinates, neighbor_valid)
    total = weights.sum(dim=1)
    supported = total[:, 0] >= minimum_total_weight
    denominator = total.clamp_min(minimum_total_weight)
    value = (weights * local_values).sum(dim=1) / denominator
    variance = (weights * (local_values - value[:, None]).square()).sum(dim=1) / denominator
    value = torch.where(supported[:, None], value, torch.full_like(value, float("nan")))
    epsilon = torch.finfo(variance.dtype).eps
    disagreement = torch.sqrt(variance + epsilon) - math.sqrt(epsilon)
    return StructuralFieldOutput(
        value=value, supported=supported, total_weight=total, disagreement=disagreement,
        local_values=local_values, support_weights=weights,
    )
