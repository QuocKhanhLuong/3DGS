"""Locked quadratic compact-support spatial affinity."""

from __future__ import annotations

import math

import torch


def spatial_affinity(
    distance_mm: torch.Tensor,
    support_radius_mm: float,
) -> torch.Tensor:
    """Return ``(1 - d / r)^2`` for ``0 <= d < r`` and zero otherwise."""
    if not isinstance(distance_mm, torch.Tensor):
        raise TypeError("distance_mm must be a torch.Tensor")
    if not distance_mm.is_floating_point():
        raise TypeError("distance_mm must be floating point")
    if not bool(torch.isfinite(distance_mm).all()):
        raise ValueError("distance_mm must be finite")
    radius = float(support_radius_mm)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("support_radius_mm must be positive and finite")
    inside = (distance_mm >= 0.0) & (distance_mm < radius)
    kernel = (1.0 - distance_mm / radius).square()
    return torch.where(inside, kernel, torch.zeros_like(distance_mm))


quadratic_compact_spatial_affinity = spatial_affinity


__all__ = ["quadratic_compact_spatial_affinity", "spatial_affinity"]
