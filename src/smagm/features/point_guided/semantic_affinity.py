"""Locked semantic compatibility used by the sparse point PoU."""

from __future__ import annotations

import torch

from .contracts import validate_probability_simplex


def semantic_affinity(
    point_semantic: torch.Tensor,
    voxel_semantic: torch.Tensor,
) -> torch.Tensor:
    """Return exact L1 semantic compatibility for probability vectors.

    The final dimension is the semantic-class dimension and all preceding
    dimensions use normal PyTorch broadcasting.  For valid probability
    vectors the result lies in ``[0, 1]`` and is exactly
    ``1 - 0.5 * sum_k(abs(point[k] - voxel[k]))``.
    """
    if not isinstance(point_semantic, torch.Tensor) or not isinstance(voxel_semantic, torch.Tensor):
        raise TypeError("point_semantic and voxel_semantic must be torch.Tensor values")
    if not point_semantic.is_floating_point() or not voxel_semantic.is_floating_point():
        raise TypeError("point_semantic and voxel_semantic must be floating point")
    if point_semantic.ndim < 1 or voxel_semantic.ndim < 1:
        raise ValueError("semantic vectors must have a final class dimension")
    if point_semantic.shape[-1] <= 1 or voxel_semantic.shape[-1] != point_semantic.shape[-1]:
        raise ValueError("semantic vectors must share a class dimension with K > 1")
    if point_semantic.device != voxel_semantic.device:
        raise ValueError("semantic vectors must share one device")
    if not bool(torch.isfinite(point_semantic).all()) or not bool(torch.isfinite(voxel_semantic).all()):
        raise ValueError("semantic vectors must be finite")
    validate_probability_simplex("point_semantic", point_semantic, class_dimension=-1)
    validate_probability_simplex("voxel_semantic", voxel_semantic, class_dimension=-1)
    # The exact expression is in [0, 1] on a probability simplex.  Clamp only
    # floating-point roundoff at the closed endpoints so sparse construction
    # never turns a mathematically zero denominator into a negative edge.
    return torch.clamp(
        1.0 - 0.5 * (point_semantic - voxel_semantic).abs().sum(dim=-1),
        min=0.0,
        max=1.0,
    )


exact_l1_semantic_affinity = semantic_affinity


__all__ = ["exact_l1_semantic_affinity", "semantic_affinity"]
