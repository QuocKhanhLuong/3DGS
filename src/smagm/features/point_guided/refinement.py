"""Hard-bounded, valid point refinement and point-centre semantics."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import PointGuidedConfig
from .contracts import PointField, PointGuidedGeometryError, VolumeGeometry
from .directional import DirectionalDescriptor
from .offset_predictor import OffsetPredictor
from .sampling import ras_mm_in_bounds, ras_mm_to_voxel_dhw, sample_volume_ras_mm


def _validate_point_tensor(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} must be a floating-point tensor with shape [B, N, 3]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def bound_displacement_ras_mm(raw_displacement_ras_mm: torch.Tensor, max_displacement_mm: float) -> torch.Tensor:
    """Radially project raw vectors to the closed physical displacement ball."""

    _validate_point_tensor("raw_displacement_ras_mm", raw_displacement_ras_mm)
    if not math.isfinite(float(max_displacement_mm)) or max_displacement_mm <= 0.0:
        raise ValueError("max_displacement_mm must be positive and finite")
    norm = torch.linalg.vector_norm(raw_displacement_ras_mm, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_displacement_mm) / norm, max=1.0)
    return raw_displacement_ras_mm * scale


def project_displacement_to_validity(
    original_centres_ras_mm: torch.Tensor,
    bounded_displacement_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
) -> torch.Tensor:
    """Line-project a bounded displacement until the affine volume remains valid.

    Starting from a valid original centre, the maximal scalar ``t`` in
    ``[0, 1]`` is selected such that ``original + t * displacement`` stays in
    the continuous voxel-centre box.  This avoids component-wise world-space
    clipping and cannot increase the original-centre-relative displacement.
    """

    _validate_point_tensor("original_centres_ras_mm", original_centres_ras_mm)
    _validate_point_tensor("bounded_displacement_ras_mm", bounded_displacement_ras_mm)
    if bounded_displacement_ras_mm.shape != original_centres_ras_mm.shape:
        raise ValueError("bounded_displacement_ras_mm must match original_centres_ras_mm")
    if (
        bounded_displacement_ras_mm.device != original_centres_ras_mm.device
        or bounded_displacement_ras_mm.dtype != original_centres_ras_mm.dtype
    ):
        raise ValueError("original centres and displacement must share device and dtype")
    if not bool(ras_mm_in_bounds(original_centres_ras_mm, geometry).all()):
        raise PointGuidedGeometryError("original point centres must be inside the valid volume")

    original_voxel_dhw = ras_mm_to_voxel_dhw(original_centres_ras_mm, geometry)
    candidate_voxel_dhw = ras_mm_to_voxel_dhw(
        original_centres_ras_mm + bounded_displacement_ras_mm,
        geometry,
    )
    delta_voxel_dhw = candidate_voxel_dhw - original_voxel_dhw
    upper = torch.as_tensor(
        tuple(length - 1 for length in geometry.shape_dhw),
        dtype=original_voxel_dhw.dtype,
        device=original_voxel_dhw.device,
    )
    infinity = torch.full_like(delta_voxel_dhw, float("inf"))
    upper_limit = (upper - original_voxel_dhw) / delta_voxel_dhw
    lower_limit = -original_voxel_dhw / delta_voxel_dhw
    limits = torch.where(
        delta_voxel_dhw > 0.0,
        upper_limit,
        torch.where(delta_voxel_dhw < 0.0, lower_limit, infinity),
    )
    fraction = torch.clamp(limits.amin(dim=-1, keepdim=True), min=0.0, max=1.0)
    return bounded_displacement_ras_mm * fraction


def refine_points_ras_mm(
    original_centres_ras_mm: torch.Tensor,
    raw_displacement_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
    max_displacement_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid refined centres and their exact original-relative offset."""

    _validate_point_tensor("original_centres_ras_mm", original_centres_ras_mm)
    _validate_point_tensor("raw_displacement_ras_mm", raw_displacement_ras_mm)
    if raw_displacement_ras_mm.shape != original_centres_ras_mm.shape:
        raise ValueError("raw_displacement_ras_mm must match original_centres_ras_mm")
    if (
        raw_displacement_ras_mm.device != original_centres_ras_mm.device
        or raw_displacement_ras_mm.dtype != original_centres_ras_mm.dtype
    ):
        raise ValueError("original centres and raw displacement must share device and dtype")
    bounded = bound_displacement_ras_mm(raw_displacement_ras_mm, max_displacement_mm)
    displacement = project_displacement_to_validity(original_centres_ras_mm, bounded, geometry)
    refined = original_centres_ras_mm + displacement
    return refined, refined - original_centres_ras_mm


class PointRefiner(nn.Module):
    """Directional descriptor, raw MLP offset, hard projection, and semantics."""

    def __init__(self, config: PointGuidedConfig) -> None:
        super().__init__()
        self.config = config
        self.directional_descriptor = DirectionalDescriptor(config)
        self.offset_predictor = OffsetPredictor.from_config(config)

    def forward(
        self,
        mri_volume: torch.Tensor,
        coarse_semantic: torch.Tensor,
        original_centres_ras_mm: torch.Tensor,
        geometry: VolumeGeometry,
    ) -> PointField:
        descriptor = self.directional_descriptor(
            mri_volume,
            coarse_semantic,
            original_centres_ras_mm,
            geometry,
        )
        raw_displacement = self.offset_predictor(descriptor)
        refined_centres, displacement = refine_points_ras_mm(
            original_centres_ras_mm,
            raw_displacement,
            geometry,
            self.config.max_displacement_mm,
        )
        semantic_vectors = sample_volume_ras_mm(
            coarse_semantic,
            refined_centres,
            geometry,
            require_in_bounds=True,
        )
        return PointField(
            original_centers_ras_mm=original_centres_ras_mm,
            refined_centers_ras_mm=refined_centres,
            displacement_ras_mm=displacement,
            semantic_vectors=semantic_vectors,
            support_radius_mm=self.config.support_radius_mm,
        )


__all__ = [
    "PointRefiner",
    "bound_displacement_ras_mm",
    "project_displacement_to_validity",
    "refine_points_ras_mm",
]
