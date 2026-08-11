"""Sparse semantic-aware partition-of-unity construction.

This module deliberately builds only each point's affine-bounded compact
neighbourhood.  It never materializes a point-by-volume tensor.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Final

import torch
from torch import nn

from .config import PointGuidedConfig
from .contracts import (
    EmptySparseSupportError,
    PointField,
    PointGuidedGeometryError,
    SparsePoU,
    VolumeGeometry,
    validate_probability_simplex,
)
from .semantic_affinity import semantic_affinity
from .spatial_affinity import spatial_affinity


_DEFAULT_MAX_LOCAL_VOXELS_PER_POINT: Final[int] = 4096
_MAX_ENUMERATION_CHUNK_VOXELS: Final[int] = 4096


def _validate_semantic_volume(
    semantic_volume: torch.Tensor,
    point_field: PointField,
    geometry: VolumeGeometry,
) -> tuple[int, int, int, int, int]:
    if not isinstance(semantic_volume, torch.Tensor) or semantic_volume.ndim != 5:
        raise ValueError("semantic_volume must have shape [B, K, D, H, W]")
    if not semantic_volume.is_floating_point():
        raise TypeError("semantic_volume must be floating point")
    if not bool(torch.isfinite(semantic_volume).all()):
        raise ValueError("semantic_volume must be finite")
    batch, classes, depth, height, width = semantic_volume.shape
    if classes <= 1:
        raise ValueError("semantic_volume requires K > 1 semantic classes")
    validate_probability_simplex("semantic_volume", semantic_volume, class_dimension=1)
    if tuple(geometry.shape_dhw) != (depth, height, width):
        raise PointGuidedGeometryError("geometry and semantic_volume must agree on [D, H, W]")
    if point_field.refined_centers_ras_mm.shape[:2] != point_field.semantic_vectors.shape[:2]:
        raise ValueError("point-field centres and semantics must agree")
    if point_field.refined_centers_ras_mm.shape[0] != batch:
        raise ValueError("point_field and semantic_volume must share batch size")
    if point_field.semantic_vectors.shape[-1] != classes:
        raise ValueError("point_field and semantic_volume must share semantic classes")
    if point_field.refined_centers_ras_mm.device != semantic_volume.device:
        raise ValueError("point_field and semantic_volume must share one device")
    if point_field.semantic_vectors.dtype != semantic_volume.dtype:
        raise ValueError("point_field and semantic_volume must share one floating dtype")
    return batch, classes, depth, height, width


def _canonical_brain_mask(
    valid_brain_mask: torch.Tensor | None,
    *,
    batch: int,
    shape_dhw: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if valid_brain_mask is None:
        return None
    if not isinstance(valid_brain_mask, torch.Tensor):
        raise TypeError("valid_brain_mask must be a torch.Tensor or None")
    if valid_brain_mask.device != device:
        raise ValueError("valid_brain_mask must share the semantic-volume device")
    expected = (batch, *shape_dhw)
    if valid_brain_mask.shape == (batch, 1, *shape_dhw):
        valid_brain_mask = valid_brain_mask[:, 0]
    if valid_brain_mask.shape != expected:
        raise ValueError("valid_brain_mask must have shape [B, D, H, W] or [B, 1, D, H, W]")
    if valid_brain_mask.is_floating_point() and not bool(torch.isfinite(valid_brain_mask).all()):
        raise ValueError("valid_brain_mask must be finite")
    is_binary = (valid_brain_mask == 0) | (valid_brain_mask == 1)
    if not bool(is_binary.all()):
        raise ValueError("valid_brain_mask must be binary (zero outside and one inside)")
    return valid_brain_mask.to(dtype=torch.bool)


def _index_bounds_whd(
    center_ras_mm: torch.Tensor,
    inverse_spatial: torch.Tensor,
    origin_ras_mm: torch.Tensor,
    support_radius_mm: float,
    shape_dhw: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Conservatively bound an RAS sphere in affine ``[w, h, d]`` indices."""
    centre_whd = inverse_spatial @ (center_ras_mm - origin_ras_mm)
    half_extent_whd = support_radius_mm * torch.linalg.vector_norm(inverse_spatial, dim=1)
    lower = torch.floor(centre_whd - half_extent_whd).to(dtype=torch.long).detach().cpu().tolist()
    upper = torch.ceil(centre_whd + half_extent_whd).to(dtype=torch.long).detach().cpu().tolist()
    depth, height, width = shape_dhw
    w_bounds = (max(0, int(lower[0])), min(width - 1, int(upper[0])))
    h_bounds = (max(0, int(lower[1])), min(height - 1, int(upper[1])))
    d_bounds = (max(0, int(lower[2])), min(depth - 1, int(upper[2])))
    return d_bounds, h_bounds, w_bounds


def _iter_local_voxel_index_chunks(
    d_bounds: tuple[int, int],
    h_bounds: tuple[int, int],
    w_bounds: tuple[int, int],
    *,
    chunk_voxels: int,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Yield bounded pieces of an affine sphere's conservative index box.

    The exact spherical cap is enforced by the caller after distance/mask
    filtering.  Crucially, no full bounding cube is materialized first, so a
    fine-spacing geometry fails through its explicit local-support cap rather
    than allocating an unbounded temporary meshgrid.
    """

    if d_bounds[0] > d_bounds[1] or h_bounds[0] > h_bounds[1] or w_bounds[0] > w_bounds[1]:
        return
    if chunk_voxels <= 0:
        raise ValueError("chunk_voxels must be positive")
    d_count = d_bounds[1] - d_bounds[0] + 1
    h_count = h_bounds[1] - h_bounds[0] + 1
    w_count = w_bounds[1] - w_bounds[0] + 1
    plane_voxels = h_count * w_count
    total_voxels = d_count * plane_voxels
    for start in range(0, total_voxels, chunk_voxels):
        stop = min(start + chunk_voxels, total_voxels)
        linear = torch.arange(start, stop, dtype=torch.long, device=device)
        d = torch.div(linear, plane_voxels, rounding_mode="floor") + d_bounds[0]
        remainder = torch.remainder(linear, plane_voxels)
        h = torch.div(remainder, w_count, rounding_mode="floor") + h_bounds[0]
        w = torch.remainder(remainder, w_count) + w_bounds[0]
        yield torch.stack((d, h, w), dim=-1)


def _ras_mm_from_voxel_indices(
    voxel_indices_dhw: torch.Tensor,
    affine: torch.Tensor,
) -> torch.Tensor:
    """Map local tensor ``[d, h, w]`` indices to RAS-mm voxel centres."""
    index_whd = torch.stack(
        (voxel_indices_dhw[:, 2], voxel_indices_dhw[:, 1], voxel_indices_dhw[:, 0]),
        dim=-1,
    ).to(dtype=affine.dtype)
    return index_whd @ affine[:3, :3].transpose(0, 1) + affine[:3, 3]


def _normalize_per_voxel(
    batch_indices: torch.Tensor,
    voxel_indices_dhw: torch.Tensor,
    raw_affinity: torch.Tensor,
    volume_shape_dhw: tuple[int, int, int],
) -> torch.Tensor:
    """Normalize positive edges by their compact contributors at each voxel."""
    depth, height, width = volume_shape_dhw
    linear_voxel = (
        batch_indices * (depth * height * width)
        + voxel_indices_dhw[:, 0] * (height * width)
        + voxel_indices_dhw[:, 1] * width
        + voxel_indices_dhw[:, 2]
    )
    _, inverse = torch.unique(linear_voxel, sorted=True, return_inverse=True)
    denominators = torch.zeros(
        int(inverse.max().item()) + 1,
        dtype=raw_affinity.dtype,
        device=raw_affinity.device,
    )
    denominators.scatter_add_(0, inverse, raw_affinity)
    return raw_affinity / denominators[inverse]


def _linear_voxel_indices(
    batch_indices: torch.Tensor,
    voxel_indices_dhw: torch.Tensor,
    volume_shape_dhw: tuple[int, int, int],
) -> torch.Tensor:
    depth, height, width = volume_shape_dhw
    return (
        batch_indices * (depth * height * width)
        + voxel_indices_dhw[:, 0] * (height * width)
        + voxel_indices_dhw[:, 1] * width
        + voxel_indices_dhw[:, 2]
    )


def _unravel_linear_voxels(
    linear_voxels: torch.Tensor,
    volume_shape_dhw: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    depth, height, width = volume_shape_dhw
    per_volume = depth * height * width
    batch = torch.div(linear_voxels, per_volume, rounding_mode="floor")
    remainder = torch.remainder(linear_voxels, per_volume)
    d = torch.div(remainder, height * width, rounding_mode="floor")
    remainder = torch.remainder(remainder, height * width)
    h = torch.div(remainder, width, rounding_mode="floor")
    w = torch.remainder(remainder, width)
    return batch, torch.stack((d, h, w), dim=-1)


def build_sparse_pou(
    point_field: PointField,
    semantic_volume: torch.Tensor,
    geometry: VolumeGeometry,
    *,
    valid_brain_mask: torch.Tensor | None = None,
    max_local_voxels_per_point: int = _DEFAULT_MAX_LOCAL_VOXELS_PER_POINT,
) -> SparsePoU:
    """Build local semantic-aware PoU edges from point-centre RAS semantics.

    The routine enumerates only each affine-bounded sphere and returns one
    positive edge per point/covered voxel.  An empty global edge set is an
    explicit :class:`EmptySparseSupportError`, never a silently empty output.
    """
    if not isinstance(point_field, PointField):
        raise TypeError("point_field must be a PointField")
    if not isinstance(geometry, VolumeGeometry):
        raise TypeError("geometry must be a VolumeGeometry")
    if not isinstance(max_local_voxels_per_point, int) or max_local_voxels_per_point <= 0:
        raise ValueError("max_local_voxels_per_point must be a positive integer")
    batch, _, depth, height, width = _validate_semantic_volume(semantic_volume, point_field, geometry)
    shape_dhw = (depth, height, width)
    brain_mask = _canonical_brain_mask(
        valid_brain_mask,
        batch=batch,
        shape_dhw=shape_dhw,
        device=semantic_volume.device,
    )

    geometry_dtype = (
        semantic_volume.dtype
        if semantic_volume.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    affine = torch.as_tensor(
        geometry.voxel_to_ras_mm,
        dtype=geometry_dtype,
        device=semantic_volume.device,
    )
    inverse_spatial = torch.linalg.inv(affine[:3, :3])
    origin_ras_mm = affine[:3, 3]

    batch_parts: list[torch.Tensor] = []
    voxel_parts: list[torch.Tensor] = []
    point_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    spatial_batch_parts: list[torch.Tensor] = []
    spatial_voxel_parts: list[torch.Tensor] = []
    centres = point_field.refined_centers_ras_mm
    point_semantics = point_field.semantic_vectors

    for batch_index in range(batch):
        for point_index in range(centres.shape[1]):
            center_ras_mm = centres[batch_index, point_index]
            d_bounds, h_bounds, w_bounds = _index_bounds_whd(
                center_ras_mm.to(dtype=geometry_dtype),
                inverse_spatial,
                origin_ras_mm,
                point_field.support_radius_mm,
                shape_dhw,
            )
            point_spatial_count = 0
            for local_dhw in _iter_local_voxel_index_chunks(
                d_bounds,
                h_bounds,
                w_bounds,
                chunk_voxels=min(max_local_voxels_per_point, _MAX_ENUMERATION_CHUNK_VOXELS),
                device=semantic_volume.device,
            ):
                local_ras_mm = _ras_mm_from_voxel_indices(local_dhw, affine)
                local_distance_mm = torch.linalg.vector_norm(local_ras_mm - center_ras_mm, dim=-1)
                spatial = spatial_affinity(local_distance_mm, point_field.support_radius_mm)
                keep = spatial > 0.0
                if brain_mask is not None:
                    keep = keep & brain_mask[
                        batch_index,
                        local_dhw[:, 0],
                        local_dhw[:, 1],
                        local_dhw[:, 2],
                    ]
                if not bool(keep.any()):
                    continue
                local_dhw = local_dhw[keep]
                spatial = spatial[keep]
                point_spatial_count += local_dhw.shape[0]
                # Apply the explicit cap to the *actual* compact spherical
                # support, not its conservative rectangular index bound.  For
                # example, a 4-mm sphere at 0.5-mm spacing has a 17^3 bounding
                # box but fewer than 4096 in-sphere voxel centres.
                if point_spatial_count > max_local_voxels_per_point:
                    raise PointGuidedGeometryError(
                        "a compact spherical PoU neighbourhood exceeds max_local_voxels_per_point; "
                        "increase the explicit bound for this geometry"
                    )
                spatial_batch_parts.append(
                    torch.full(
                        (local_dhw.shape[0],),
                        batch_index,
                        dtype=torch.long,
                        device=semantic_volume.device,
                    )
                )
                spatial_voxel_parts.append(local_dhw)
                voxel_semantics = semantic_volume[
                    batch_index,
                    :,
                    local_dhw[:, 0],
                    local_dhw[:, 1],
                    local_dhw[:, 2],
                ].transpose(0, 1)
                raw = spatial * semantic_affinity(point_semantics[batch_index, point_index], voxel_semantics)
                keep_positive = raw > 0.0
                if not bool(keep_positive.any()):
                    continue
                local_dhw = local_dhw[keep_positive]
                raw = raw[keep_positive]
                edge_count = raw.numel()
                batch_parts.append(torch.full((edge_count,), batch_index, dtype=torch.long, device=semantic_volume.device))
                voxel_parts.append(local_dhw)
                point_parts.append(torch.full((edge_count,), point_index, dtype=torch.long, device=semantic_volume.device))
                raw_parts.append(raw)

    if not spatial_voxel_parts:
        raise EmptySparseSupportError("no compact-support voxels remain after the valid brain mask")
    spatial_batch_indices = torch.cat(spatial_batch_parts, dim=0)
    spatial_voxel_indices_dhw = torch.cat(spatial_voxel_parts, dim=0)
    spatial_linear = torch.unique(
        _linear_voxel_indices(spatial_batch_indices, spatial_voxel_indices_dhw, shape_dhw),
        sorted=True,
    )
    if raw_parts:
        batch_indices = torch.cat(batch_parts, dim=0)
        voxel_indices_dhw = torch.cat(voxel_parts, dim=0)
        point_indices = torch.cat(point_parts, dim=0)
        raw_affinity = torch.cat(raw_parts, dim=0)
        normalized_weight = _normalize_per_voxel(batch_indices, voxel_indices_dhw, raw_affinity, shape_dhw)
        supported_linear = torch.unique(
            _linear_voxel_indices(batch_indices, voxel_indices_dhw, shape_dhw),
            sorted=True,
        )
    else:
        supported_linear = torch.empty(0, dtype=torch.long, device=semantic_volume.device)
    unsupported_linear = spatial_linear[~torch.isin(spatial_linear, supported_linear)]
    unsupported_batch_indices, unsupported_voxel_indices_dhw = _unravel_linear_voxels(
        unsupported_linear,
        shape_dhw,
    )
    if not raw_parts:
        raise EmptySparseSupportError(
            "no positive compact-support PoU contributions exist",
            unsupported_batch_indices=unsupported_batch_indices,
            unsupported_voxel_indices_dhw=unsupported_voxel_indices_dhw,
        )
    return SparsePoU(
        batch_indices=batch_indices,
        voxel_indices_dhw=voxel_indices_dhw,
        point_indices=point_indices,
        raw_affinity=raw_affinity,
        normalized_weight=normalized_weight,
        unsupported_batch_indices=unsupported_batch_indices,
        unsupported_voxel_indices_dhw=unsupported_voxel_indices_dhw,
        volume_shape_dhw=shape_dhw,
    )


class SparseSemanticPoU(nn.Module):
    """Configuration-bound, parameter-free wrapper for :func:`build_sparse_pou`."""

    def __init__(self, config: PointGuidedConfig) -> None:
        super().__init__()
        if not isinstance(config, PointGuidedConfig):
            raise TypeError("config must be a PointGuidedConfig")
        self.config = config

    def forward(
        self,
        point_field: PointField,
        semantic_volume: torch.Tensor,
        geometry: VolumeGeometry,
        *,
        valid_brain_mask: torch.Tensor | None = None,
    ) -> SparsePoU:
        if point_field.semantic_vectors.shape[-1] != self.config.num_semantic_classes:
            raise ValueError("point_field semantics do not match config.num_semantic_classes")
        if not math.isclose(point_field.support_radius_mm, self.config.support_radius_mm, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("point_field support radius does not match the locked configuration")
        return build_sparse_pou(
            point_field,
            semantic_volume,
            geometry,
            valid_brain_mask=valid_brain_mask,
            max_local_voxels_per_point=self.config.max_local_voxels_per_point,
        )


build_semantic_aware_sparse_pou = build_sparse_pou


__all__ = [
    "SparseSemanticPoU",
    "build_semantic_aware_sparse_pou",
    "build_sparse_pou",
]
