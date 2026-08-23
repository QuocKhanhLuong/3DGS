"""Differentiable sampling between canonical RAS millimetres and volumes.

The frontend stores tensors in ``[B, C, D, H, W]`` order while the volume
affine consumes ``[w, h, d, 1]``.  Keeping the conversion here prevents a
silent switch between tensor and physical coordinate conventions at call
sites.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
from torch.nn import functional as F

from .contracts import PointGuidedGeometryError, VolumeGeometry


GridSamplePaddingMode = Literal["zeros", "border", "reflection"]


def _require_finite_float_tensor(name: str, value: torch.Tensor, *, final_dimension: int | None = None) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if final_dimension is not None and (value.ndim == 0 or value.shape[-1] != final_dimension):
        raise ValueError(f"{name} must have final dimension {final_dimension}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _affine_tensor(geometry: VolumeGeometry, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(geometry.voxel_to_ras_mm, dtype=dtype, device=device)


def _shape_dhw(shape_dhw: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(item) for item in shape_dhw)
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise PointGuidedGeometryError("shape_dhw must contain three positive integers")
    return shape


def voxel_dhw_to_ras_mm(voxel_dhw: torch.Tensor, geometry: VolumeGeometry) -> torch.Tensor:
    """Map continuous ``[..., d, h, w]`` voxel centres to RAS ``XYZ`` mm.

    This operation is tensor-only and therefore preserves a gradient to
    ``voxel_dhw`` when callers use it inside a differentiable path.
    """

    _require_finite_float_tensor("voxel_dhw", voxel_dhw, final_dimension=3)
    # Physical coordinates are an explicit interface boundary, not a neural
    # activation.  CUDA AMP would otherwise cast this matmul to fp16 even when
    # its coordinates are fp32, violating the point/refinement dtype contract.
    # Keep the affine operation in the caller-requested coordinate dtype while
    # preserving its gradient path.
    with torch.autocast(device_type=voxel_dhw.device.type, enabled=False):
        affine = _affine_tensor(geometry, dtype=voxel_dhw.dtype, device=voxel_dhw.device)
        whd = voxel_dhw[..., (2, 1, 0)]
        homogeneous = torch.cat((whd, torch.ones_like(whd[..., :1])), dim=-1)
        ras_mm = torch.matmul(homogeneous, affine.transpose(0, 1))[..., :3]
    return ras_mm


def ras_mm_to_voxel_dhw(ras_mm: torch.Tensor, geometry: VolumeGeometry) -> torch.Tensor:
    """Map canonical RAS ``XYZ`` mm to continuous ``[..., d, h, w]`` indices."""

    _require_finite_float_tensor("ras_mm", ras_mm, final_dimension=3)
    # Match the forward physical-coordinate boundary.  In particular, do not
    # feed an fp16 affine to linalg.inv or allow the inverse matmul to narrow
    # caller-owned fp32/fp64 RAS coordinates under ambient AMP.
    with torch.autocast(device_type=ras_mm.device.type, enabled=False):
        affine = _affine_tensor(geometry, dtype=ras_mm.dtype, device=ras_mm.device)
        inverse = torch.linalg.inv(affine)
        homogeneous = torch.cat((ras_mm, torch.ones_like(ras_mm[..., :1])), dim=-1)
        whd = torch.matmul(homogeneous, inverse.transpose(0, 1))[..., :3]
        voxel_dhw = whd[..., (2, 1, 0)]
    return voxel_dhw


def voxel_dhw_to_grid_sample_coordinates(
    voxel_dhw: torch.Tensor,
    shape_dhw: Sequence[int],
) -> torch.Tensor:
    """Return 5-D ``grid_sample`` coordinates ``[..., x, y, z]``.

    The conversion is the locked half-voxel formula for
    ``align_corners=False``.  In particular, voxel centre zero maps to
    ``-1 + 1 / length`` rather than ``-1``.
    """

    _require_finite_float_tensor("voxel_dhw", voxel_dhw, final_dimension=3)
    depth, height, width = _shape_dhw(shape_dhw)
    d, h, w = voxel_dhw.unbind(dim=-1)
    x = (2.0 * w + 1.0) / float(width) - 1.0
    y = (2.0 * h + 1.0) / float(height) - 1.0
    z = (2.0 * d + 1.0) / float(depth) - 1.0
    return torch.stack((x, y, z), dim=-1)


def ras_mm_in_bounds(
    ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
    *,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Return whether RAS points lie in the closed voxel-centre volume box."""

    if not math.isfinite(float(atol)) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    voxel_dhw = ras_mm_to_voxel_dhw(ras_mm, geometry)
    upper = torch.as_tensor(
        tuple(length - 1 for length in geometry.shape_dhw),
        dtype=voxel_dhw.dtype,
        device=voxel_dhw.device,
    )
    return ((voxel_dhw >= -atol) & (voxel_dhw <= upper + atol)).all(dim=-1)


def project_ras_mm_to_volume(ras_mm: torch.Tensor, geometry: VolumeGeometry) -> torch.Tensor:
    """Project points to the affine image of the valid voxel-centre box.

    This helper is useful for diagnostics.  Refinement itself uses a
    line-box projection from the original point so its displacement remains
    bounded in physical millimetres.
    """

    voxel_dhw = ras_mm_to_voxel_dhw(ras_mm, geometry)
    upper = torch.as_tensor(
        tuple(length - 1 for length in geometry.shape_dhw),
        dtype=voxel_dhw.dtype,
        device=voxel_dhw.device,
    )
    projected = torch.maximum(torch.minimum(voxel_dhw, upper), torch.zeros_like(voxel_dhw))
    return voxel_dhw_to_ras_mm(projected, geometry)


def _validate_volume_and_points(
    volume: torch.Tensor,
    points_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
) -> None:
    _require_finite_float_tensor("volume", volume)
    if volume.ndim != 5:
        raise ValueError("volume must have shape [B, C, D, H, W]")
    batch, channels, depth, height, width = volume.shape
    if batch <= 0 or channels <= 0:
        raise ValueError("volume must have positive batch and channel dimensions")
    if tuple(geometry.shape_dhw) != (depth, height, width):
        raise PointGuidedGeometryError("volume spatial shape must agree with geometry.shape_dhw")
    _require_finite_float_tensor("points_ras_mm", points_ras_mm, final_dimension=3)
    if points_ras_mm.ndim != 3 or points_ras_mm.shape[0] != batch or points_ras_mm.shape[1] <= 0:
        raise ValueError("points_ras_mm must have shape [B, N, 3] with the volume batch size")
    if points_ras_mm.device != volume.device:
        raise ValueError("volume and points_ras_mm must share one device")
    if points_ras_mm.dtype != volume.dtype:
        raise TypeError("volume and points_ras_mm must share one floating-point dtype")


def sample_volume_ras_mm(
    volume: torch.Tensor,
    points_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
    *,
    padding_mode: GridSamplePaddingMode = "zeros",
    require_in_bounds: bool = False,
) -> torch.Tensor:
    """Trilinearly sample a ``[B, C, D, H, W]`` volume at RAS-mm points.

    Returns ``[B, N, C]`` and uses PyTorch's 5-D ``grid_sample`` with
    ``mode='bilinear'`` (trilinear for a 5-D input) and
    ``align_corners=False``.  No coordinate rounding, indexing, or detaching
    occurs, so gradients flow to both the volume and the point locations.
    """

    _validate_volume_and_points(volume, points_ras_mm, geometry)
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
    if require_in_bounds and not bool(ras_mm_in_bounds(points_ras_mm, geometry).all()):
        raise PointGuidedGeometryError("all sampled RAS points must be inside the volume")

    voxel_dhw = ras_mm_to_voxel_dhw(points_ras_mm, geometry)
    grid = voxel_dhw_to_grid_sample_coordinates(voxel_dhw, geometry.shape_dhw)
    grid = grid.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, x/y/z]
    sampled = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    return sampled[:, :, :, 0, 0].transpose(1, 2)


# Compact aliases make the RAS/voxel convention obvious at older call sites.
ras_to_voxel_dhw = ras_mm_to_voxel_dhw
sample_at_ras_mm = sample_volume_ras_mm
voxel_dhw_to_ras = voxel_dhw_to_ras_mm


__all__ = [
    "GridSamplePaddingMode",
    "project_ras_mm_to_volume",
    "ras_mm_in_bounds",
    "ras_mm_to_voxel_dhw",
    "ras_to_voxel_dhw",
    "sample_at_ras_mm",
    "sample_volume_ras_mm",
    "voxel_dhw_to_grid_sample_coordinates",
    "voxel_dhw_to_ras",
    "voxel_dhw_to_ras_mm",
]
