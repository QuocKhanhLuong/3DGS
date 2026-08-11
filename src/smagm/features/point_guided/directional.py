"""Point-centred directional MRI/semantic context without 3-D patches."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .config import PointGuidedConfig
from .contracts import VolumeGeometry
from .sampling import sample_volume_ras_mm


DIRECTIONAL_OFFSETS_MM: tuple[float, float, float] = (1.0, 2.0, 3.0)


def _validate_centres(centres_ras_mm: torch.Tensor) -> None:
    if (
        not isinstance(centres_ras_mm, torch.Tensor)
        or centres_ras_mm.ndim != 3
        or centres_ras_mm.shape[-1] != 3
        or centres_ras_mm.shape[0] <= 0
        or centres_ras_mm.shape[1] <= 0
    ):
        raise ValueError("centres_ras_mm must have shape [B, N, 3]")
    if not centres_ras_mm.is_floating_point():
        raise TypeError("centres_ras_mm must be floating point")
    if not bool(torch.isfinite(centres_ras_mm).all()):
        raise ValueError("centres_ras_mm must be finite")


def directional_offsets_ras_mm(
    offsets_mm: Sequence[float] = DIRECTIONAL_OFFSETS_MM,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return centre then ``+/- RAS x/y/z`` offsets at each locked distance.

    The row order is centre followed by, for every distance in ascending order,
    ``(+x, -x, +y, -y, +z, -z)``.  It is deliberately expressed in RAS mm,
    not in anisotropic voxel units.
    """

    values = tuple(float(value) for value in offsets_mm)
    if values != DIRECTIONAL_OFFSETS_MM or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("directional offsets are locked to (1.0, 2.0, 3.0) mm")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    rows = [(0.0, 0.0, 0.0)]
    for distance in values:
        for axis in range(3):
            positive = [0.0, 0.0, 0.0]
            positive[axis] = distance
            rows.append(tuple(positive))
            rows.append(tuple(-value for value in positive))
    return torch.tensor(rows, dtype=dtype, device=device)


def directional_locations_ras_mm(
    centres_ras_mm: torch.Tensor,
    offsets_mm: Sequence[float] = DIRECTIONAL_OFFSETS_MM,
) -> torch.Tensor:
    """Return ``[B, N, 19, 3]`` exact physical directional sample locations."""

    _validate_centres(centres_ras_mm)
    offsets = directional_offsets_ras_mm(offsets_mm, device=centres_ras_mm.device, dtype=centres_ras_mm.dtype)
    return centres_ras_mm.unsqueeze(-2) + offsets.view(1, 1, -1, 3)


def directional_descriptor_channels(num_semantic_classes: int) -> int:
    if num_semantic_classes <= 1:
        raise ValueError("num_semantic_classes must be greater than one")
    return (1 + 2 * 3 * len(DIRECTIONAL_OFFSETS_MM)) * (3 + num_semantic_classes)


def _validate_volumes(
    mri_volume: torch.Tensor,
    coarse_semantic: torch.Tensor,
    centres_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
) -> None:
    for name, volume in (("mri_volume", mri_volume), ("coarse_semantic", coarse_semantic)):
        if not isinstance(volume, torch.Tensor) or volume.ndim != 5 or not volume.is_floating_point():
            raise ValueError(f"{name} must be a floating-point [B, C, D, H, W] tensor")
        if not bool(torch.isfinite(volume).all()):
            raise ValueError(f"{name} must be finite")
        if tuple(volume.shape[-3:]) != tuple(geometry.shape_dhw):
            raise ValueError(f"{name} spatial shape must agree with geometry")
    if mri_volume.shape[1] != 3:
        raise ValueError("mri_volume channels must be ordered [T1, T2, FLAIR]")
    if coarse_semantic.shape[0] != mri_volume.shape[0] or coarse_semantic.device != mri_volume.device or coarse_semantic.dtype != mri_volume.dtype:
        raise ValueError("mri_volume and coarse_semantic must share batch, device, and dtype")
    _validate_centres(centres_ras_mm)
    if centres_ras_mm.shape[0] != mri_volume.shape[0] or centres_ras_mm.device != mri_volume.device or centres_ras_mm.dtype != mri_volume.dtype:
        raise ValueError("centres_ras_mm must share mri_volume batch, device, and dtype")


def build_directional_descriptor(
    mri_volume: torch.Tensor,
    coarse_semantic: torch.Tensor,
    centres_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
    *,
    offsets_mm: Sequence[float] = DIRECTIONAL_OFFSETS_MM,
) -> torch.Tensor:
    """Sample centre and directional values, returning centre-relative features.

    The feature order is centre values ``[semantic, T1, T2, FLAIR]`` followed
    by the 18 directional samples in :func:`directional_offsets_ras_mm` order,
    each minus the centre.  This is a point descriptor, never a cropped 3-D
    patch; points outside the volume use standard zero padding only for that
    directional sample.
    """

    _validate_volumes(mri_volume, coarse_semantic, centres_ras_mm, geometry)
    locations = directional_locations_ras_mm(centres_ras_mm, offsets_mm)
    batch, points, locations_per_point, _ = locations.shape
    joint_volume = torch.cat((coarse_semantic, mri_volume), dim=1)
    samples = sample_volume_ras_mm(
        joint_volume,
        locations.reshape(batch, points * locations_per_point, 3),
        geometry,
        padding_mode="zeros",
    ).reshape(batch, points, locations_per_point, joint_volume.shape[1])
    centre = samples[:, :, 0]
    relative = samples[:, :, 1:] - centre.unsqueeze(2)
    return torch.cat((centre, relative.flatten(start_dim=2)), dim=-1)


class DirectionalDescriptor(nn.Module):
    """Configuration-bound directional descriptor for the locked frontend."""

    def __init__(self, config: PointGuidedConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def output_channels(self) -> int:
        return directional_descriptor_channels(self.config.num_semantic_classes)

    def forward(
        self,
        mri_volume: torch.Tensor,
        coarse_semantic: torch.Tensor,
        centres_ras_mm: torch.Tensor,
        geometry: VolumeGeometry,
    ) -> torch.Tensor:
        if coarse_semantic.shape[1] != self.config.num_semantic_classes:
            raise ValueError("coarse_semantic channels must equal config.num_semantic_classes")
        return build_directional_descriptor(
            mri_volume,
            coarse_semantic,
            centres_ras_mm,
            geometry,
            offsets_mm=self.config.directional_offsets_mm,
        )


__all__ = [
    "DIRECTIONAL_OFFSETS_MM",
    "DirectionalDescriptor",
    "build_directional_descriptor",
    "directional_descriptor_channels",
    "directional_locations_ras_mm",
    "directional_offsets_ras_mm",
]
