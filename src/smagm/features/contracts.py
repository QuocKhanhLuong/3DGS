"""Explicit feature-grid geometry and compact encoder output contracts.

The feature tensor index order is ``[batch, channel, v, u]``.  A feature centre
``[v_f, u_f]`` maps to the input-plane pixel centre

``[offset_v + stride_v * v_f, offset_u + stride_u * u_f]``.

This contract prevents hidden half-pixel assumptions when support points sample
encoder maps at physical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from ..contracts.coordinates import PhysicalPlane


@dataclass(frozen=True)
class FeatureGridToPlaneTransform:
    """Map feature-grid centres to one observed physical MRI plane."""

    input_shape_hw: Sequence[int]
    feature_shape_hw: Sequence[int]
    stride_vu: Sequence[int] = (1, 1)
    offset_vu_input_pixels: Sequence[float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        input_shape = tuple(self.input_shape_hw)
        feature_shape = tuple(self.feature_shape_hw)
        stride = tuple(self.stride_vu)
        offset = tuple(float(value) for value in self.offset_vu_input_pixels)
        if len(input_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in input_shape):
            raise ValueError("input_shape_hw must contain two positive integers")
        if len(feature_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in feature_shape):
            raise ValueError("feature_shape_hw must contain two positive integers")
        if len(stride) != 2 or any(not isinstance(value, int) or value <= 0 for value in stride):
            raise ValueError("stride_vu must contain two positive integers")
        if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
            raise ValueError("offset_vu_input_pixels must contain two finite values")
        max_v = offset[0] + stride[0] * (feature_shape[0] - 1)
        max_u = offset[1] + stride[1] * (feature_shape[1] - 1)
        if offset[0] < -1e-6 or offset[1] < -1e-6:
            raise ValueError("feature centres must not start before the input image")
        if max_v > input_shape[0] - 1 + 1e-6 or max_u > input_shape[1] - 1 + 1e-6:
            raise ValueError("feature centres must lie inside the input image")
        object.__setattr__(self, "input_shape_hw", input_shape)
        object.__setattr__(self, "feature_shape_hw", feature_shape)
        object.__setattr__(self, "stride_vu", stride)
        object.__setattr__(self, "offset_vu_input_pixels", offset)

    def input_vu_from_feature_vu(self, v: float, u: float) -> tuple[float, float]:
        """Return the source-plane pixel-centre coordinate for a feature centre."""
        return (
            self.offset_vu_input_pixels[0] + float(v) * self.stride_vu[0],
            self.offset_vu_input_pixels[1] + float(u) * self.stride_vu[1],
        )

    def world_from_feature_vu(self, plane: PhysicalPlane, v: float, u: float) -> tuple[float, float, float]:
        if tuple(plane.shape_hw) != self.input_shape_hw:
            raise ValueError("plane shape does not match feature transform input_shape_hw")
        input_v, input_u = self.input_vu_from_feature_vu(v, u)
        return plane.world_from_vu(input_v, input_u)


@dataclass(frozen=True)
class EncoderFeatureMaps:
    """Compact structural, appearance, and reliability maps for one mini-batch."""

    structural: torch.Tensor  # [B, C_str, Hf, Wf]
    appearance: torch.Tensor  # [B, C_app, Hf, Wf]
    reliability: torch.Tensor  # [B, 1, Hf, Wf]
    grid_to_plane: FeatureGridToPlaneTransform
    modality_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, tensor in (
            ("structural", self.structural),
            ("appearance", self.appearance),
            ("reliability", self.reliability),
        ):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(f"{name} must be a rank-4 torch.Tensor")
            if tensor.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"{name} must use float32 or float64")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite")
        batch = self.structural.shape[0]
        feature_shape = tuple(self.structural.shape[-2:])
        if batch <= 0 or self.structural.shape[1] <= 0 or self.appearance.shape[1] <= 0:
            raise ValueError("feature maps require positive batch and channel dimensions")
        if self.appearance.shape[0] != batch or tuple(self.appearance.shape[-2:]) != feature_shape:
            raise ValueError("appearance must share batch and spatial shape with structural")
        if self.reliability.shape != (batch, 1, *feature_shape):
            raise ValueError("reliability must have shape [B, 1, Hf, Wf]")
        device = self.structural.device
        dtype = self.structural.dtype
        if self.appearance.device != device or self.reliability.device != device:
            raise ValueError("all feature maps must share device")
        if self.appearance.dtype != dtype or self.reliability.dtype != dtype:
            raise ValueError("all feature maps must share dtype")
        if not isinstance(self.grid_to_plane, FeatureGridToPlaneTransform):
            raise TypeError("grid_to_plane must be a FeatureGridToPlaneTransform")
        if feature_shape != self.grid_to_plane.feature_shape_hw:
            raise ValueError("tensor spatial shape disagrees with grid_to_plane")
        if self.modality_ids and (
            len(self.modality_ids) != batch
            or any(not isinstance(value, str) or not value for value in self.modality_ids)
        ):
            raise ValueError("modality_ids must be empty or contain one non-empty ID per batch item")
        object.__setattr__(self, "modality_ids", tuple(self.modality_ids))

    @property
    def batch_size(self) -> int:
        return self.structural.shape[0]

    @property
    def feature_shape_hw(self) -> tuple[int, int]:
        return tuple(self.structural.shape[-2:])

    @property
    def concatenated_channels(self) -> int:
        return self.structural.shape[1] + self.appearance.shape[1] + 1

    def concatenated(self) -> torch.Tensor:
        return torch.cat((self.structural, self.appearance, self.reliability), dim=1)
