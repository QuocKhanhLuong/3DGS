"""Explicit feature-grid geometry and compact encoder output contracts.

The feature tensor index order is ``[batch, channel, v, u]``.  A feature centre
``[v_f, u_f]`` maps to the input-plane pixel centre

``[offset_v + stride_v * v_f, offset_u + stride_u * u_f]``.

This contract prevents hidden half-pixel assumptions when support points sample
encoder maps at physical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import torch

from ..contracts.coordinates import PhysicalPlane


@dataclass(frozen=True)
class FeatureGridToPlaneTransform:
    """Locked half-pixel feature-grid geometry for one observed plane.

    Feature centres use PyTorch's ``align_corners=False`` convention.  A
    stride ``s`` centre is at input pixel coordinate ``(f + .5) * s - .5``;
    right/bottom padding for odd input shapes is represented by an explicit
    invalid feature mask rather than a different coordinate convention.
    """

    input_shape_hw: Sequence[int]
    feature_shape_hw: Sequence[int]
    stride_vu: Sequence[int] = (1, 1)
    offset_vu_input_pixels: Sequence[float] | None = None
    input_plane: PhysicalPlane | None = None

    def __post_init__(self) -> None:
        input_shape = tuple(self.input_shape_hw)
        feature_shape = tuple(self.feature_shape_hw)
        stride = tuple(self.stride_vu)
        if len(input_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in input_shape):
            raise ValueError("input_shape_hw must contain two positive integers")
        if len(feature_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in feature_shape):
            raise ValueError("feature_shape_hw must contain two positive integers")
        if len(stride) != 2 or stride[0] != stride[1] or stride[0] not in (1, 2, 4):
            raise ValueError("stride_vu must be one shared output stride in {1, 2, 4}")
        stride_value = stride[0]
        expected_offset = (stride_value - 1.0) / 2.0
        offset = (expected_offset, expected_offset) if self.offset_vu_input_pixels is None else tuple(float(value) for value in self.offset_vu_input_pixels)
        if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
            raise ValueError("offset_vu_input_pixels must contain two finite values")
        expected_shape = tuple((length + stride_value - 1) // stride_value for length in input_shape)
        if feature_shape != expected_shape:
            raise ValueError("feature_shape_hw must be ceil(input_shape_hw / output_stride) after right/bottom padding")
        if any(abs(value - expected_offset) > 1e-6 for value in offset):
            raise ValueError("offset_vu_input_pixels must use the locked half-pixel align_corners=False convention")
        if self.input_plane is not None:
            if not isinstance(self.input_plane, PhysicalPlane):
                raise TypeError("input_plane must be a PhysicalPlane or None")
            if tuple(self.input_plane.shape_hw) != input_shape:
                raise ValueError("input_plane shape does not match input_shape_hw")
        object.__setattr__(self, "input_shape_hw", input_shape)
        object.__setattr__(self, "feature_shape_hw", feature_shape)
        object.__setattr__(self, "stride_vu", stride)
        object.__setattr__(self, "offset_vu_input_pixels", offset)

    @property
    def output_stride(self) -> int:
        return self.stride_vu[0]

    @property
    def sampling_convention(self) -> str:
        return "half_pixel_align_corners_false"

    @property
    def valid_feature_shape_hw(self) -> tuple[int, int]:
        return tuple(length // self.output_stride for length in self.input_shape_hw)

    def valid_feature_mask(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Return the topology-only mask that excludes right/bottom padding."""
        feature_height, feature_width = self.feature_shape_hw
        valid_height, valid_width = self.valid_feature_shape_hw
        v = torch.arange(feature_height, device=device) < valid_height
        u = torch.arange(feature_width, device=device) < valid_width
        return v[:, None] & u[None, :]

    @property
    def valid_feature_mask_hash(self) -> str:
        mask = self.valid_feature_mask().to(dtype=torch.uint8).contiguous().cpu()
        return hashlib.sha256(mask.numpy().tobytes()).hexdigest()

    @property
    def source_plane_hash(self) -> str:
        """Canonical source-plane identity, including observation provenance."""
        if self.input_plane is None:
            raise ValueError("source_plane_hash requires a bound canonical input_plane")
        return hashlib.sha256(self.input_plane.canonical_json().encode("utf-8")).hexdigest()

    def input_vu_from_feature_vu(
        self, v: float | torch.Tensor, u: float | torch.Tensor
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Return the source-plane pixel-centre coordinate for a feature centre."""
        return (
            (v + 0.5) * self.output_stride - 0.5,
            (u + 0.5) * self.output_stride - 0.5,
        )

    def feature_vu_from_input_vu(
        self, v: float | torch.Tensor, u: float | torch.Tensor
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Inverse of :meth:`input_vu_from_feature_vu` without grid rounding."""
        return ((v + 0.5) / self.output_stride - 0.5, (u + 0.5) / self.output_stride - 0.5)

    def _resolve_plane(self, plane: PhysicalPlane | None) -> PhysicalPlane:
        if self.input_plane is not None:
            if plane is not None and (
                not isinstance(plane, PhysicalPlane)
                or plane.canonical_json() != self.input_plane.canonical_json()
            ):
                raise ValueError("explicit plane must canonically match this transform's bound source PhysicalPlane")
            resolved = self.input_plane
        else:
            resolved = plane
        if not isinstance(resolved, PhysicalPlane):
            raise ValueError("a PhysicalPlane is required when this transform has no bound input_plane")
        if tuple(resolved.shape_hw) != self.input_shape_hw:
            raise ValueError("plane shape does not match feature transform input_shape_hw")
        return resolved

    def ras_mm_from_feature_vu(
        self,
        v: float | torch.Tensor,
        u: float | torch.Tensor,
        *,
        plane: PhysicalPlane | None = None,
    ) -> torch.Tensor:
        """Map continuous feature coordinates to canonical RAS millimetres."""
        resolved = self._resolve_plane(plane)
        input_v, input_u = self.input_vu_from_feature_vu(v, u)
        reference = input_v if isinstance(input_v, torch.Tensor) else input_u
        if isinstance(reference, torch.Tensor):
            dtype, device = reference.dtype, reference.device
            v_tensor = input_v if isinstance(input_v, torch.Tensor) else torch.as_tensor(input_v, dtype=dtype, device=device)
            u_tensor = input_u if isinstance(input_u, torch.Tensor) else torch.as_tensor(input_u, dtype=dtype, device=device)
        else:
            dtype, device = torch.float64, None
            v_tensor = torch.as_tensor(input_v, dtype=dtype)
            u_tensor = torch.as_tensor(input_u, dtype=dtype)
        origin = torch.as_tensor(resolved.pixel_center_origin_ras_mm, dtype=dtype, device=device)
        axis_u = torch.as_tensor(resolved.axis_u_ras, dtype=dtype, device=device)
        axis_v = torch.as_tensor(resolved.axis_v_ras, dtype=dtype, device=device)
        return origin + (u_tensor * resolved.spacing_uv_mm[0]).unsqueeze(-1) * axis_u + (v_tensor * resolved.spacing_uv_mm[1]).unsqueeze(-1) * axis_v

    def grid_sample_coordinates(
        self, ras_mm: torch.Tensor, *, plane: PhysicalPlane | None = None
    ) -> torch.Tensor:
        """Return ``[..., 2]`` ``(x, y)`` coordinates for ``grid_sample``.

        The resulting coordinates sample a feature tensor using
        ``align_corners=False`` and preserve gradients to ``ras_mm``.
        """
        if not isinstance(ras_mm, torch.Tensor) or ras_mm.shape[-1] != 3:
            raise ValueError("ras_mm must be a tensor with final dimension 3")
        if ras_mm.dtype not in (torch.float32, torch.float64):
            raise TypeError("ras_mm must use float32 or float64")
        resolved = self._resolve_plane(plane)
        origin = torch.as_tensor(resolved.pixel_center_origin_ras_mm, dtype=ras_mm.dtype, device=ras_mm.device)
        axis_u = torch.as_tensor(resolved.axis_u_ras, dtype=ras_mm.dtype, device=ras_mm.device)
        axis_v = torch.as_tensor(resolved.axis_v_ras, dtype=ras_mm.dtype, device=ras_mm.device)
        delta = ras_mm - origin
        input_u = (delta * axis_u).sum(dim=-1) / resolved.spacing_uv_mm[0]
        input_v = (delta * axis_v).sum(dim=-1) / resolved.spacing_uv_mm[1]
        feature_v, feature_u = self.feature_vu_from_input_vu(input_v, input_u)
        feature_height, feature_width = self.feature_shape_hw
        x = 2.0 * (feature_u + 0.5) / feature_width - 1.0
        y = 2.0 * (feature_v + 0.5) / feature_height - 1.0
        return torch.stack((x, y), dim=-1)

    def world_from_feature_vu(self, plane: PhysicalPlane, v: float, u: float) -> tuple[float, float, float]:
        point = self.ras_mm_from_feature_vu(v, u, plane=plane)
        return tuple(float(value) for value in point.tolist())


@dataclass(frozen=True)
class EncoderFeatureMaps:
    """Compact structural, appearance, and reliability maps for one mini-batch."""

    structural: torch.Tensor  # [B, C_str, Hf, Wf]
    appearance: torch.Tensor  # [B, C_app, Hf, Wf]
    reliability: torch.Tensor  # [B, 1, Hf, Wf]
    grid_to_plane: FeatureGridToPlaneTransform
    modality_ids: tuple[str, ...] = ()
    valid_feature_mask: torch.Tensor | None = None  # [B, 1, Hf, Wf], bool

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
        valid_feature_mask = self.valid_feature_mask
        if valid_feature_mask is None:
            valid_feature_mask = self.grid_to_plane.valid_feature_mask(device=device).expand(batch, 1, -1, -1)
        if (
            not isinstance(valid_feature_mask, torch.Tensor)
            or valid_feature_mask.shape != (batch, 1, *feature_shape)
            or valid_feature_mask.dtype is not torch.bool
            or valid_feature_mask.device != device
        ):
            raise ValueError("valid_feature_mask must be bool with shape [B, 1, Hf, Wf] on the feature device")
        if not bool(valid_feature_mask.flatten(1).any(dim=1).all()):
            raise ValueError("every feature map requires at least one valid feature centre")
        allowed_mask = self.grid_to_plane.valid_feature_mask(device=device).expand(batch, 1, -1, -1)
        if bool((valid_feature_mask & ~allowed_mask).any()):
            raise ValueError("valid_feature_mask cannot mark right/bottom padded feature centres as legal")
        if self.modality_ids and (
            len(self.modality_ids) != batch
            or any(not isinstance(value, str) or not value for value in self.modality_ids)
        ):
            raise ValueError("modality_ids must be empty or contain one non-empty ID per batch item")
        object.__setattr__(self, "modality_ids", tuple(self.modality_ids))
        object.__setattr__(self, "valid_feature_mask", valid_feature_mask)

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
