"""Gate-C C1 shared initialization of dynamic tri-plane state from base planes.

The state is an internal reconstruction latent only.  It is deliberately
constructed from ``B`` alone: static anchor, semantics, point evidence, and
any target data are excluded at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .triplane_projection import BaseTriPlanes


DYNAMIC_STATE_CHANNELS = 32
"""Locked MAIN width of every dynamic Gate-C plane."""


def _validate_plane(name: str, value: Tensor, *, channels: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating rank-4 tensor [B, C, H, W]")
    if value.shape[0] <= 0 or value.shape[1] != channels or any(length <= 0 for length in value.shape[-2:]):
        raise ValueError(f"{name} must have positive shape [B, {channels}, H, W]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class DynamicTriPlanes:
    """Named dynamic planes retaining the static XY/XZ/YZ spatial grids."""

    xy: Tensor  # [B, 32, H, W]
    xz: Tensor  # [B, 32, D, W]
    yz: Tensor  # [B, 32, D, H]

    def __post_init__(self) -> None:
        for name in ("xy", "xz", "yz"):
            _validate_plane(name, getattr(self, name), channels=DYNAMIC_STATE_CHANNELS)
        batch, _, height, width = self.xy.shape
        if self.xz.shape[:2] != (batch, DYNAMIC_STATE_CHANNELS) or self.yz.shape[:2] != (
            batch,
            DYNAMIC_STATE_CHANNELS,
        ):
            raise ValueError("dynamic planes must share batch and channel dimensions")
        if self.xz.shape[-1] != width or self.yz.shape[-1] != height or self.xz.shape[-2] != self.yz.shape[-2]:
            raise ValueError("dynamic planes must retain consistent DHW dimensions")
        reference = self.xy
        for name in ("xz", "yz"):
            value = getattr(self, name)
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must share xy dtype and device")


class DynamicStateInitializer(nn.Module):
    """Apply the one locked shared ``64 -> 32`` pointwise map to all B planes."""

    def __init__(self, *, input_channels: int = 64, state_channels: int = DYNAMIC_STATE_CHANNELS) -> None:
        super().__init__()
        if input_channels != 64:
            raise ValueError("Gate-C MAIN base-plane input must have exactly 64 channels")
        if state_channels != DYNAMIC_STATE_CHANNELS:
            raise ValueError("Gate-C MAIN dynamic state must have exactly 32 channels")
        self.input_channels = input_channels
        self.state_channels = state_channels
        self.shared_projection = nn.Conv2d(input_channels, state_channels, kernel_size=1, bias=True)

    def forward(self, base_planes: BaseTriPlanes) -> DynamicTriPlanes:
        if not isinstance(base_planes, BaseTriPlanes):
            raise TypeError("base_planes must be a BaseTriPlanes instance")
        for name in ("xy", "xz", "yz"):
            _validate_plane(f"base_planes.{name}", getattr(base_planes, name), channels=self.input_channels)
        return DynamicTriPlanes(
            xy=self.shared_projection(base_planes.xy),
            xz=self.shared_projection(base_planes.xz),
            yz=self.shared_projection(base_planes.yz),
        )


__all__ = ["DYNAMIC_STATE_CHANNELS", "DynamicStateInitializer", "DynamicTriPlanes"]
