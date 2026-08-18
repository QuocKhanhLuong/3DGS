"""Small learned raw-offset predictor for point refinement."""

from __future__ import annotations

import torch
from torch import nn

from .config import PointGuidedConfig
from .directional import directional_descriptor_channels


class OffsetPredictor(nn.Module):
    """A deliberately small per-point MLP that predicts a raw RAS displacement."""

    def __init__(self, descriptor_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if descriptor_channels <= 0 or hidden_channels <= 0:
            raise ValueError("descriptor_channels and hidden_channels must be positive")
        self.descriptor_channels = int(descriptor_channels)
        self.hidden_channels = int(hidden_channels)
        self.network = nn.Sequential(
            nn.Linear(self.descriptor_channels, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, 3),
        )

    @classmethod
    def from_config(cls, config: PointGuidedConfig) -> "OffsetPredictor":
        return cls(
            directional_descriptor_channels(config.num_semantic_classes),
            config.offset_hidden_channels,
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(descriptor, torch.Tensor)
            or descriptor.ndim != 3
            or descriptor.shape[-1] != self.descriptor_channels
            or not descriptor.is_floating_point()
        ):
            raise ValueError(
                f"descriptor must be a floating-point [B, N, {self.descriptor_channels}] tensor"
            )
        if not bool(torch.isfinite(descriptor).all()):
            raise ValueError("descriptor must be finite")
        raw_displacement = self.network(descriptor)
        # The MLP may execute under AMP, but its output is a physical RAS-mm
        # displacement that immediately enters the refinement geometry path.
        # Restore the caller's coordinate dtype at this module boundary rather
        # than weakening physical-coordinate validation downstream.
        return raw_displacement.to(dtype=descriptor.dtype)


__all__ = ["OffsetPredictor"]
