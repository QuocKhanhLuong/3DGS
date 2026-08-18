"""Gate-C C5 shared local state correction network."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .state_init import DYNAMIC_STATE_CHANNELS


UPDATER_INPUT_CHANNELS = 96 + 168 + 3 + 3
UPDATER_HIDDEN_CHANNELS = 128
UPDATER_OUTPUT_CHANNELS = 3 * DYNAMIC_STATE_CHANNELS


@dataclass(frozen=True)
class PlaneCorrections:
    """One 32-channel correction per dynamic plane for every batch item."""

    xy: Tensor  # [B,32]
    xz: Tensor  # [B,32]
    yz: Tensor  # [B,32]

    def __post_init__(self) -> None:
        for name in ("xy", "xz", "yz"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != DYNAMIC_STATE_CHANNELS or not value.is_floating_point():
                raise ValueError(f"{name} must have shape [B,32]")
            if value.shape[0] <= 0 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must have positive batch and finite values")
        for name in ("xz", "yz"):
            value = getattr(self, name)
            if value.shape != self.xy.shape or value.dtype != self.xy.dtype or value.device != self.xy.device:
                raise ValueError(f"{name} must match xy shape, dtype, and device")

    @property
    def packed(self) -> Tensor:
        return torch.cat((self.xy, self.xz, self.yz), dim=-1)


class UpdateNet(nn.Module):
    """The one shared locked `270 -> 128 -> 96` bounded correction network."""

    def __init__(self, *, hidden_channels: int = UPDATER_HIDDEN_CHANNELS) -> None:
        super().__init__()
        if hidden_channels != UPDATER_HIDDEN_CHANNELS:
            raise ValueError("Gate-C MAIN UpdateNet hidden width must be exactly 128")
        self.network = nn.Sequential(
            nn.Linear(UPDATER_INPUT_CHANNELS, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, UPDATER_OUTPUT_CHANNELS, bias=True),
        )

    def forward(self, updater_input: Tensor, *, write_scale: float) -> PlaneCorrections:
        if not isinstance(updater_input, Tensor) or updater_input.ndim != 2 or updater_input.shape[1] != UPDATER_INPUT_CHANNELS:
            raise ValueError("updater_input must have shape [B,270]")
        if not updater_input.is_floating_point() or updater_input.shape[0] <= 0 or not bool(torch.isfinite(updater_input).all()):
            raise ValueError("updater_input must be finite floating values")
        scale = float(write_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("write_scale must be positive and finite")
        correction = torch.tanh(self.network(updater_input)) * scale
        xy, xz, yz = correction.split(DYNAMIC_STATE_CHANNELS, dim=-1)
        return PlaneCorrections(xy=xy, xz=xz, yz=yz)


__all__ = [
    "PlaneCorrections",
    "UPDATER_HIDDEN_CHANNELS",
    "UPDATER_INPUT_CHANNELS",
    "UPDATER_OUTPUT_CHANNELS",
    "UpdateNet",
]
