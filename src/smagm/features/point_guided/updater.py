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


@dataclass(frozen=True)
class CandidateCorrections:
    """One bounded update vector per ``[batch, candidate]`` pair."""

    xy: Tensor  # [B,N,32]
    xz: Tensor  # [B,N,32]
    yz: Tensor  # [B,N,32]

    def __post_init__(self) -> None:
        reference = self.xy
        if not isinstance(reference, Tensor) or reference.ndim != 3 or reference.shape[-1] != DYNAMIC_STATE_CHANNELS:
            raise ValueError("xy must have shape [B,N,32]")
        if reference.shape[0] <= 0 or reference.shape[1] <= 0 or not reference.is_floating_point() or not bool(torch.isfinite(reference).all()):
            raise ValueError("xy must be finite with positive batch and candidate dimensions")
        for name in ("xz", "yz"):
            value = getattr(self, name)
            if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must match xy shape, dtype, and device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")

    @property
    def packed(self) -> Tensor:
        """Return the exact ``[B,N,96]`` action feature consumed by RewardNet."""

        return torch.cat((self.xy, self.xz, self.yz), dim=-1)

    def rows(self, mask: Tensor) -> "CandidateCorrections":
        """Retain batch rows using a boolean mask without changing candidates."""

        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != self.xy.shape[0] or mask.device != self.xy.device:
            raise ValueError("row mask must be a device-matched bool [B] tensor")
        return CandidateCorrections(xy=self.xy[mask], xz=self.xz[mask], yz=self.yz[mask])

    def weighted(self, weights: Tensor) -> PlaneCorrections:
        """Apply hard/straight-through candidate weights to one update per row."""

        if not isinstance(weights, Tensor) or weights.ndim != 2 or weights.shape != self.xy.shape[:2] or weights.dtype != self.xy.dtype or weights.device != self.xy.device:
            raise ValueError("weights must be a device-matched floating [B,N] tensor")
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("weights must be finite")
        expanded = weights.unsqueeze(-1)
        return PlaneCorrections(
            xy=(self.xy * expanded).sum(dim=1),
            xz=(self.xz * expanded).sum(dim=1),
            yz=(self.yz * expanded).sum(dim=1),
        )


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

    def forward_candidates(self, updater_input: Tensor, *, write_scale: float) -> CandidateCorrections:
        """Predict the same bounded update for every candidate before selection."""

        if not isinstance(updater_input, Tensor) or updater_input.ndim != 3 or updater_input.shape[-1] != UPDATER_INPUT_CHANNELS:
            raise ValueError("updater_input must have shape [B,N,270]")
        if not updater_input.is_floating_point() or updater_input.shape[0] <= 0 or updater_input.shape[1] <= 0 or not bool(torch.isfinite(updater_input).all()):
            raise ValueError("updater_input must be finite floating values")
        batch, candidates = updater_input.shape[:2]
        corrections = self(
            updater_input.reshape(batch * candidates, UPDATER_INPUT_CHANNELS),
            write_scale=write_scale,
        )
        return CandidateCorrections(
            xy=corrections.xy.reshape(batch, candidates, DYNAMIC_STATE_CHANNELS),
            xz=corrections.xz.reshape(batch, candidates, DYNAMIC_STATE_CHANNELS),
            yz=corrections.yz.reshape(batch, candidates, DYNAMIC_STATE_CHANNELS),
        )


__all__ = [
    "CandidateCorrections",
    "PlaneCorrections",
    "UPDATER_HIDDEN_CHANNELS",
    "UPDATER_INPUT_CHANNELS",
    "UPDATER_OUTPUT_CHANNELS",
    "UpdateNet",
]
