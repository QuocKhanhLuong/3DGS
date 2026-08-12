"""PLAN-locked static base tri-plane projection from one selected feature map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import PointGuidedConfig

__all__ = ["BaseTriPlaneProjector", "BaseTriPlaneScores", "BaseTriPlanes"]


def _validate_float_tensor(name: str, value: Tensor, dimensions: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != dimensions:
        raise ValueError(f"{name} must be a rank-{dimensions} torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class BaseTriPlanes:
    """Named static base planes in the frontend's locked DHW/RAS convention.

    ``xy`` is ``Bxy`` and has collapsed D/Z; ``xz`` is ``Bxz`` and has
    collapsed H/Y; ``yz`` is ``Byz`` and has collapsed W/X.
    """

    xy: Tensor  # [B, C, H, W]
    xz: Tensor  # [B, C, D, W]
    yz: Tensor  # [B, C, D, H]

    def __post_init__(self) -> None:
        for name in ("xy", "xz", "yz"):
            _validate_float_tensor(name, getattr(self, name), 4)
        batch, channels, height, width = self.xy.shape
        if any(length <= 0 for plane in (self.xy, self.xz, self.yz) for length in plane.shape):
            raise ValueError("base planes must have positive dimensions")
        if self.xz.shape[:2] != (batch, channels) or self.yz.shape[:2] != (batch, channels):
            raise ValueError("base planes must share batch and channel dimensions")
        if (
            self.xz.shape[-1] != width
            or self.yz.shape[-1] != height
            or self.xz.shape[-2] != self.yz.shape[-2]
        ):
            raise ValueError("base planes must agree on their retained DHW dimensions")

    @property
    def bxy(self) -> Tensor:
        """Explicit PLAN spelling for :attr:`xy`."""

        return self.xy

    @property
    def bxz(self) -> Tensor:
        """Explicit PLAN spelling for :attr:`xz`."""

        return self.xz

    @property
    def byz(self) -> Tensor:
        """Explicit PLAN spelling for :attr:`yz`."""

        return self.yz


@dataclass(frozen=True)
class BaseTriPlaneScores:
    """Named scalar score/weight volumes aligned with the three base planes."""

    xy: Tensor  # [B, 1, D, H, W]
    xz: Tensor  # [B, 1, D, H, W]
    yz: Tensor  # [B, 1, D, H, W]

    def __post_init__(self) -> None:
        for name in ("xy", "xz", "yz"):
            _validate_float_tensor(name, getattr(self, name), 5)
        shape = self.xy.shape
        if shape[0] <= 0 or shape[1] != 1 or any(length <= 0 for length in shape[-3:]):
            raise ValueError("base-plane scores must have shape [B, 1, D, H, W]")
        if self.xz.shape != shape or self.yz.shape != shape:
            raise ValueError("base-plane scores must share shape [B, 1, D, H, W]")

    @property
    def bxy(self) -> Tensor:
        return self.xy

    @property
    def bxz(self) -> Tensor:
        return self.xz

    @property
    def byz(self) -> Tensor:
        return self.yz


class BaseTriPlaneProjector(nn.Module):
    """Project a supplied selected map ``[B,C,D,H,W]`` into static base planes.

    The caller owns the Phase 2 tap and detach decision. This module neither
    calls MedicalNet nor changes the autograd attachment of the supplied map.
    PyTorch's DHW order maps to physical ZYX: XY collapses D/Z, XZ collapses
    H/Y, and YZ collapses W/X.
    """

    def __init__(self, config: PointGuidedConfig, *, input_channels: int) -> None:
        super().__init__()
        if not isinstance(config, PointGuidedConfig):
            raise TypeError("config must be a PointGuidedConfig")
        if not isinstance(input_channels, int) or isinstance(input_channels, bool) or input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")

        self.config = config
        self.input_channels = input_channels
        self.projection_mode = config.projection_mode
        self.pointwise_scorer: nn.Conv3d | None = None
        self.xy_scorer: nn.Conv3d | None = None
        self.xz_scorer: nn.Conv3d | None = None
        self.yz_scorer: nn.Conv3d | None = None

        if self.projection_mode == "pointwise_weighted":
            self.pointwise_scorer = self._new_zero_initialized_scorer((1, 1, 1), (0, 0, 0))
        elif self.projection_mode == "axis_local_weighted":
            self.xy_scorer = self._new_zero_initialized_scorer((3, 1, 1), (1, 0, 0))
            self.xz_scorer = self._new_zero_initialized_scorer((1, 3, 1), (0, 1, 0))
            self.yz_scorer = self._new_zero_initialized_scorer((1, 1, 3), (0, 0, 1))
        elif self.projection_mode not in ("mean", "max"):
            raise ValueError(f"unsupported projection_mode: {self.projection_mode!r}")

    def _new_zero_initialized_scorer(
        self,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
    ) -> nn.Conv3d:
        scorer = nn.Conv3d(
            self.input_channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        nn.init.zeros_(scorer.weight)
        nn.init.zeros_(scorer.bias)
        return scorer

    def _validate_feature(self, feature: Tensor) -> None:
        _validate_float_tensor("feature", feature, 5)
        if feature.shape[0] <= 0 or feature.shape[1] != self.input_channels or any(
            length <= 0 for length in feature.shape[-3:]
        ):
            raise ValueError(
                "feature must have positive shape [B, C, D, H, W] with the configured input channels"
            )

    def _scorer_logits(self, feature: Tensor) -> BaseTriPlaneScores:
        if self.projection_mode == "pointwise_weighted":
            if self.pointwise_scorer is None:
                raise RuntimeError("pointwise_weighted projection is missing its scorer")
            logits = self.pointwise_scorer(feature)
            return BaseTriPlaneScores(xy=logits, xz=logits, yz=logits)
        if self.projection_mode == "axis_local_weighted":
            if self.xy_scorer is None or self.xz_scorer is None or self.yz_scorer is None:
                raise RuntimeError("axis_local_weighted projection is missing an axis-local scorer")
            return BaseTriPlaneScores(
                xy=self.xy_scorer(feature),
                xz=self.xz_scorer(feature),
                yz=self.yz_scorer(feature),
            )
        raise RuntimeError("scorer logits are available only for weighted projection modes")

    def scorer_logits(self, feature: Tensor) -> BaseTriPlaneScores:
        """Return scalar ``[B,1,D,H,W]`` logits for weighted modes only."""

        self._validate_feature(feature)
        return self._scorer_logits(feature)

    def _normalized_weights(self, feature: Tensor) -> BaseTriPlaneScores:
        logits = self._scorer_logits(feature)
        return BaseTriPlaneScores(
            xy=torch.softmax(logits.xy, dim=2),
            xz=torch.softmax(logits.xz, dim=3),
            yz=torch.softmax(logits.yz, dim=4),
        )

    def normalized_weights(self, feature: Tensor) -> BaseTriPlaneScores:
        """Return non-negative weights normalized along each plane's collapsed axis."""

        self._validate_feature(feature)
        return self._normalized_weights(feature)

    @staticmethod
    def _mean_projection(feature: Tensor) -> BaseTriPlanes:
        return BaseTriPlanes(
            xy=feature.mean(dim=2),
            xz=feature.mean(dim=3),
            yz=feature.mean(dim=4),
        )

    @staticmethod
    def _max_projection(feature: Tensor) -> BaseTriPlanes:
        return BaseTriPlanes(
            xy=feature.max(dim=2).values,
            xz=feature.max(dim=3).values,
            yz=feature.max(dim=4).values,
        )

    @staticmethod
    def _weighted_projection(feature: Tensor, weights: BaseTriPlaneScores) -> BaseTriPlanes:
        return BaseTriPlanes(
            xy=(feature * weights.xy).sum(dim=2),
            xz=(feature * weights.xz).sum(dim=3),
            yz=(feature * weights.yz).sum(dim=4),
        )

    def forward(self, feature: Tensor) -> BaseTriPlanes:
        """Apply the configured PLAN-locked static base-plane projection."""

        self._validate_feature(feature)
        if self.projection_mode == "mean":
            return self._mean_projection(feature)
        if self.projection_mode == "max":
            return self._max_projection(feature)
        return self._weighted_projection(feature, self._normalized_weights(feature))
