"""Static Phase-6 SWT-Haar spectral anchor built from base tri-planes only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import PointGuidedConfig
from .swt_haar import SWT_HAAR_BAND_NAMES, SwtHaarBands, TwoLevelSwtHaar
from .triplane_projection import BaseTriPlanes


SPECTRAL_ANCHOR_INPUT_CHANNELS = 64
SPECTRAL_ANCHOR_CHANNELS_PER_BAND = 8
SPECTRAL_ANCHOR_BAND_COUNT = len(SWT_HAAR_BAND_NAMES)
SPECTRAL_ANCHOR_CHANNELS = SPECTRAL_ANCHOR_BAND_COUNT * SPECTRAL_ANCHOR_CHANNELS_PER_BAND

__all__ = [
    "SPECTRAL_ANCHOR_BAND_COUNT",
    "SPECTRAL_ANCHOR_CHANNELS",
    "SPECTRAL_ANCHOR_CHANNELS_PER_BAND",
    "SPECTRAL_ANCHOR_INPUT_CHANNELS",
    "SpectralAnchor",
    "StaticSpectralAnchor",
]


def _validate_anchor_plane(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a rank-4 torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.shape[0] <= 0 or value.shape[1] != SPECTRAL_ANCHOR_CHANNELS or any(
        length <= 0 for length in value.shape[-2:]
    ):
        raise ValueError(f"{name} must have shape [B, {SPECTRAL_ANCHOR_CHANNELS}, H, W] with positive dimensions")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class SpectralAnchor:
    """Named static spectral planes with stable seven-by-eight channel layout.

    The first 8-channel block is ``LL2``; the remaining blocks follow
    :data:`SWT_HAAR_BAND_NAMES` exactly.  ``xy`` preserves Bxy's Y/X grid,
    ``xz`` preserves Bxz's Z/X grid, and ``yz`` preserves Byz's Z/Y grid.
    This is a Phase-6 diagnostic representation only: it contains no point
    query, geometry mapping, reliability, or trajectory state.
    """

    xy: Tensor  # [B, 56, H, W]
    xz: Tensor  # [B, 56, D, W]
    yz: Tensor  # [B, 56, D, H]

    def __post_init__(self) -> None:
        for name in ("xy", "xz", "yz"):
            _validate_anchor_plane(name, getattr(self, name))
        batch, _, height, width = self.xy.shape
        if self.xz.shape[0] != batch or self.yz.shape[0] != batch:
            raise ValueError("spectral anchor planes must share their batch dimension")
        if self.xz.shape[-1] != width or self.yz.shape[-1] != height or self.xz.shape[-2] != self.yz.shape[-2]:
            raise ValueError("spectral anchor planes must agree on retained DHW dimensions")
        if len({self.xy.device, self.xz.device, self.yz.device}) != 1:
            raise ValueError("spectral anchor planes must share one device")
        if len({self.xy.dtype, self.xz.dtype, self.yz.dtype}) != 1:
            raise ValueError("spectral anchor planes must share one dtype")

    @property
    def axy(self) -> Tensor:
        """PLAN spelling for :attr:`xy`."""

        return self.xy

    @property
    def axz(self) -> Tensor:
        """PLAN spelling for :attr:`xz`."""

        return self.xz

    @property
    def ayz(self) -> Tensor:
        """PLAN spelling for :attr:`yz`."""

        return self.yz


class StaticSpectralAnchor(nn.Module):
    """Build one static Phase-6 spectral anchor from three supplied B planes.

    A single persistent ``Conv2d(64, 8, 1, bias=True)`` is intentionally
    reused for all seven bands of all three planes.  The bias follows PyTorch's
    default and is an implementation detail, not a newly locked scientific
    decision.  Fixed SWT buffers and this projector remain autograd-connected
    to B; this module never detaches or mutates B in place.
    """

    def __init__(self, config: PointGuidedConfig, *, input_channels: int) -> None:
        super().__init__()
        if not isinstance(config, PointGuidedConfig):
            raise TypeError("config must be a PointGuidedConfig")
        if not isinstance(input_channels, int) or isinstance(input_channels, bool):
            raise ValueError("input_channels must be an integer")
        if input_channels != SPECTRAL_ANCHOR_INPUT_CHANNELS:
            raise ValueError("Phase-6 spectral anchor is locked to 64-channel base planes")

        self.config = config
        self.input_channels = input_channels
        self.anchor_norm = config.anchor_norm
        self.swt = TwoLevelSwtHaar()
        # This is deliberately the sole learned Phase-6 projection module.
        self.band_projector = nn.Conv2d(
            SPECTRAL_ANCHOR_INPUT_CHANNELS,
            SPECTRAL_ANCHOR_CHANNELS_PER_BAND,
            kernel_size=1,
            bias=True,
        )
        self.band_gn: nn.GroupNorm | None = None
        if self.anchor_norm == "band_gn":
            self.band_gn = nn.GroupNorm(
                num_groups=SPECTRAL_ANCHOR_BAND_COUNT,
                num_channels=SPECTRAL_ANCHOR_CHANNELS,
                # The optional normalization may not introduce additional
                # learnable Phase-6 state beyond the locked shared 1x1.
                affine=False,
            )
        elif self.anchor_norm != "none":
            raise ValueError(f"unsupported anchor_norm: {self.anchor_norm!r}")

    def _validate_base_planes(self, base_planes: BaseTriPlanes) -> None:
        if not isinstance(base_planes, BaseTriPlanes):
            raise TypeError("base_planes must be a BaseTriPlanes instance")
        for name, plane in (("base_planes.xy", base_planes.xy), ("base_planes.xz", base_planes.xz), ("base_planes.yz", base_planes.yz)):
            if plane.shape[1] != self.input_channels:
                raise ValueError(f"{name} must have exactly {self.input_channels} channels")
            if plane.device != self.band_projector.weight.device:
                raise ValueError(f"{name} device must match the persistent shared band projector")
            if plane.dtype != self.band_projector.weight.dtype:
                raise ValueError(f"{name} dtype must match the persistent shared band projector")

    def _project_bands(self, bands: SwtHaarBands) -> Tensor:
        """Project the named stable bands in their canonical layout order."""

        projected = tuple(self.band_projector(band) for band in bands.as_tuple())
        anchor_plane = torch.cat(projected, dim=1)
        if self.band_gn is not None:
            anchor_plane = self.band_gn(anchor_plane)
        return anchor_plane

    def forward(self, base_planes: BaseTriPlanes) -> SpectralAnchor:
        """Apply fixed SWT then the one shared learned band projector once."""

        self._validate_base_planes(base_planes)
        return SpectralAnchor(
            xy=self._project_bands(self.swt(base_planes.xy)),
            xz=self._project_bands(self.swt(base_planes.xz)),
            yz=self._project_bands(self.swt(base_planes.yz)),
        )
