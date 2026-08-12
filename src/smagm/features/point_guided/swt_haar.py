"""Fixed two-level stationary Haar analysis for static 2-D base planes.

The transform accepts a plane of shape ``[B, C, H, W]``.  Its filter symbols
are deliberately tied to those tensor axes: the first symbol addresses the
H/row axis and the second addresses the W/column axis.  Therefore ``LH`` is
low-pass along H and high-pass along W, while ``HL`` is high-pass along H and
low-pass along W.  For the frontend's physical planes this means row/column
are Y/X for XY, Z/X for XZ, and Z/Y for YZ.

This module intentionally contains only the fixed SWT-Haar analysis.  It does
not project bands, mix planes, or query them at points.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["SWT_HAAR_BAND_NAMES", "SwtHaarBands", "TwoLevelSwtHaar"]


SWT_HAAR_BAND_NAMES = ("LL2", "LH1", "HL1", "HH1", "LH2", "HL2", "HH2")
"""Canonical public SWT-Haar band order used by the spectral anchor."""


def _validate_band_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a rank-4 torch.Tensor [B, C, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must have positive [B, C, H, W] dimensions")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class SwtHaarBands:
    """The seven public bands of the locked two-level SWT-Haar contract.

    ``LL1`` is purposefully absent: it is an internal approximation used only
    to produce level-two outputs.  :meth:`as_tuple` preserves the exact stable
    order declared by :data:`SWT_HAAR_BAND_NAMES`.
    """

    ll2: Tensor
    lh1: Tensor
    hl1: Tensor
    hh1: Tensor
    lh2: Tensor
    hl2: Tensor
    hh2: Tensor

    def __post_init__(self) -> None:
        values = self.as_tuple()
        for name, value in zip(SWT_HAAR_BAND_NAMES, values, strict=True):
            _validate_band_tensor(name, value)

        reference = values[0]
        for name, value in zip(SWT_HAAR_BAND_NAMES[1:], values[1:], strict=True):
            if value.shape != reference.shape:
                raise ValueError(f"{name} must match LL2 shape")
            if value.dtype != reference.dtype:
                raise ValueError(f"{name} must match LL2 dtype")
            if value.device != reference.device:
                raise ValueError(f"{name} must match LL2 device")

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return bands in ``(LL2, LH1, HL1, HH1, LH2, HL2, HH2)`` order."""

        return (self.ll2, self.lh1, self.hl1, self.hh1, self.lh2, self.hl2, self.hh2)


class TwoLevelSwtHaar(nn.Module):
    """Apply the fixed differentiable two-level 2-D stationary Haar transform.

    Level one uses dilation one.  Level two analyses only the internal ``LL1``
    approximation with dilation two (the stationary/a-trous step).  Both use
    explicit reflect padding and stride one, so every returned band stays on
    the input grid.  For a two-tap filter and dilation ``d``, total required
    padding per axis is ``d``.  We use a deterministic TensorFlow-SAME-style
    convention: floor half on left/top and the remaining sample on
    right/bottom.  Thus dilation one is right/bottom-biased ``(0, 1)`` and
    dilation two is symmetric ``(1, 1)``.

    The Haar coefficients are registered buffers rather than parameters.  The
    buffers are cast at use time to the input's dtype and device so a default
    module remains safe for, for example, float64 CPU inputs without mutating
    its persistent state.
    """

    def __init__(self) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(2.0)
        low = torch.tensor((scale, scale), dtype=torch.get_default_dtype())
        high = torch.tensor((scale, -scale), dtype=torch.get_default_dtype())

        # Keep the normalized 1-D filters visible as fixed buffers as well as
        # their separable 2-D analysis masks used by grouped conv2d.
        self.register_buffer("low_filter", low)
        self.register_buffer("high_filter", high)
        self.register_buffer("ll_filter", torch.outer(low, low).reshape(1, 1, 2, 2))
        self.register_buffer("lh_filter", torch.outer(low, high).reshape(1, 1, 2, 2))
        self.register_buffer("hl_filter", torch.outer(high, low).reshape(1, 1, 2, 2))
        self.register_buffer("hh_filter", torch.outer(high, high).reshape(1, 1, 2, 2))

    @staticmethod
    def _validate_plane(plane: Tensor) -> None:
        if not isinstance(plane, Tensor) or plane.ndim != 4:
            raise ValueError("plane must be a rank-4 torch.Tensor [B, C, H, W]")
        if not plane.is_floating_point():
            raise TypeError("plane must be floating point")
        batch, channels, height, width = plane.shape
        if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("plane must have positive [B, C, H, W] dimensions")
        if height == 1 or width == 1:
            raise ValueError(
                "reflect-padded SWT-Haar requires H and W to both be at least 2"
            )
        if not bool(torch.isfinite(plane).all()):
            raise ValueError("plane must be finite")

    @staticmethod
    def _reflect_same_pad(plane: Tensor, *, dilation: int) -> Tensor:
        """Apply explicit reflect padding that makes a dilated 2-tap conv same-grid."""

        if dilation not in (1, 2):
            raise ValueError(f"unsupported SWT-Haar dilation: {dilation}")
        before = dilation // 2
        after = dilation - before
        # F.pad's 2-D order is left, right, top, bottom.
        return F.pad(plane, (before, after, before, after), mode="reflect")

    @staticmethod
    def _grouped_analysis(
        plane: Tensor,
        filter_2d: Tensor,
        *,
        dilation: int,
    ) -> Tensor:
        """Apply one fixed separable analysis filter independently per channel."""

        padded = TwoLevelSwtHaar._reflect_same_pad(plane, dilation=dilation)
        channels = plane.shape[1]
        weight = filter_2d.to(device=plane.device, dtype=plane.dtype).expand(channels, -1, -1, -1)
        return F.conv2d(padded, weight, bias=None, stride=1, padding=0, dilation=dilation, groups=channels)

    def forward(self, plane: Tensor) -> SwtHaarBands:
        """Return the seven public SWT bands for a valid ``[B, C, H, W]`` plane."""

        self._validate_plane(plane)

        ll1 = self._grouped_analysis(plane, self.ll_filter, dilation=1)
        lh1 = self._grouped_analysis(plane, self.lh_filter, dilation=1)
        hl1 = self._grouped_analysis(plane, self.hl_filter, dilation=1)
        hh1 = self._grouped_analysis(plane, self.hh_filter, dilation=1)

        ll2 = self._grouped_analysis(ll1, self.ll_filter, dilation=2)
        lh2 = self._grouped_analysis(ll1, self.lh_filter, dilation=2)
        hl2 = self._grouped_analysis(ll1, self.hl_filter, dilation=2)
        hh2 = self._grouped_analysis(ll1, self.hh_filter, dilation=2)

        return SwtHaarBands(
            ll2=ll2,
            lh1=lh1,
            hl1=hl1,
            hh1=hh1,
            lh2=lh2,
            hl2=hl2,
            hh2=hh2,
        )
