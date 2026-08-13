"""Deterministic Phase-7 cross-plane spectral consistency.

This module consumes only already-sampled raw 56-channel features from the
static XY, XZ, and YZ spectral-anchor planes.  It deliberately has no
geometry, sampling, learned confidence, or trajectory responsibility.  The
raw plane features remain intact apart from their final scalar reliability
weight and are packed in permanent ``XY, XZ, YZ`` provenance order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spectral_anchor import (
    SPECTRAL_ANCHOR_CHANNELS,
    SPECTRAL_ANCHOR_CHANNELS_PER_BAND,
)
from .swt_haar import SWT_HAAR_BAND_NAMES


SPECTRAL_BAND_COUNT = len(SWT_HAAR_BAND_NAMES)
"""Number of stored Phase-6 SWT-Haar bands in each raw plane feature."""

CONSISTENCY_DESCRIPTOR_CHANNELS = 3 * SPECTRAL_ANCHOR_CHANNELS_PER_BAND
"""Channels in the fixed ``[LL2, E1, E2]`` reliability descriptor."""

PLANE_COUNT = 3
"""The fixed XY, XZ, YZ plane count for Phase-7 evidence."""

POINT_SPECTRAL_CHANNELS = PLANE_COUNT * SPECTRAL_ANCHOR_CHANNELS
"""Channels in the final reliability-weighted XY/XZ/YZ packing."""

# This is intentionally the sole numerical-stability constant in this module.
# It is fixed, non-trainable, and shared by the energy and cosine calculations.
CONSISTENCY_EPSILON = 1.0e-6


__all__ = [
    "CONSISTENCY_DESCRIPTOR_CHANNELS",
    "CONSISTENCY_EPSILON",
    "CrossPlaneConsistency",
    "CrossPlaneConsistencyResult",
    "PLANE_COUNT",
    "POINT_SPECTRAL_CHANNELS",
    "SPECTRAL_BAND_COUNT",
    "consistency_descriptor",
]


_BAND_INDEX = {name: index for index, name in enumerate(SWT_HAAR_BAND_NAMES)}
_LEVEL_ONE_ORIENTATIONS = ("LH1", "HL1", "HH1")
_LEVEL_TWO_ORIENTATIONS = ("LH2", "HL2", "HH2")


def _validate_feature(name: str, feature: Tensor, *, channels: int) -> None:
    if not isinstance(feature, Tensor) or feature.ndim != 3:
        raise ValueError(f"{name} must be a rank-3 torch.Tensor [B, N, {channels}]")
    if not feature.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    batch, points, feature_channels = feature.shape
    if batch <= 0 or points <= 0 or feature_channels != channels:
        raise ValueError(f"{name} must have shape [B, N, {channels}] with positive B and N")
    if not bool(torch.isfinite(feature).all()):
        raise ValueError(f"{name} must be finite")


def _validate_matching_triplet(
    xy: Tensor,
    xz: Tensor,
    yz: Tensor,
    *,
    channels: int,
    label: str,
) -> None:
    named_features = ((f"{label}_xy", xy), (f"{label}_xz", xz), (f"{label}_yz", yz))
    for name, feature in named_features:
        _validate_feature(name, feature, channels=channels)

    reference = xy
    for name, feature in named_features[1:]:
        if feature.shape != reference.shape:
            raise ValueError(f"{name} must match {label}_xy shape")
        if feature.dtype != reference.dtype:
            raise ValueError(f"{name} must match {label}_xy dtype")
        if feature.device != reference.device:
            raise ValueError(f"{name} must match {label}_xy device")


def _energy(blocks: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """Return the elementwise SWT orientation energy at one locked scale."""

    first, second, third = blocks
    epsilon = first.new_tensor(CONSISTENCY_EPSILON)
    return torch.sqrt(first.square() + second.square() + third.square() + epsilon)


def _descriptor_from_valid_feature(feature: Tensor) -> Tensor:
    """Build ``[LL2, E1, E2]`` after the raw feature has been validated."""

    bands = feature.reshape(
        *feature.shape[:-1],
        SPECTRAL_BAND_COUNT,
        SPECTRAL_ANCHOR_CHANNELS_PER_BAND,
    )

    def block(name: str) -> Tensor:
        return bands[..., _BAND_INDEX[name], :]

    ll2 = block("LL2")
    e1 = _energy(tuple(block(name) for name in _LEVEL_ONE_ORIENTATIONS))
    e2 = _energy(tuple(block(name) for name in _LEVEL_TWO_ORIENTATIONS))
    return torch.cat((ll2, e1, e2), dim=-1)


def consistency_descriptor(feature: Tensor) -> Tensor:
    """Derive the fixed 24-d reliability descriptor from one raw 56-d feature.

    ``feature`` is a sampled anchor feature of shape ``[B, N, 56]`` whose
    blocks retain Phase-6's ``LL2, LH1, HL1, HH1, LH2, HL2, HH2`` order.  The
    returned descriptor is used only for reliability; it does not replace or
    alter the raw feature used in the final evidence packing.
    """

    _validate_feature("feature", feature, channels=SPECTRAL_ANCHOR_CHANNELS)
    return _descriptor_from_valid_feature(feature)


@dataclass(frozen=True)
class CrossPlaneConsistencyResult:
    """Typed diagnostics and final packed evidence for one batch of points.

    Pairwise cosine columns are ordered ``(xy_xz, xy_yz, xz_yz)``.  Agreement
    and reliability columns are ordered ``(xy, xz, yz)``.  ``spectral_evidence``
    has its fixed 56-channel blocks in that same ``XY, XZ, YZ`` plane order.
    """

    q_xy: Tensor
    q_xz: Tensor
    q_yz: Tensor
    pairwise_cosines: Tensor
    mean_agreement: Tensor
    reliability: Tensor
    spectral_evidence: Tensor

    def __post_init__(self) -> None:
        _validate_matching_triplet(
            self.q_xy,
            self.q_xz,
            self.q_yz,
            channels=CONSISTENCY_DESCRIPTOR_CHANNELS,
            label="q",
        )
        batch, points, _ = self.q_xy.shape
        expected_diagnostic_shape = (batch, points, PLANE_COUNT)
        for name, value in (
            ("pairwise_cosines", self.pairwise_cosines),
            ("mean_agreement", self.mean_agreement),
            ("reliability", self.reliability),
        ):
            _validate_feature(name, value, channels=PLANE_COUNT)
            if value.shape != expected_diagnostic_shape:
                raise ValueError(f"{name} must have shape [B, N, {PLANE_COUNT}]")
            if value.dtype != self.q_xy.dtype:
                raise ValueError(f"{name} must match q_xy dtype")
            if value.device != self.q_xy.device:
                raise ValueError(f"{name} must match q_xy device")

        _validate_feature(
            "spectral_evidence",
            self.spectral_evidence,
            channels=POINT_SPECTRAL_CHANNELS,
        )
        if self.spectral_evidence.shape[:2] != (batch, points):
            raise ValueError("spectral_evidence must match q_xy batch and point dimensions")
        if self.spectral_evidence.dtype != self.q_xy.dtype:
            raise ValueError("spectral_evidence must match q_xy dtype")
        if self.spectral_evidence.device != self.q_xy.device:
            raise ValueError("spectral_evidence must match q_xy device")
        if not bool(torch.all(self.reliability >= 0)):
            raise ValueError("reliability must be nonnegative")
        if not bool(torch.allclose(self.reliability.sum(dim=-1), torch.ones_like(self.reliability[..., 0]))):
            raise ValueError("reliability must sum to one across XY, XZ, and YZ")

    @property
    def alpha_xy(self) -> Tensor:
        """Reliability of the XY raw-plane feature, shape ``[B, N]``."""

        return self.reliability[..., 0]

    @property
    def alpha_xz(self) -> Tensor:
        """Reliability of the XZ raw-plane feature, shape ``[B, N]``."""

        return self.reliability[..., 1]

    @property
    def alpha_yz(self) -> Tensor:
        """Reliability of the YZ raw-plane feature, shape ``[B, N]``."""

        return self.reliability[..., 2]

    @property
    def f_spec(self) -> Tensor:
        """PLAN spelling for the final 168-d point spectral evidence."""

        return self.spectral_evidence


class CrossPlaneConsistency(nn.Module):
    """Apply the locked parameter-free cross-plane reliability rule.

    Inputs are the three geometry-sampled raw anchor features, each exactly
    ``[B, N, 56]``.  This module owns no coordinate conversion or sampling and
    has neither parameters nor buffers.  Its only output is diagnostics plus
    the 168-d reliability-weighted concatenation; it does not construct any
    dynamic state or reconstruction.
    """

    def forward(self, f_xy: Tensor, f_xz: Tensor, f_yz: Tensor) -> CrossPlaneConsistencyResult:
        _validate_matching_triplet(
            f_xy,
            f_xz,
            f_yz,
            channels=SPECTRAL_ANCHOR_CHANNELS,
            label="f",
        )

        q_xy = _descriptor_from_valid_feature(f_xy)
        q_xz = _descriptor_from_valid_feature(f_xz)
        q_yz = _descriptor_from_valid_feature(f_yz)

        cosine_xy_xz = F.cosine_similarity(q_xy, q_xz, dim=-1, eps=CONSISTENCY_EPSILON)
        cosine_xy_yz = F.cosine_similarity(q_xy, q_yz, dim=-1, eps=CONSISTENCY_EPSILON)
        cosine_xz_yz = F.cosine_similarity(q_xz, q_yz, dim=-1, eps=CONSISTENCY_EPSILON)
        pairwise_cosines = torch.stack(
            (cosine_xy_xz, cosine_xy_yz, cosine_xz_yz),
            dim=-1,
        )

        mean_agreement = torch.stack(
            (
                (cosine_xy_xz + cosine_xy_yz) / 2,
                (cosine_xy_xz + cosine_xz_yz) / 2,
                (cosine_xy_yz + cosine_xz_yz) / 2,
            ),
            dim=-1,
        )
        reliability = torch.softmax(mean_agreement, dim=-1)
        spectral_evidence = torch.cat(
            (
                reliability[..., 0:1] * f_xy,
                reliability[..., 1:2] * f_xz,
                reliability[..., 2:3] * f_yz,
            ),
            dim=-1,
        )

        return CrossPlaneConsistencyResult(
            q_xy=q_xy,
            q_xz=q_xz,
            q_yz=q_yz,
            pairwise_cosines=pairwise_cosines,
            mean_agreement=mean_agreement,
            reliability=reliability,
            spectral_evidence=spectral_evidence,
        )
