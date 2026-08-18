"""Gate-C C2 dynamic-state point query and the locked shared RewardNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cross_plane_consistency import CONSISTENCY_DESCRIPTOR_CHANNELS
from .spectral_query import FeatureGridGeometry
from .state_init import DYNAMIC_STATE_CHANNELS, DynamicTriPlanes
from .updater import UPDATER_OUTPUT_CHANNELS


STATE_QUERY_CHANNELS = 3 * DYNAMIC_STATE_CHANNELS
BASE_REWARD_DESCRIPTOR_CHANNELS = STATE_QUERY_CHANNELS + 3 + CONSISTENCY_DESCRIPTOR_CHANNELS + 3
ACTION_DESCRIPTOR_CHANNELS = UPDATER_OUTPUT_CHANNELS
REWARD_DESCRIPTOR_CHANNELS = BASE_REWARD_DESCRIPTOR_CHANNELS + ACTION_DESCRIPTOR_CHANNELS
REWARD_HIDDEN_CHANNELS = 64


def _float_tensor(name: str, value: Tensor, *, rank: int, last: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != rank or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating rank-{rank} tensor")
    if value.shape[-1] != last or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have positive shape [B, N, {last}]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _sample_plane(plane: Tensor, row: Tensor, column: Tensor) -> Tensor:
    grid = torch.stack(
        (
            (2.0 * column + 1.0) / float(plane.shape[-1]) - 1.0,
            (2.0 * row + 1.0) / float(plane.shape[-2]) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(2)
    # ``grid`` derives from physical RAS-mm geometry and deliberately retains
    # its coordinate dtype.  The dynamic state is latent storage and may be
    # autocast to lower precision, so align it to the physical query only at
    # the differentiable sampling boundary rather than narrowing geometry.
    with torch.autocast(device_type=grid.device.type, enabled=False):
        sampling_plane = plane.to(dtype=grid.dtype)
        sampled = F.grid_sample(
            sampling_plane,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    return sampled[..., 0].transpose(1, 2)


@dataclass(frozen=True)
class DynamicPointSamples:
    """Current-state queries at fixed refined points in XY/XZ/YZ order."""

    xy: Tensor  # [B,N,32]
    xz: Tensor  # [B,N,32]
    yz: Tensor  # [B,N,32]

    def __post_init__(self) -> None:
        named = (("xy", self.xy), ("xz", self.xz), ("yz", self.yz))
        for name, value in named:
            _float_tensor(name, value, rank=3, last=DYNAMIC_STATE_CHANNELS)
        for name, value in named[1:]:
            if value.shape != self.xy.shape or value.dtype != self.xy.dtype or value.device != self.xy.device:
                raise ValueError(f"{name} must match xy shape, dtype, and device")

    @property
    def packed(self) -> Tensor:
        return torch.cat((self.xy, self.xz, self.yz), dim=-1)


@dataclass(frozen=True)
class GateBDescriptorContext:
    """Internal fixed Gate-B descriptors reused without recovering from f_spec."""

    q_xy: Tensor  # [B,N,24]
    q_xz: Tensor  # [B,N,24]
    q_yz: Tensor  # [B,N,24]

    def __post_init__(self) -> None:
        named = (("q_xy", self.q_xy), ("q_xz", self.q_xz), ("q_yz", self.q_yz))
        for name, value in named:
            _float_tensor(name, value, rank=3, last=CONSISTENCY_DESCRIPTOR_CHANNELS)
        for name, value in named[1:]:
            if value.shape != self.q_xy.shape or value.dtype != self.q_xy.dtype or value.device != self.q_xy.device:
                raise ValueError(f"{name} must match q_xy shape, dtype, and device")

    def reliability_weighted_mean(self, reliability: Tensor) -> Tensor:
        _float_tensor("reliability", reliability, rank=3, last=3)
        if reliability.shape[:2] != self.q_xy.shape[:2] or reliability.dtype != self.q_xy.dtype or reliability.device != self.q_xy.device:
            raise ValueError("reliability must align with Gate-B descriptor tensors")
        return (
            reliability[..., 0:1] * self.q_xy
            + reliability[..., 1:2] * self.q_xz
            + reliability[..., 2:3] * self.q_yz
        )


class DynamicStatePointQuery(nn.Module):
    """Parameter-free, pointwise bilinear query of the current dynamic state."""

    def forward(
        self,
        state: DynamicTriPlanes,
        points_ras_mm: Tensor,
        feature_geometry: FeatureGridGeometry,
    ) -> DynamicPointSamples:
        if not isinstance(state, DynamicTriPlanes):
            raise TypeError("state must be a DynamicTriPlanes instance")
        if not isinstance(feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be a FeatureGridGeometry instance")
        _float_tensor("points_ras_mm", points_ras_mm, rank=3, last=3)
        if points_ras_mm.shape[0] != state.xy.shape[0] or points_ras_mm.device != state.xy.device:
            raise ValueError("points_ras_mm must match dynamic-state batch and device")
        depth, height, width = feature_geometry.shape_dhw
        expected = ((state.xy, (height, width)), (state.xz, (depth, width)), (state.yz, (depth, height)))
        if any(tuple(plane.shape[-2:]) != shape for plane, shape in expected):
            raise ValueError("dynamic state planes must retain the derived selected-feature grids")
        d, h, w = feature_geometry.ras_mm_to_feature_dhw(points_ras_mm).unbind(dim=-1)
        return DynamicPointSamples(
            xy=_sample_plane(state.xy, h, w),
            xz=_sample_plane(state.xz, d, w),
            yz=_sample_plane(state.yz, d, h),
        )


def build_reward_descriptor(
    dynamic_samples: DynamicPointSamples,
    point_semantic: Tensor,
    gate_b_descriptors: GateBDescriptorContext,
    reliability: Tensor,
    candidate_updates: Tensor,
) -> Tensor:
    """Build the target-free ``[base descriptor | candidate update]`` input."""

    if not isinstance(dynamic_samples, DynamicPointSamples):
        raise TypeError("dynamic_samples must be a DynamicPointSamples instance")
    if not isinstance(gate_b_descriptors, GateBDescriptorContext):
        raise TypeError("gate_b_descriptors must be a GateBDescriptorContext instance")
    _float_tensor("point_semantic", point_semantic, rank=3, last=3)
    _float_tensor("reliability", reliability, rank=3, last=3)
    _float_tensor("candidate_updates", candidate_updates, rank=3, last=ACTION_DESCRIPTOR_CHANNELS)
    reference = dynamic_samples.xy
    for name, value in (
        ("point_semantic", point_semantic),
        ("reliability", reliability),
        ("q_xy", gate_b_descriptors.q_xy),
        ("candidate_updates", candidate_updates),
    ):
        if value.shape[:2] != reference.shape[:2] or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must align with dynamic state samples")
    descriptor = torch.cat(
        (
            dynamic_samples.packed,
            point_semantic,
            gate_b_descriptors.reliability_weighted_mean(reliability),
            reliability,
            candidate_updates,
        ),
        dim=-1,
    )
    if descriptor.shape[-1] != REWARD_DESCRIPTOR_CHANNELS:
        raise RuntimeError(f"Gate-C RewardNet descriptor must have exactly {REWARD_DESCRIPTOR_CHANNELS} channels")
    return descriptor


class RewardNet(nn.Module):
    """The one shared locked state/action-dependent `222 -> 64 -> 1 -> sigmoid` score."""

    def __init__(self, *, hidden_channels: int = REWARD_HIDDEN_CHANNELS) -> None:
        super().__init__()
        if hidden_channels != REWARD_HIDDEN_CHANNELS:
            raise ValueError("Gate-C MAIN RewardNet hidden width must be exactly 64")
        self.network = nn.Sequential(
            nn.Linear(REWARD_DESCRIPTOR_CHANNELS, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, descriptor: Tensor) -> Tensor:
        _float_tensor("descriptor", descriptor, rank=3, last=REWARD_DESCRIPTOR_CHANNELS)
        return self.network(descriptor).squeeze(-1)


__all__ = [
    "DynamicPointSamples",
    "DynamicStatePointQuery",
    "GateBDescriptorContext",
    "ACTION_DESCRIPTOR_CHANNELS",
    "BASE_REWARD_DESCRIPTOR_CHANNELS",
    "REWARD_DESCRIPTOR_CHANNELS",
    "REWARD_HIDDEN_CHANNELS",
    "RewardNet",
    "STATE_QUERY_CHANNELS",
    "build_reward_descriptor",
]
