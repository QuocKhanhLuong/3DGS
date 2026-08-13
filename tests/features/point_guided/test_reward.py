"""Focused C2 invariants for dynamic queries and state-dependent reward."""

from __future__ import annotations

import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.reward import (
    DynamicStatePointQuery,
    GateBDescriptorContext,
    REWARD_DESCRIPTOR_CHANNELS,
    RewardNet,
    build_reward_descriptor,
)
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes


def _geometry() -> FeatureGridGeometry:
    source = VolumeGeometry.from_spacing((3, 5, 7), (1.0, 1.0, 1.0))
    return FeatureGridGeometry(
        source_geometry=source,
        feature_geometry=source,
        tap="conv1_pre_maxpool",
        feature_to_source_scale_dhw=(1.0, 1.0, 1.0),
        feature_to_source_offset_dhw=(0.0, 0.0, 0.0),
        operator_chain=("synthetic",),
    )


def _state() -> DynamicTriPlanes:
    xy = torch.arange(32 * 5 * 7, dtype=torch.float32).reshape(1, 32, 5, 7)
    xz = torch.arange(32 * 3 * 7, dtype=torch.float32).reshape(1, 32, 3, 7)
    yz = torch.arange(32 * 3 * 5, dtype=torch.float32).reshape(1, 32, 3, 5)
    return DynamicTriPlanes(xy=xy, xz=xz, yz=yz)


def _gate_b(batch: int = 1, points: int = 2) -> GateBDescriptorContext:
    return GateBDescriptorContext(
        q_xy=torch.full((batch, points, 24), 1.0),
        q_xz=torch.full((batch, points, 24), 2.0),
        q_yz=torch.full((batch, points, 24), 3.0),
    )


def test_dynamic_query_is_pointwise_bilinear_and_uses_xy_xz_yz_axis_order() -> None:
    geometry = _geometry()
    points = geometry.feature_dhw_to_ras_mm(torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]]))
    samples = DynamicStatePointQuery()(_state(), points, geometry)

    assert samples.xy.shape == samples.xz.shape == samples.yz.shape == (1, 2, 32)
    torch.testing.assert_close(samples.xy[0, 0], _state().xy[0, :, 2, 3])
    torch.testing.assert_close(samples.xz[0, 0], _state().xz[0, :, 1, 3])
    torch.testing.assert_close(samples.yz[0, 0], _state().yz[0, :, 1, 2])
    assert sum(parameter.numel() for parameter in DynamicStatePointQuery().parameters()) == 0


def test_reward_descriptor_is_exact_126_d_and_reliability_weighted_q_bar() -> None:
    geometry = _geometry()
    points = geometry.feature_dhw_to_ras_mm(torch.tensor([[[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]]]))
    samples = DynamicStatePointQuery()(_state(), points, geometry)
    semantic = torch.tensor([[[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]])
    reliability = torch.tensor([[[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]])
    descriptor = build_reward_descriptor(samples, semantic, _gate_b(), reliability)

    assert descriptor.shape == (1, 2, REWARD_DESCRIPTOR_CHANNELS)
    torch.testing.assert_close(descriptor[..., 99:123], torch.tensor([[[2.3] * 24, [1.0] * 24]]))
    torch.testing.assert_close(descriptor[..., 123:], reliability)


def test_shared_reward_net_is_state_dependent_bounded_and_receives_gradient() -> None:
    torch.manual_seed(7)
    descriptor = torch.randn(2, 5, REWARD_DESCRIPTOR_CHANNELS, requires_grad=True)
    reward_net = RewardNet()
    reward = reward_net(descriptor)

    assert reward.shape == (2, 5)
    assert bool(((reward >= 0.0) & (reward <= 1.0)).all())
    assert sum(parameter.numel() for parameter in reward_net.parameters()) == 8193
    changed = descriptor.detach().clone()
    changed[..., :96] += 1.0
    assert not torch.allclose(reward, reward_net(changed))
    reward.square().mean().backward()
    assert descriptor.grad is not None and bool(descriptor.grad.abs().sum() > 0.0)
    assert all(parameter.grad is not None for parameter in reward_net.parameters())
