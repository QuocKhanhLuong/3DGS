"""Focused C6 tests for compact physical local tri-plane writes."""

from __future__ import annotations

import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes
from smagm.features.point_guided.updater import PlaneCorrections
from smagm.features.point_guided.writeback import CompactTriPlaneWriteback


def _geometry() -> FeatureGridGeometry:
    source = VolumeGeometry.from_spacing((9, 11, 13), (1.0, 1.0, 1.0))
    return FeatureGridGeometry(
        source_geometry=source,
        feature_geometry=source,
        tap="conv1_pre_maxpool",
        feature_to_source_scale_dhw=(1.0, 1.0, 1.0),
        feature_to_source_offset_dhw=(0.0, 0.0, 0.0),
        operator_chain=("synthetic",),
    )


def _state() -> DynamicTriPlanes:
    return DynamicTriPlanes(
        xy=torch.zeros(1, 32, 11, 13),
        xz=torch.zeros(1, 32, 9, 13),
        yz=torch.zeros(1, 32, 9, 11),
    )


def test_compact_write_is_physical_local_and_preserves_xy_xz_yz_coordinate_mapping() -> None:
    geometry = _geometry()
    point = geometry.feature_dhw_to_ras_mm(torch.tensor([[[4.0, 5.0, 6.0]]]))[:, 0]
    corrections = PlaneCorrections(
        xy=torch.ones(1, 32),
        xz=torch.full((1, 32), 2.0),
        yz=torch.full((1, 32), 3.0),
    )
    before = _state()
    after = CompactTriPlaneWriteback()(before, point, corrections, geometry)

    torch.testing.assert_close(after.xy[0, :, 5, 6], torch.ones(32))
    torch.testing.assert_close(after.xz[0, :, 4, 6], torch.full((32,), 2.0))
    torch.testing.assert_close(after.yz[0, :, 4, 5], torch.full((32,), 3.0))
    assert int((after.xy != 0.0).sum()) < after.xy.numel()
    assert int((after.xz != 0.0).sum()) < after.xz.numel()
    assert int((after.yz != 0.0).sum()) < after.yz.numel()
    torch.testing.assert_close(before.xy, torch.zeros_like(before.xy))
    torch.testing.assert_close(before.xz, torch.zeros_like(before.xz))
    torch.testing.assert_close(before.yz, torch.zeros_like(before.yz))


def test_write_support_is_zero_outside_four_mm_and_gradients_reach_corrections() -> None:
    geometry = _geometry()
    point_all = geometry.feature_dhw_to_ras_mm(torch.tensor([[[4.0, 5.0, 6.0]]])).detach().requires_grad_(True)
    point = point_all[:, 0]
    correction = torch.ones(1, 32, requires_grad=True)
    corrections = PlaneCorrections(xy=correction, xz=correction * 2.0, yz=correction * 3.0)
    after = CompactTriPlaneWriteback()(_state(), point, corrections, geometry)

    torch.testing.assert_close(after.xy[..., 0, 0], torch.zeros_like(after.xy[..., 0, 0]))
    after.xy.square().mean().backward()
    assert correction.grad is not None and bool(correction.grad.abs().sum() > 0.0)
    assert point_all.grad is not None and bool(torch.isfinite(point_all.grad).all())
