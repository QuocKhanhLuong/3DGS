"""Focused Phase-7 tests for derived feature geometry and anchor queries."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from smagm.features.point_guided.cross_plane_consistency import CrossPlaneConsistency
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10
from smagm.features.point_guided.sampling import voxel_dhw_to_ras_mm
from smagm.features.point_guided.spectral_anchor import SPECTRAL_ANCHOR_CHANNELS, SpectralAnchor
from smagm.features.point_guided.spectral_query import (
    FeatureGridGeometry,
    SpectralPointQuery,
    derive_feature_grid_geometry,
)


def _backbone() -> MedicalNetResNet10:
    return MedicalNetResNet10(in_channels=3).eval()


def _anchor_from_coordinate_ramps(
    depth: int,
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> SpectralAnchor:
    channels = torch.arange(SPECTRAL_ANCHOR_CHANNELS, dtype=dtype).view(1, -1, 1, 1)
    d = torch.arange(depth, dtype=dtype).view(1, 1, depth, 1)
    h = torch.arange(height, dtype=dtype).view(1, 1, height, 1)
    h_yz = torch.arange(height, dtype=dtype).view(1, 1, 1, height)
    w = torch.arange(width, dtype=dtype).view(1, 1, 1, width)
    return SpectralAnchor(
        xy=channels + 100.0 * h + w,
        xz=channels + 100.0 * d + w,
        yz=channels + 100.0 * d + h_yz,
    )


def _triple(value: int | Sequence[int] | None, *, fallback: int | Sequence[int] | None = None) -> tuple[int, int, int]:
    raw = fallback if value is None else value
    if isinstance(raw, int):
        return (raw, raw, raw)
    return tuple(raw)  # type: ignore[arg-type,return-value]


def _conv_output_shape(shape_dhw: Sequence[int], module: torch.nn.Module) -> tuple[int, int, int]:
    kernel = _triple(module.kernel_size)  # type: ignore[attr-defined]
    stride_value = module.stride if module.stride is not None else module.kernel_size  # type: ignore[attr-defined]
    stride = _triple(stride_value)
    padding = _triple(module.padding)  # type: ignore[attr-defined]
    dilation = _triple(module.dilation)  # type: ignore[attr-defined]
    return tuple(
        (length + 2 * pad - dilation_axis * (kernel_axis - 1) - 1) // stride_axis + 1
        for length, kernel_axis, stride_axis, pad, dilation_axis in zip(shape_dhw, kernel, stride, padding, dilation)
    )


def _compose_actual_operator_centres(
    backbone: MedicalNetResNet10,
    *,
    tap: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Test-side composition reads the live modules, not a fixed scale."""

    scale = (1.0, 1.0, 1.0)
    offset = (0.0, 0.0, 0.0)

    def apply(module: torch.nn.Module) -> None:
        nonlocal scale, offset
        kernel = _triple(module.kernel_size)  # type: ignore[attr-defined]
        stride_value = module.stride if module.stride is not None else module.kernel_size  # type: ignore[attr-defined]
        stride = _triple(stride_value)
        padding = _triple(module.padding)  # type: ignore[attr-defined]
        dilation = _triple(module.dilation)  # type: ignore[attr-defined]
        local_offset = tuple(dilation_axis * (kernel_axis - 1) / 2.0 - pad for kernel_axis, pad, dilation_axis in zip(kernel, padding, dilation))
        scale, offset = (
            tuple(current * step for current, step in zip(scale, stride)),
            tuple(current_scale * local + current_offset for current_scale, local, current_offset in zip(scale, local_offset, offset)),
        )

    apply(backbone.conv1)
    if tap == "layer1":
        apply(backbone.maxpool)
        for block in backbone.layer1:
            apply(block.conv1)
            apply(block.conv2)
    return scale, offset


def _derived_geometry(
    *,
    source_geometry: VolumeGeometry | None = None,
    tap: str = "conv1_pre_maxpool",
) -> FeatureGridGeometry:
    source = source_geometry or VolumeGeometry.from_spacing((8, 10, 12), (1.25, 1.5, 2.0), (3.0, -2.0, 4.0))
    backbone = _backbone()
    observed = _conv_output_shape(source.shape_dhw, backbone.conv1)
    if tap == "layer1":
        observed = _conv_output_shape(observed, backbone.maxpool)
        for block in backbone.layer1:
            observed = _conv_output_shape(observed, block.conv1)
            observed = _conv_output_shape(observed, block.conv2)
    return derive_feature_grid_geometry(backbone, source, tap=tap, observed_shape_dhw=observed)


@pytest.mark.parametrize(
    ("tap", "source_shape_dhw"),
    (("conv1_pre_maxpool", (8, 10, 12)), ("layer1", (16, 20, 24))),
)
def test_derived_geometry_uses_live_medicalnet_metadata_and_validates_observed_shape(
    tap: str,
    source_shape_dhw: tuple[int, int, int],
) -> None:
    backbone = _backbone()
    source = VolumeGeometry.from_spacing(source_shape_dhw, (1.25, 1.5, 2.0), (3.0, -2.0, 4.0))
    observed = _conv_output_shape(source.shape_dhw, backbone.conv1)
    if tap == "layer1":
        observed = _conv_output_shape(observed, backbone.maxpool)
        for block in backbone.layer1:
            observed = _conv_output_shape(observed, block.conv1)
            observed = _conv_output_shape(observed, block.conv2)

    geometry = derive_feature_grid_geometry(backbone, source, tap=tap, observed_shape_dhw=observed)
    expected_scale, expected_offset = _compose_actual_operator_centres(backbone, tap=tap)
    assert geometry.shape_dhw == observed
    assert geometry.feature_to_source_scale_dhw == expected_scale
    assert geometry.feature_to_source_offset_dhw == expected_offset
    assert geometry.operator_chain[0] == "conv1"
    if tap == "layer1":
        assert "maxpool" in geometry.operator_chain
        assert any(name.startswith("layer1[") for name in geometry.operator_chain)

    bad_shape = (observed[0] + 1, observed[1], observed[2])
    with pytest.raises(ValueError, match="does not match observed feature"):
        derive_feature_grid_geometry(backbone, source, tap=tap, observed_shape_dhw=bad_shape)


@pytest.mark.parametrize("tap", ("conv1_pre_maxpool", "layer1"))
def test_derived_shape_agrees_with_actual_live_medicalnet_feature(tap: str) -> None:
    backbone = _backbone()
    source = VolumeGeometry.from_spacing((9, 11, 13), (1.0, 1.0, 1.0))
    with torch.no_grad():
        features = backbone.forward_intermediate_features(torch.randn(1, 3, *source.shape_dhw))
    observed = features.shallow.shape[-3:] if tap == "conv1_pre_maxpool" else features.layer1.shape[-3:]
    derived = derive_feature_grid_geometry(backbone, source, tap=tap, observed_shape_dhw=observed)
    assert derived.shape_dhw == tuple(observed)


@pytest.mark.parametrize(
    ("tap", "source_shape_dhw"),
    (("conv1_pre_maxpool", (16, 20, 24)), ("layer1", (16, 20, 24))),
)
def test_feature_index_centres_round_trip_through_source_ras_for_both_taps(
    tap: str,
    source_shape_dhw: tuple[int, int, int],
) -> None:
    """Prove the live operator chain's centre map, not only its output shape."""

    backbone = _backbone()
    source = VolumeGeometry.from_spacing(
        source_shape_dhw,
        (1.25, 1.5, 2.0),
        (3.0, -2.0, 4.0),
    )
    observed = _conv_output_shape(source.shape_dhw, backbone.conv1)
    if tap == "layer1":
        observed = _conv_output_shape(observed, backbone.maxpool)
        for block in backbone.layer1:
            observed = _conv_output_shape(observed, block.conv1)
            observed = _conv_output_shape(observed, block.conv2)
    grid = derive_feature_grid_geometry(backbone, source, tap=tap, observed_shape_dhw=observed)

    feature_dhw = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    scale, offset = _compose_actual_operator_centres(backbone, tap=tap)
    expected_source_dhw = (
        feature_dhw * torch.tensor(scale, dtype=torch.float64)
        + torch.tensor(offset, dtype=torch.float64)
    )
    expected_ras_mm = voxel_dhw_to_ras_mm(expected_source_dhw, source)

    actual_ras_mm = grid.feature_dhw_to_ras_mm(feature_dhw)
    recovered_feature_dhw = grid.ras_mm_to_feature_dhw(actual_ras_mm)
    torch.testing.assert_close(actual_ras_mm, expected_ras_mm, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(recovered_feature_dhw, feature_dhw, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "affine",
    (
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.1, 0.0, 0.0, 0.0), (0.0, 1.7, 0.0, 0.0), (0.0, 0.0, 2.3, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0, 8.0), (0.0, 1.0, 0.0, -3.0), (0.0, 0.0, 1.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, -1.5, 0.0, 0.0), (1.5, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.2, 0.25, -0.1, 0.0), (0.1, 1.7, 0.3, 0.0), (0.2, -0.15, 2.4, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, -1.3, 0.15, 4.0), (1.2, 0.2, 0.25, -3.0), (0.1, -0.1, 2.2, 7.0), (0.0, 0.0, 0.0, 1.0)),
    ),
)
def test_feature_grid_round_trips_identity_anisotropy_translation_rotation_and_shear(
    affine: tuple[tuple[float, ...], ...],
) -> None:
    source = VolumeGeometry((8, 10, 12), affine)
    grid = _derived_geometry(source_geometry=source)
    feature_dhw = torch.tensor([[[1.25, 2.5, 3.75]]], dtype=torch.float64)
    ras = grid.feature_dhw_to_ras_mm(feature_dhw)
    recovered = grid.ras_mm_to_feature_dhw(ras)
    torch.testing.assert_close(recovered, feature_dhw, rtol=1e-11, atol=1e-11)

    scale = torch.tensor(grid.feature_to_source_scale_dhw, dtype=torch.float64)
    offset = torch.tensor(grid.feature_to_source_offset_dhw, dtype=torch.float64)
    expected_source_dhw = feature_dhw * scale + offset
    expected_ras = voxel_dhw_to_ras_mm(expected_source_dhw, source)
    torch.testing.assert_close(ras, expected_ras, rtol=1e-11, atol=1e-11)


def test_exact_feature_centres_query_exact_xy_xz_yz_anchor_pixels() -> None:
    grid = _derived_geometry()
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width)
    feature_dhw = torch.tensor([[[1.0, 2.0, 3.0]]])
    points_ras_mm = grid.feature_dhw_to_ras_mm(feature_dhw)

    samples = SpectralPointQuery()(anchor, points_ras_mm, grid)
    torch.testing.assert_close(samples.xy[0, 0], anchor.xy[0, :, 2, 3], rtol=0.0, atol=1e-4)
    torch.testing.assert_close(samples.xz[0, 0], anchor.xz[0, :, 1, 3], rtol=0.0, atol=1e-4)
    torch.testing.assert_close(samples.yz[0, 0], anchor.yz[0, :, 1, 2], rtol=0.0, atol=1e-4)


def test_fractional_query_matches_manual_bilinear_interpolation() -> None:
    grid = _derived_geometry()
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width)
    feature_dhw = torch.tensor([[[1.25, 1.25, 2.5]]])
    points_ras_mm = grid.feature_dhw_to_ras_mm(feature_dhw)
    samples = SpectralPointQuery()(anchor, points_ras_mm, grid)

    def manual(plane: torch.Tensor, row: float, column: float) -> torch.Tensor:
        row_low, column_low = int(row), int(column)
        row_high, column_high = row_low + 1, column_low + 1
        row_fraction, column_fraction = row - row_low, column - column_low
        return (
            (1.0 - row_fraction) * (1.0 - column_fraction) * plane[0, :, row_low, column_low]
            + (1.0 - row_fraction) * column_fraction * plane[0, :, row_low, column_high]
            + row_fraction * (1.0 - column_fraction) * plane[0, :, row_high, column_low]
            + row_fraction * column_fraction * plane[0, :, row_high, column_high]
        )

    torch.testing.assert_close(samples.xy[0, 0], manual(anchor.xy, 1.25, 2.5), rtol=0.0, atol=1e-4)
    torch.testing.assert_close(samples.xz[0, 0], manual(anchor.xz, 1.25, 2.5), rtol=0.0, atol=1e-4)
    torch.testing.assert_close(samples.yz[0, 0], manual(anchor.yz, 1.25, 1.25), rtol=0.0, atol=1e-4)


def test_plane_axis_mapping_uses_feature_dhw_as_xy_xz_yz() -> None:
    grid = _derived_geometry()
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width)
    feature_dhw = torch.tensor(
        [[[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [1.0, 3.0, 3.0], [2.0, 2.0, 3.0]]]
    )
    samples = SpectralPointQuery()(anchor, grid.feature_dhw_to_ras_mm(feature_dhw), grid)
    base_xy, base_xz, base_yz = samples.xy[:, 0], samples.xz[:, 0], samples.yz[:, 0]

    for actual, expected in (
        (samples.xy[:, 1] - base_xy, torch.ones_like(base_xy)),
        (samples.xz[:, 1] - base_xz, torch.ones_like(base_xz)),
        (samples.yz[:, 1] - base_yz, torch.zeros_like(base_yz)),
        (samples.xy[:, 2] - base_xy, torch.full_like(base_xy, 100.0)),
        (samples.xz[:, 2] - base_xz, torch.zeros_like(base_xz)),
        (samples.yz[:, 2] - base_yz, torch.ones_like(base_yz)),
        (samples.xy[:, 3] - base_xy, torch.zeros_like(base_xy)),
        (samples.xz[:, 3] - base_xz, torch.full_like(base_xz, 100.0)),
        (samples.yz[:, 3] - base_yz, torch.full_like(base_yz, 100.0)),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-4)


def test_rotated_sheared_affine_query_hits_known_feature_pixel() -> None:
    source = VolumeGeometry(
        (8, 10, 12),
        ((0.0, -1.3, 0.15, 4.0), (1.2, 0.2, 0.25, -3.0), (0.1, -0.1, 2.2, 7.0), (0.0, 0.0, 0.0, 1.0)),
    )
    grid = _derived_geometry(source_geometry=source)
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width)
    feature_dhw = torch.tensor([[[1.0, 2.0, 3.0]]])
    samples = SpectralPointQuery()(anchor, grid.feature_dhw_to_ras_mm(feature_dhw), grid)
    torch.testing.assert_close(samples.xy[0, 0], anchor.xy[0, :, 2, 3], rtol=0.0, atol=1e-4)
    torch.testing.assert_close(samples.xz[0, 0], anchor.xz[0, :, 1, 3], rtol=0.0, atol=1e-4)
    torch.testing.assert_close(samples.yz[0, 0], anchor.yz[0, :, 1, 2], rtol=0.0, atol=1e-4)


def test_query_preserves_nonzero_finite_coordinate_gradients() -> None:
    grid = _derived_geometry()
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width)
    feature_dhw = torch.tensor([[[1.25, 1.5, 2.25]]])
    points_ras_mm = grid.feature_dhw_to_ras_mm(feature_dhw).detach().requires_grad_(True)
    samples = SpectralPointQuery()(anchor, points_ras_mm, grid)
    (samples.xy.square().mean() + samples.xz.square().mean() + samples.yz.square().mean()).backward()
    assert points_ras_mm.grad is not None
    assert bool(torch.isfinite(points_ras_mm.grad).all())
    assert bool(points_ras_mm.grad.abs().sum() > 0.0)


@pytest.mark.parametrize("point_count", (2048, 3072))
def test_query_uses_only_pointwise_2d_grids_for_large_point_counts(
    monkeypatch: pytest.MonkeyPatch,
    point_count: int,
) -> None:
    import smagm.features.point_guided.spectral_query as spectral_query_module

    grid = _derived_geometry()
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width)
    feature_dhw = torch.empty(1, point_count, 3).uniform_(0.25, 0.75)
    feature_dhw[..., 0] *= depth - 1
    feature_dhw[..., 1] *= height - 1
    feature_dhw[..., 2] *= width - 1
    points_ras_mm = grid.feature_dhw_to_ras_mm(feature_dhw)
    grids: list[tuple[int, ...]] = []
    original_grid_sample = spectral_query_module.F.grid_sample

    def record_grid(*args: object, **kwargs: object) -> torch.Tensor:
        grids.append(tuple(args[1].shape))  # type: ignore[union-attr,index]
        return original_grid_sample(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(spectral_query_module.F, "grid_sample", record_grid)
    samples = SpectralPointQuery()(anchor, points_ras_mm, grid)
    assert samples.xy.shape == samples.xz.shape == samples.yz.shape == (1, point_count, SPECTRAL_ANCHOR_CHANNELS)
    assert grids == [(1, point_count, 1, 2)] * 3
    evidence = CrossPlaneConsistency()(samples.xy, samples.xz, samples.yz)
    assert evidence.f_spec.shape == (1, point_count, 3 * SPECTRAL_ANCHOR_CHANNELS)


def test_query_is_parameter_free_and_preserves_float64_dtype() -> None:
    grid = _derived_geometry()
    depth, height, width = grid.shape_dhw
    anchor = _anchor_from_coordinate_ramps(depth, height, width, dtype=torch.float64)
    points_ras_mm = grid.feature_dhw_to_ras_mm(torch.tensor([[[1.25, 1.5, 2.25]]], dtype=torch.float64))
    module = SpectralPointQuery()
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    samples = module(anchor, points_ras_mm, grid)
    assert samples.xy.dtype == samples.xz.dtype == samples.yz.dtype == torch.float64
