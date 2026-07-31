"""Blocking half-pixel feature-grid geometry contracts for T1-A."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from smagm.baselines.fixed_support import FixedSupportConfig, sample_fixed_supports
from smagm.contracts.coordinates import PhysicalPlane, SourceAffineTransform, SourceConvention
from smagm.features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform


def _plane(shape_hw: tuple[int, int], **changes: object) -> PhysicalPlane:
    fields: dict[str, object] = {
        "pixel_center_origin_ras_mm": (10.0, 20.0, 30.0),
        "axis_u_ras": (0.0, 1.0, 0.0),
        "axis_v_ras": (-1.0, 0.0, 0.0),
        "spacing_uv_mm": (2.0, 3.0),
        "thickness_mm": 4.0,
        "shape_hw": shape_hw,
        "signed_normal_ras": (0.0, 0.0, 1.0),
        "observation_id": "feature-plane",
    }
    fields.update(changes)
    return PhysicalPlane(**fields)  # type: ignore[arg-type]


def _source_affine_plane(shape_hw: tuple[int, int]) -> PhysicalPlane:
    return _plane(
        shape_hw,
        source_transform=SourceAffineTransform(
            ((0.0, -3.0, 0.0, 10.0), (2.0, 0.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0), (0.0, 0.0, 0.0, 1.0)),
            SourceConvention.CANONICAL_RAS,
        ),
    )


def _foreign_same_shape_planes(shape_hw: tuple[int, int]) -> tuple[PhysicalPlane, ...]:
    return (
        _plane(shape_hw, pixel_center_origin_ras_mm=(11.0, 20.0, 30.0)),
        _plane(shape_hw, axis_u_ras=(1.0, 0.0, 0.0), axis_v_ras=(0.0, 1.0, 0.0)),
        _source_affine_plane(shape_hw),
        _plane(shape_hw, observation_id="spoofed-observation"),
    )


def _bound_transform(plane: PhysicalPlane) -> FeatureGridToPlaneTransform:
    return FeatureGridToPlaneTransform(plane.shape_hw, (4, 6), (2, 2), input_plane=plane)


def _call_public_mapping(transform: FeatureGridToPlaneTransform, method: str, plane: PhysicalPlane) -> torch.Tensor | tuple[float, float, float]:
    if method == "ras_mm_from_feature_vu":
        return transform.ras_mm_from_feature_vu(1.0, 2.0, plane=plane)
    if method == "world_from_feature_vu":
        return transform.world_from_feature_vu(plane, 1.0, 2.0)
    if method == "grid_sample_coordinates":
        point = transform.ras_mm_from_feature_vu(1.0, 2.0, plane=plane)
        return transform.grid_sample_coordinates(point, plane=plane)
    raise AssertionError(f"unexpected method: {method}")


@pytest.mark.parametrize("method", ("ras_mm_from_feature_vu", "world_from_feature_vu", "grid_sample_coordinates"))
@pytest.mark.parametrize("foreign_plane", _foreign_same_shape_planes((8, 12)))
def test_bound_public_mappings_reject_all_foreign_same_shape_planes(method: str, foreign_plane: PhysicalPlane) -> None:
    transform = _bound_transform(_plane((8, 12)))
    with pytest.raises(ValueError, match="canonically match"):
        _call_public_mapping(transform, method, foreign_plane)


def test_public_mappings_accept_canonical_equivalent_plane_and_unbound_explicit_plane() -> None:
    bound_plane = _plane((8, 12))
    equivalent_plane = _plane((8, 12))
    bound = _bound_transform(bound_plane)
    unbound = FeatureGridToPlaneTransform((8, 12), (4, 6), (2, 2))

    for method in ("ras_mm_from_feature_vu", "world_from_feature_vu", "grid_sample_coordinates"):
        expected = _call_public_mapping(bound, method, bound_plane)
        equivalent = _call_public_mapping(bound, method, equivalent_plane)
        explicit_unbound = _call_public_mapping(unbound, method, bound_plane)
        if isinstance(expected, tuple):
            assert equivalent == expected
            assert explicit_unbound == expected
        else:
            assert torch.allclose(equivalent, expected)
            assert torch.allclose(explicit_unbound, expected)


@pytest.mark.parametrize(
    "stride,feature_shape,feature_vu,input_vu,expected_ras",
    [
        (2, (4, 6), (1.0, 2.0), (2.5, 4.5), (2.5, 29.0, 30.0)),
        (4, (2, 3), (1.0, 2.0), (5.5, 9.5), (-6.5, 39.0, 30.0)),
    ],
)
def test_half_pixel_ras_inverse_and_grid_sample_coordinates(
    stride: int,
    feature_shape: tuple[int, int],
    feature_vu: tuple[float, float],
    input_vu: tuple[float, float],
    expected_ras: tuple[float, float, float],
) -> None:
    plane = _plane((8, 12))
    transform = FeatureGridToPlaneTransform(
        input_shape_hw=plane.shape_hw,
        feature_shape_hw=feature_shape,
        stride_vu=(stride, stride),
        input_plane=plane,
    )
    assert transform.offset_vu_input_pixels == ((stride - 1.0) / 2.0,) * 2
    assert transform.input_vu_from_feature_vu(*feature_vu) == input_vu
    assert transform.feature_vu_from_input_vu(*input_vu) == feature_vu

    ras = transform.ras_mm_from_feature_vu(*feature_vu)
    assert torch.allclose(ras, torch.tensor(expected_ras, dtype=torch.float64))
    grid = transform.grid_sample_coordinates(ras)
    expected_grid = torch.tensor(
        [
            2.0 * (feature_vu[1] + 0.5) / feature_shape[1] - 1.0,
            2.0 * (feature_vu[0] + 0.5) / feature_shape[0] - 1.0,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(grid, expected_grid, atol=1e-12)

    ramp = (10.0 * torch.arange(feature_shape[0], dtype=torch.float64)[:, None]
            + torch.arange(feature_shape[1], dtype=torch.float64)[None, :]).reshape(1, 1, *feature_shape)
    sampled = F.grid_sample(ramp, grid.reshape(1, 1, 1, 2), align_corners=False)
    assert torch.allclose(sampled[0, 0, 0, 0], torch.tensor(10.0 * feature_vu[0] + feature_vu[1], dtype=torch.float64))


def test_odd_shape_padding_is_explicit_and_never_becomes_support() -> None:
    plane = _plane((5, 7))
    transform = FeatureGridToPlaneTransform((5, 7), (3, 4), (2, 2), input_plane=plane)
    assert transform.valid_feature_shape_hw == (2, 3)
    assert torch.equal(
        transform.valid_feature_mask(),
        torch.tensor([[True, True, True, False], [True, True, True, False], [False, False, False, False]]),
    )
    valid_feature_mask = transform.valid_feature_mask().reshape(1, 1, 3, 4).clone()
    valid_feature_mask[0, 0, 0, 1] = False
    features = EncoderFeatureMaps(
        structural=torch.zeros((1, 1, 3, 4), dtype=torch.float64),
        appearance=torch.zeros((1, 1, 3, 4), dtype=torch.float64),
        reliability=torch.ones((1, 1, 3, 4), dtype=torch.float64),
        grid_to_plane=transform,
        valid_feature_mask=valid_feature_mask,
    )
    supports = sample_fixed_supports(features, plane, config=FixedSupportConfig(step_vu=(1, 1), max_points=4))
    assert supports.feature_indices_vu.tolist() == [[0, 0], [0, 2], [1, 0], [1, 1]]

    with pytest.raises(ValueError, match="padded"):
        EncoderFeatureMaps(
            structural=torch.zeros((1, 1, 3, 4), dtype=torch.float64),
            appearance=torch.zeros((1, 1, 3, 4), dtype=torch.float64),
            reliability=torch.ones((1, 1, 3, 4), dtype=torch.float64),
            grid_to_plane=transform,
            valid_feature_mask=torch.ones((1, 1, 3, 4), dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stride_vu": (2, 2), "offset_vu_input_pixels": (0.0, 0.0)},
        {"stride_vu": (3, 3)},
        {"stride_vu": (2, 4)},
        {"stride_vu": (4, 4), "feature_shape_hw": (3, 3)},
    ],
)
def test_feature_grid_rejects_incompatible_stride_offset_or_shape(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "input_shape_hw": (8, 8),
        "feature_shape_hw": (4, 4),
        "stride_vu": (2, 2),
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        FeatureGridToPlaneTransform(**values)  # type: ignore[arg-type]
