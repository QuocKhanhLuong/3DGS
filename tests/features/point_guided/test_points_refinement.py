"""CPU synthetic checks for the locked point placement and refinement path."""

from __future__ import annotations

import inspect

import pytest
import torch

from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.contracts import PointGuidedGeometryError, VolumeGeometry
from smagm.features.point_guided.directional import (
    build_directional_descriptor,
    directional_locations_ras_mm,
)
from smagm.features.point_guided.points import DeterministicPointInitializer, initialize_quasi_uniform_points
from smagm.features.point_guided.refinement import PointRefiner, refine_points_ras_mm
from smagm.features.point_guided.sampling import (
    ras_mm_in_bounds,
    ras_mm_to_voxel_dhw,
    sample_volume_ras_mm,
    voxel_dhw_to_ras_mm,
)


def _geometry() -> VolumeGeometry:
    return VolumeGeometry.from_spacing(
        shape_dhw=(7, 8, 9),
        spacing_xyz_mm=(2.0, 3.0, 4.0),
        origin_ras_mm=(-10.0, 5.0, 20.0),
    )


def _config(*, num_points: int = 8) -> PointGuidedConfig:
    return PointGuidedConfig(
        num_semantic_classes=3,
        num_points=num_points,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
    )


def test_ras_sampler_is_identity_at_voxel_centres_and_reports_bounds() -> None:
    geometry = _geometry()
    depth, height, width = geometry.shape_dhw
    values = torch.arange(depth * height * width, dtype=torch.float32).reshape(1, 1, depth, height, width)
    d, h, w = torch.meshgrid(
        torch.arange(depth, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    voxel_dhw = torch.stack((d.reshape(-1), h.reshape(-1), w.reshape(-1)), dim=-1).unsqueeze(0)
    ras_mm = voxel_dhw_to_ras_mm(voxel_dhw, geometry)

    sampled = sample_volume_ras_mm(values, ras_mm, geometry, require_in_bounds=True)

    torch.testing.assert_close(sampled[..., 0], values.reshape(1, -1), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(ras_mm_to_voxel_dhw(ras_mm, geometry), voxel_dhw, atol=1e-5, rtol=0.0)
    assert bool(ras_mm_in_bounds(ras_mm, geometry).all())
    outside = ras_mm[:, :1] + torch.tensor([100.0, 0.0, 0.0])
    assert not bool(ras_mm_in_bounds(outside, geometry).all())
    with pytest.raises(PointGuidedGeometryError, match="inside the volume"):
        sample_volume_ras_mm(values, outside, geometry, require_in_bounds=True)


def test_ras_sampler_round_trips_a_rotated_sheared_affine_without_axis_guessing() -> None:
    geometry = VolumeGeometry(
        (4, 5, 6),
        (
            (0.0, -2.0, 0.25, 7.0),
            (1.5, 0.0, 0.15, -3.0),
            (0.0, 0.2, 3.0, 11.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    voxel_dhw = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]])
    ras_mm = voxel_dhw_to_ras_mm(voxel_dhw, geometry)
    values = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(1, 1, 4, 5, 6)

    torch.testing.assert_close(ras_mm_to_voxel_dhw(ras_mm, geometry), voxel_dhw, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        sample_volume_ras_mm(values, ras_mm, geometry, require_in_bounds=True)[..., 0],
        torch.tensor([[0.0, 82.0, 119.0]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_directional_locations_are_exact_ras_mm_offsets_under_anisotropic_spacing() -> None:
    geometry = _geometry()
    centre_voxel = torch.tensor([[[3.0, 4.0, 5.0]]])
    centre_ras = voxel_dhw_to_ras_mm(centre_voxel, geometry)
    locations = directional_locations_ras_mm(centre_ras)

    torch.testing.assert_close(locations[0, 0, 1] - centre_ras[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(locations[0, 0, 2] - centre_ras[0, 0], torch.tensor([-1.0, 0.0, 0.0]))
    torch.testing.assert_close(locations[0, 0, 7] - centre_ras[0, 0], torch.tensor([2.0, 0.0, 0.0]))
    # The x-axis is W and has 2-mm spacing: +2 RAS mm advances one W voxel,
    # while the physical descriptor remains exactly a 2-mm RAS displacement.
    shifted_voxel = ras_mm_to_voxel_dhw(locations[:, :, 7], geometry)
    torch.testing.assert_close(shifted_voxel - centre_voxel, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5, rtol=0.0)


def test_directional_descriptor_is_centre_relative_and_uses_only_three_input_modalities() -> None:
    geometry = VolumeGeometry.from_spacing((7, 7, 7))
    d, h, w = torch.meshgrid(
        torch.arange(7, dtype=torch.float32),
        torch.arange(7, dtype=torch.float32),
        torch.arange(7, dtype=torch.float32),
        indexing="ij",
    )
    mri = torch.stack((w, h, d), dim=0).unsqueeze(0)
    semantic = torch.zeros(1, 3, 7, 7, 7)
    semantic[:, 0] = 1.0
    centre = voxel_dhw_to_ras_mm(torch.tensor([[[3.0, 3.0, 3.0]]]), geometry)

    descriptor = build_directional_descriptor(mri, semantic, centre, geometry)

    channels = 3 + 3
    assert descriptor.shape == (1, 1, 19 * channels)
    torch.testing.assert_close(descriptor[0, 0, :channels], torch.tensor([1.0, 0.0, 0.0, 3.0, 3.0, 3.0]))
    # First directional group is +RAS-x. Its semantic differences are zero;
    # only the T1-like W ramp changes by +1 mm at unit spacing.
    torch.testing.assert_close(
        descriptor[0, 0, channels : 2 * channels],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_initializer_is_deterministic_value_independent_and_not_a_cartesian_grid() -> None:
    geometry = _geometry()
    initializer = DeterministicPointInitializer(_config(num_points=11))

    first = initializer(geometry, batch_size=2, dtype=torch.float32)
    unrelated_mri_values = torch.randn(2, 3, *geometry.shape_dhw)
    unrelated_mri_values.mul_(17.0).add_(3.0)
    second = initializer(geometry, batch_size=2, dtype=torch.float32)

    assert bool(torch.isfinite(unrelated_mri_values).all())
    torch.testing.assert_close(first, second)
    assert "volume" not in inspect.signature(initialize_quasi_uniform_points).parameters
    assert first.shape == (2, 11, 3)
    assert bool(ras_mm_in_bounds(first, geometry).all())
    voxel_dhw = ras_mm_to_voxel_dhw(first, geometry)
    # A Halton candidate sequence has continuous coordinates; a rigid Cartesian
    # voxel grid would make every coordinate integral.
    assert bool((voxel_dhw - voxel_dhw.round()).abs().gt(1e-4).any())


def test_initializer_returns_exact_mask_valid_count_or_fails_clearly() -> None:
    geometry = _geometry()
    mask = torch.zeros(2, 1, *geometry.shape_dhw, dtype=torch.bool)
    mask[0, 0, 1:5, 2:6, 2:7] = True
    mask[1, 0, 2:6, 1:6, 1:5] = True
    points = DeterministicPointInitializer(_config(num_points=9))(
        geometry,
        batch_size=2,
        brain_mask=mask,
        dtype=torch.float32,
    )

    voxel_dhw = ras_mm_to_voxel_dhw(points, geometry).round().to(dtype=torch.long)
    batch = torch.arange(2).view(2, 1).expand(2, 9)
    assert points.shape == (2, 9, 3)
    assert bool(mask[:, 0][batch, voxel_dhw[..., 0], voxel_dhw[..., 1], voxel_dhw[..., 2]].all())
    # Mask-constrained candidates retain a deterministic sub-voxel offset;
    # the optional mask must not collapse the placement to a Cartesian grid.
    continuous_voxel_dhw = ras_mm_to_voxel_dhw(points, geometry)
    assert bool((continuous_voxel_dhw - continuous_voxel_dhw.round()).abs().gt(1e-4).any())
    too_small = torch.zeros(1, *geometry.shape_dhw, dtype=torch.bool)
    too_small[0, 0, 0, :3] = True
    with pytest.raises(PointGuidedGeometryError, match="fewer than requested"):
        initialize_quasi_uniform_points(geometry, 1, 4, brain_mask=too_small)


def test_corner_only_mask_keeps_masked_candidates_subvoxel_and_legal() -> None:
    geometry = VolumeGeometry.from_spacing((3, 3, 3))
    mask = torch.zeros(1, 3, 3, 3, dtype=torch.bool)
    mask[0, 0, 0, 0] = True

    points = initialize_quasi_uniform_points(
        geometry,
        batch_size=1,
        num_points=1,
        brain_mask=mask,
        candidate_multiplier=1,
    )
    voxel_dhw = ras_mm_to_voxel_dhw(points, geometry)
    assert torch.equal(voxel_dhw.round().to(dtype=torch.long), torch.zeros((1, 1, 3), dtype=torch.long))
    # The origin's would-be outward Halton offsets must be reflected inward,
    # rather than clipped to the integral corner voxel centre.
    assert bool((voxel_dhw - voxel_dhw.round()).abs().gt(1e-4).all())


def test_refinement_is_original_relative_bounded_and_valid() -> None:
    geometry = _geometry()
    original_voxel = torch.tensor([[[0.1, 0.2, 0.3], [6.0, 7.0, 8.0]]])
    original = voxel_dhw_to_ras_mm(original_voxel, geometry)
    raw = torch.tensor([[[50.0, -30.0, -40.0], [-80.0, 30.0, 20.0]]])

    refined, displacement = refine_points_ras_mm(original, raw, geometry, max_displacement_mm=2.0)

    torch.testing.assert_close(refined - original, displacement, atol=1e-6, rtol=0.0)
    assert bool((torch.linalg.vector_norm(displacement, dim=-1) <= 2.0 + 1e-6).all())
    assert bool(ras_mm_in_bounds(refined, geometry).all())


def test_refiner_keeps_gradient_to_the_offset_mlp() -> None:
    torch.manual_seed(7)
    geometry = _geometry()
    config = _config(num_points=2)
    refiner = PointRefiner(config)
    mri = torch.randn(1, 3, *geometry.shape_dhw)
    semantic = torch.softmax(torch.randn(1, config.num_semantic_classes, *geometry.shape_dhw), dim=1)
    original = voxel_dhw_to_ras_mm(torch.tensor([[[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]]]), geometry)

    field = refiner(mri, semantic, original, geometry)
    loss = field.refined_centers_ras_mm.square().sum() + field.semantic_vectors.square().sum()
    loss.backward()

    assert bool((torch.linalg.vector_norm(field.displacement_ras_mm, dim=-1) <= config.max_displacement_mm + 1e-6).all())
    assert bool(ras_mm_in_bounds(field.refined_centers_ras_mm, geometry).all())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0.0)
        for parameter in refiner.offset_predictor.parameters()
    )
