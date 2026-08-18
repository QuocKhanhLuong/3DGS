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
from smagm.features.point_guided.offset_predictor import OffsetPredictor
from smagm.features.point_guided.points import DeterministicPointInitializer, initialize_quasi_uniform_points
from smagm.features.point_guided.refinement import (
    PointRefiner,
    bound_displacement_ras_mm,
    project_displacement_to_validity,
    refine_points_ras_mm,
)
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


def _rotated_sheared_geometry() -> VolumeGeometry:
    return VolumeGeometry(
        (7, 8, 9),
        (
            (0.0, -1.0, 0.5, 7.0),
            (1.0, 0.0, 0.0, -3.0),
            (0.0, 0.0, 1.0, 11.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _previous_non_singular_projection_formula(
    original_centres_ras_mm: torch.Tensor,
    bounded_displacement_ras_mm: torch.Tensor,
    geometry: VolumeGeometry,
) -> torch.Tensor:
    """Reference the pre-fix formula only where every denominator is nonzero."""

    original_voxel_dhw = ras_mm_to_voxel_dhw(original_centres_ras_mm, geometry)
    candidate_voxel_dhw = ras_mm_to_voxel_dhw(original_centres_ras_mm + bounded_displacement_ras_mm, geometry)
    delta_voxel_dhw = candidate_voxel_dhw - original_voxel_dhw
    assert bool((delta_voxel_dhw != 0.0).all())
    upper = torch.as_tensor(
        tuple(length - 1 for length in geometry.shape_dhw),
        dtype=original_voxel_dhw.dtype,
        device=original_voxel_dhw.device,
    )
    infinity = torch.full_like(delta_voxel_dhw, float("inf"))
    upper_limit = (upper - original_voxel_dhw) / delta_voxel_dhw
    lower_limit = -original_voxel_dhw / delta_voxel_dhw
    limits = torch.where(
        delta_voxel_dhw > 0.0,
        upper_limit,
        torch.where(delta_voxel_dhw < 0.0, lower_limit, infinity),
    )
    fraction = torch.clamp(limits.amin(dim=-1, keepdim=True), min=0.0, max=1.0)
    return bounded_displacement_ras_mm * fraction


def _ras_displacement_for_voxel_delta(
    original_centres_ras_mm: torch.Tensor,
    voxel_delta_dhw: torch.Tensor,
    geometry: VolumeGeometry,
) -> torch.Tensor:
    original_voxel_dhw = ras_mm_to_voxel_dhw(original_centres_ras_mm, geometry)
    return voxel_dhw_to_ras_mm(original_voxel_dhw + voxel_delta_dhw, geometry) - original_centres_ras_mm


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP geometry regression")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_voxel_to_ras_preserves_coordinate_dtype_under_cuda_amp(dtype: torch.dtype) -> None:
    geometry = _geometry()
    voxel_dhw = torch.tensor([[[1.25, 2.5, 3.75]]], dtype=dtype, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        ras_mm = voxel_dhw_to_ras_mm(voxel_dhw, geometry)

    assert ras_mm.dtype == dtype
    assert ras_mm.device == voxel_dhw.device
    assert bool(torch.isfinite(ras_mm).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP geometry regression")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_ras_to_voxel_preserves_coordinate_dtype_under_cuda_amp(dtype: torch.dtype) -> None:
    geometry = _geometry()
    voxel_dhw = torch.tensor([[[1.25, 2.5, 3.75]]], dtype=dtype, device="cuda")
    ras_mm = voxel_dhw_to_ras_mm(voxel_dhw, geometry)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        recovered = ras_mm_to_voxel_dhw(ras_mm, geometry)

    assert recovered.dtype == dtype
    assert recovered.device == ras_mm.device
    assert bool(torch.isfinite(recovered).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP geometry regression")
def test_affine_coordinate_round_trip_remains_float32_and_consistent_under_cuda_amp() -> None:
    geometry = _geometry()
    voxel_dhw = torch.tensor([[[1.25, 2.5, 3.75], [4.5, 5.25, 6.0]]], dtype=torch.float32, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        ras_mm = voxel_dhw_to_ras_mm(voxel_dhw, geometry)
        recovered = ras_mm_to_voxel_dhw(ras_mm, geometry)

    assert ras_mm.dtype == torch.float32
    assert recovered.dtype == torch.float32
    torch.testing.assert_close(recovered, voxel_dhw, atol=1e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP geometry regression")
def test_initializer_preserves_float32_points_under_cuda_amp() -> None:
    geometry = _geometry()
    initializer = DeterministicPointInitializer(_config(num_points=11)).cuda()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        points = initializer(geometry, batch_size=1, device="cuda", dtype=torch.float32)

    assert points.dtype == torch.float32
    assert points.device.type == "cuda"
    assert bool(ras_mm_in_bounds(points, geometry).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP geometry regression")
def test_directional_descriptor_accepts_float32_geometry_points_under_cuda_amp() -> None:
    geometry = _geometry()
    initializer = DeterministicPointInitializer(_config(num_points=2)).cuda()
    mri = torch.randn(1, 3, *geometry.shape_dhw, dtype=torch.float32, device="cuda")
    semantic = torch.softmax(
        torch.randn(1, 3, *geometry.shape_dhw, dtype=torch.float32, device="cuda"),
        dim=1,
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        points = initializer(geometry, batch_size=1, device="cuda", dtype=torch.float32)
        descriptor = build_directional_descriptor(mri, semantic, points, geometry)

    assert points.dtype == mri.dtype
    assert descriptor.shape == (1, 2, 19 * 6)
    assert bool(torch.isfinite(descriptor).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP predictor regression")
def test_offset_predictor_restores_float32_ras_displacement_under_cuda_amp_with_gradients() -> None:
    config = _config(num_points=2)
    predictor = OffsetPredictor.from_config(config).cuda().train()
    descriptor = torch.randn(
        1,
        2,
        predictor.descriptor_channels,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        displacement = predictor(descriptor)

    assert displacement.dtype == descriptor.dtype
    assert displacement.device == descriptor.device
    assert displacement.requires_grad
    assert bool(torch.isfinite(displacement).all())
    displacement.square().sum().backward()
    for layer in (predictor.network[0], predictor.network[-1]):
        assert isinstance(layer, torch.nn.Linear)
        assert layer.weight.grad is not None
        assert bool(torch.isfinite(layer.weight.grad).all())
        assert bool(layer.weight.grad.abs().sum() > 0.0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_offset_predictor_preserves_descriptor_dtype_without_amp(dtype: torch.dtype) -> None:
    config = _config(num_points=2)
    predictor = OffsetPredictor.from_config(config).to(dtype=dtype)
    descriptor = torch.randn(1, 2, predictor.descriptor_channels, dtype=dtype, requires_grad=True)

    displacement = predictor(descriptor)

    assert displacement.dtype == dtype
    assert displacement.device == descriptor.device
    assert displacement.requires_grad
    displacement.square().sum().backward()
    assert predictor.network[0].weight.grad is not None
    assert predictor.network[-1].weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the AMP refinement regression")
def test_refinement_accepts_float32_predictor_displacement_under_cuda_amp() -> None:
    geometry = _geometry()
    config = _config(num_points=2)
    refiner = PointRefiner(config).cuda().eval()
    mri = torch.randn(1, 3, *geometry.shape_dhw, dtype=torch.float32, device="cuda")
    semantic = torch.softmax(
        torch.randn(1, 3, *geometry.shape_dhw, dtype=torch.float32, device="cuda"),
        dim=1,
    )
    original = voxel_dhw_to_ras_mm(
        torch.tensor([[[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]]], dtype=torch.float32, device="cuda"),
        geometry,
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        descriptor = refiner.directional_descriptor(mri, semantic, original, geometry)
        raw_displacement = refiner.offset_predictor(descriptor)
        refined, displacement = refine_points_ras_mm(
            original,
            raw_displacement,
            geometry,
            config.max_displacement_mm,
        )

    assert descriptor.dtype == torch.float32
    assert raw_displacement.dtype == torch.float32
    assert refined.dtype == torch.float32
    assert displacement.dtype == torch.float32
    assert bool((torch.linalg.vector_norm(displacement, dim=-1) <= 2.0 + 1e-6).all())
    assert bool(ras_mm_in_bounds(refined, geometry).all())


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


def test_safe_refinement_preserves_zero_axes_boundaries_and_affine_geometry() -> None:
    geometry = _geometry()
    original = voxel_dhw_to_ras_mm(torch.tensor([[[3.0, 4.0, 5.0]]]), geometry)

    # A. An exact zero vector stays exact after radial and validity projection.
    zero = torch.zeros_like(original)
    torch.testing.assert_close(bound_displacement_ras_mm(zero, 2.0), zero, atol=0.0, rtol=0.0)
    torch.testing.assert_close(project_displacement_to_validity(original, zero, geometry), zero, atol=0.0, rtol=0.0)

    # B/C. Exact zero voxel axes impose no artificial constraint.
    for voxel_delta_dhw in (
        torch.tensor([[[0.5, 0.0, 0.25]]]),
        torch.tensor([[[0.0, 0.0, 0.5]]]),
    ):
        displacement = _ras_displacement_for_voxel_delta(original, voxel_delta_dhw, geometry)
        recovered_delta = ras_mm_to_voxel_dhw(original + displacement, geometry) - ras_mm_to_voxel_dhw(original, geometry)
        assert bool((recovered_delta == 0.0).any())
        torch.testing.assert_close(project_displacement_to_validity(original, displacement, geometry), displacement, atol=1e-6, rtol=0.0)

    # D/E. Positive and negative directions are line-projected at the boundary.
    for voxel, delta in (
        (torch.tensor([[[5.8, 6.8, 7.8]]]), torch.tensor([[[0.8, 0.8, 0.8]]])),
        (torch.tensor([[[0.2, 0.2, 0.2]]]), torch.tensor([[[-0.8, -0.8, -0.8]]])),
    ):
        boundary_original = voxel_dhw_to_ras_mm(voxel, geometry)
        displacement = _ras_displacement_for_voxel_delta(boundary_original, delta, geometry)
        projected = project_displacement_to_validity(boundary_original, displacement, geometry)
        assert bool(torch.linalg.vector_norm(projected, dim=-1) < torch.linalg.vector_norm(displacement, dim=-1))
        assert bool(ras_mm_in_bounds(boundary_original + projected, geometry).all())

    # F. An interior direction remains unprojected.
    interior = _ras_displacement_for_voxel_delta(original, torch.tensor([[[0.25, -0.25, 0.25]]]), geometry)
    torch.testing.assert_close(project_displacement_to_validity(original, interior, geometry), interior, atol=1e-6, rtol=0.0)

    # G. The same invariant holds for a rotated/sheared physical affine.
    rotated = _rotated_sheared_geometry()
    rotated_original = voxel_dhw_to_ras_mm(torch.tensor([[[3.0, 4.0, 5.0]]]), rotated)
    rotated_displacement = _ras_displacement_for_voxel_delta(
        rotated_original,
        torch.tensor([[[0.5, 0.0, 0.25]]]),
        rotated,
    )
    rotated_delta = ras_mm_to_voxel_dhw(rotated_original + rotated_displacement, rotated) - ras_mm_to_voxel_dhw(rotated_original, rotated)
    assert bool((rotated_delta == 0.0).any())
    torch.testing.assert_close(
        project_displacement_to_validity(rotated_original, rotated_displacement, rotated),
        rotated_displacement,
        atol=1e-6,
        rtol=0.0,
    )

    # H. Exactly 2 mm is retained by the radial bound for a nonzero vector.
    at_bound = torch.tensor([[[2.0, 0.0, 0.0]]])
    torch.testing.assert_close(bound_displacement_ras_mm(at_bound, 2.0), at_bound, atol=0.0, rtol=0.0)


def test_safe_projection_matches_pre_fix_formula_for_non_singular_cases() -> None:
    cases = (
        (_geometry(), torch.tensor([[[3.0, 4.0, 5.0]]]), torch.tensor([[[0.4, -0.3, 0.2]]])),
        (_geometry(), torch.tensor([[[5.8, 6.8, 7.8]]]), torch.tensor([[[0.8, 0.8, 0.8]]])),
        (_geometry(), torch.tensor([[[0.2, 0.2, 0.2]]]), torch.tensor([[[-0.8, -0.8, -0.8]]])),
        (_rotated_sheared_geometry(), torch.tensor([[[3.0, 4.0, 5.0]]]), torch.tensor([[[0.4, -0.3, 0.2]]])),
    )
    for geometry, original_voxel, displacement in cases:
        original = voxel_dhw_to_ras_mm(original_voxel, geometry)
        patched = project_displacement_to_validity(original, displacement, geometry)
        previous = _previous_non_singular_projection_formula(original, displacement, geometry)
        torch.testing.assert_close(patched, previous, atol=1e-6, rtol=1e-6)


def _assert_singular_refinement_backward_is_finite(case: str, device: torch.device) -> None:
    geometry = _rotated_sheared_geometry() if case == "rotated_sheared_zero_axis" else _geometry()
    original = voxel_dhw_to_ras_mm(
        torch.tensor([[[3.0, 4.0, 5.0]]], dtype=torch.float32, device=device),
        geometry,
    )
    if case == "zero_displacement":
        raw_value = torch.zeros_like(original)
        zero_axis = None
    elif case == "exact_zero_voxel_axis":
        raw_value = torch.tensor([[[0.5, 0.0, 1.0]]], dtype=torch.float32, device=device)
        zero_axis = 1
    elif case == "rotated_sheared_zero_axis":
        raw_value = torch.tensor([[[0.25, 0.25, 0.5]]], dtype=torch.float32, device=device)
        zero_axis = 1
    else:
        raise AssertionError(f"unknown refinement backward case: {case}")

    raw = raw_value.detach().requires_grad_(True)
    bounded = bound_displacement_ras_mm(raw, 2.0)
    delta_voxel_dhw = ras_mm_to_voxel_dhw(original + bounded, geometry) - ras_mm_to_voxel_dhw(original, geometry)
    if zero_axis is not None:
        assert bool((delta_voxel_dhw[..., zero_axis] == 0.0).all())
    refined, displacement = refine_points_ras_mm(original, raw, geometry, max_displacement_mm=2.0)
    (refined.square().mean() + displacement.square().mean()).backward()

    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())


@pytest.mark.parametrize(
    "case",
    ("zero_displacement", "exact_zero_voxel_axis", "rotated_sheared_zero_axis"),
)
def test_singular_refinement_backward_is_finite_on_cpu(case: str) -> None:
    _assert_singular_refinement_backward_is_finite(case, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the refinement backward regression")
@pytest.mark.parametrize(
    "case",
    ("zero_displacement", "exact_zero_voxel_axis", "rotated_sheared_zero_axis"),
)
def test_singular_refinement_backward_is_finite_on_cuda(case: str) -> None:
    _assert_singular_refinement_backward_is_finite(case, torch.device("cuda"))


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

    assert field.semantic_vectors.shape == (1, 2, 3)
    assert bool((field.semantic_vectors >= 0.0).all())
    assert torch.allclose(
        field.semantic_vectors.sum(dim=-1),
        torch.ones_like(field.semantic_vectors[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert bool((torch.linalg.vector_norm(field.displacement_ras_mm, dim=-1) <= config.max_displacement_mm + 1e-6).all())
    assert bool(ras_mm_in_bounds(field.refined_centers_ras_mm, geometry).all())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0.0)
        for parameter in refiner.offset_predictor.parameters()
    )
