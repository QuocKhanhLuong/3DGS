"""CPU checks for the locked sparse semantic-aware PoU formulas."""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from smagm.features.point_guided.contracts import (
    EmptySparseSupportError,
    PointField,
    PointGuidedGeometryError,
    VolumeGeometry,
)
from smagm.features.point_guided.pou import build_sparse_pou
from smagm.features.point_guided.sampling import voxel_dhw_to_ras_mm
from smagm.features.point_guided.semantic_affinity import semantic_affinity
from smagm.features.point_guided.spatial_affinity import spatial_affinity


def _point_field(
    centres_ras_mm: torch.Tensor,
    semantics: torch.Tensor,
) -> PointField:
    displacement = torch.zeros_like(centres_ras_mm)
    return PointField(
        original_centers_ras_mm=centres_ras_mm,
        refined_centers_ras_mm=centres_ras_mm,
        displacement_ras_mm=displacement,
        semantic_vectors=semantics,
        support_radius_mm=4.0,
    )


def _semantic_volume() -> torch.Tensor:
    volume = torch.zeros((1, 2, 3, 3, 3), dtype=torch.float32)
    volume[:, 0] = 1.0
    return volume


def _geometry() -> VolumeGeometry:
    return VolumeGeometry.from_spacing((3, 3, 3), spacing_xyz_mm=(1.0, 1.0, 1.0))


def _linear_voxels(edges, shape_dhw: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = shape_dhw
    return (
        edges.batch_indices * (depth * height * width)
        + edges.voxel_indices_dhw[:, 0] * (height * width)
        + edges.voxel_indices_dhw[:, 1] * width
        + edges.voxel_indices_dhw[:, 2]
    )


def test_locked_semantic_and_spatial_affinity_formulas_and_bounds() -> None:
    point = torch.tensor([1.0, 0.0, 0.0])
    voxel = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    assert torch.equal(semantic_affinity(point, voxel), torch.tensor([1.0, 0.5, 0.0]))

    distance = torch.tensor([0.0, 0.5, 0.999, 1.0, 1.5])
    expected = torch.tensor([1.0, 0.25, (1.0 - 0.999) ** 2, 0.0, 0.0])
    assert torch.allclose(spatial_affinity(distance, 1.0), expected)

    with pytest.raises(ValueError, match="sum to one"):
        semantic_affinity(torch.tensor([0.8, 0.8, 0.0]), voxel[:1])


def test_sparse_pou_normalizes_each_covered_voxel() -> None:
    field = _point_field(
        torch.tensor([[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    edges = build_sparse_pou(field, _semantic_volume(), _geometry())
    linear = _linear_voxels(edges, (3, 3, 3))
    for group in torch.unique(linear):
        assert torch.allclose(edges.normalized_weight[linear == group].sum(), torch.tensor(1.0))


def test_public_point_field_locks_supports_to_four_mm_and_pou_has_no_outside_contribution() -> None:
    with pytest.raises(ValueError, match="exactly 4.0 mm"):
        PointField(
            original_centers_ras_mm=torch.tensor([[[1.0, 1.0, 1.0]]]),
            refined_centers_ras_mm=torch.tensor([[[1.0, 1.0, 1.0]]]),
            displacement_ras_mm=torch.zeros((1, 1, 3)),
            semantic_vectors=torch.tensor([[[1.0, 0.0]]]),
            support_radius_mm=1.0,
        )
    geometry = VolumeGeometry.from_spacing((11, 11, 11))
    semantic_volume = torch.zeros((1, 2, 11, 11, 11), dtype=torch.float32)
    semantic_volume[:, 0] = 1.0
    field = _point_field(
        torch.tensor([[[5.0, 5.0, 5.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    edges = build_sparse_pou(field, semantic_volume, geometry)
    support_ras_mm = voxel_dhw_to_ras_mm(edges.voxel_indices_dhw.to(torch.float32), geometry)
    assert bool((torch.linalg.vector_norm(support_ras_mm - field.refined_centers_ras_mm[0, 0], dim=-1) < 4.0).all())


def test_public_point_field_rejects_displacement_above_two_mm() -> None:
    with pytest.raises(ValueError, match="at most 2.0 mm"):
        PointField(
            original_centers_ras_mm=torch.zeros((1, 1, 3)),
            refined_centers_ras_mm=torch.tensor([[[2.001, 0.0, 0.0]]]),
            displacement_ras_mm=torch.tensor([[[2.001, 0.0, 0.0]]]),
            semantic_vectors=torch.tensor([[[1.0, 0.0]]]),
            support_radius_mm=4.0,
        )


def test_sparse_pou_fails_with_typed_error_when_support_is_empty() -> None:
    field = _point_field(
        torch.tensor([[[1.0, 1.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    with pytest.raises(EmptySparseSupportError):
        build_sparse_pou(
            field,
            _semantic_volume(),
            _geometry(),
            valid_brain_mask=torch.zeros((1, 3, 3, 3), dtype=torch.bool),
        )


def test_sparse_pou_records_zero_semantic_denominator_as_sparse_unsupported_region() -> None:
    field = _point_field(
        torch.tensor([[[1.0, 1.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    mismatched_semantics = torch.zeros((1, 2, 3, 3, 3), dtype=torch.float32)
    mismatched_semantics[:, 1] = 1.0

    mask = torch.zeros((1, 3, 3, 3), dtype=torch.bool)
    mask[0, 1, 1, 1] = True
    with pytest.raises(EmptySparseSupportError, match="no positive compact-support") as captured:
        build_sparse_pou(field, mismatched_semantics, _geometry(), valid_brain_mask=mask)

    error = captured.value
    assert torch.equal(error.unsupported_batch_indices, torch.tensor([0]))
    assert torch.equal(error.unsupported_voxel_indices_dhw, torch.tensor([[1, 1, 1]]))


def test_sparse_pou_respects_optional_valid_brain_mask() -> None:
    field = _point_field(
        torch.tensor([[[1.0, 1.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    mask = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool)
    mask[0, 0, 1, 1, 2] = True
    edges = build_sparse_pou(field, _semantic_volume(), _geometry(), valid_brain_mask=mask)
    assert torch.equal(edges.voxel_indices_dhw, torch.tensor([[1, 1, 2]]))


def test_sparse_pou_source_and_runtime_guard_against_dense_point_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    import smagm.features.point_guided.pou as pou_module

    source = inspect.getsource(pou_module)
    assert "[B, N, D, H, W]" not in source
    assert ".expand(batch, points" not in source
    semantic_volume = _semantic_volume()

    original_zeros = torch.zeros

    def no_dense_point_volume(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        if len(shape) == 5:
            raise AssertionError("dense point-by-volume allocation is forbidden")
        return original_zeros(*size, **kwargs)

    monkeypatch.setattr(pou_module.torch, "zeros", no_dense_point_volume)
    field = _point_field(
        torch.tensor([[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    edges = build_sparse_pou(field, semantic_volume, _geometry())
    assert edges.raw_affinity.numel() > 0


def test_fine_spacing_uses_the_actual_spherical_support_for_the_local_cap() -> None:
    geometry = VolumeGeometry.from_spacing((17, 17, 17), spacing_xyz_mm=(0.5, 0.5, 0.5))
    semantic_volume = torch.zeros((1, 2, 17, 17, 17), dtype=torch.float32)
    semantic_volume[:, 0] = 1.0
    field = _point_field(
        torch.tensor([[[4.0, 4.0, 4.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )

    # The conservative 17^3 bounding cube has 4913 candidates, while the
    # strict 4-mm sphere contains fewer than the 4096 locked local-edge cap.
    edges = build_sparse_pou(field, semantic_volume, geometry, max_local_voxels_per_point=4096)
    assert 0 < edges.raw_affinity.numel() < 4096
    with pytest.raises(PointGuidedGeometryError, match="spherical PoU neighbourhood exceeds"):
        build_sparse_pou(field, semantic_volume, geometry, max_local_voxels_per_point=10)


def test_tiny_spacing_enumerates_bounded_chunks_before_failing_the_explicit_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import smagm.features.point_guided.pou as pou_module

    geometry = VolumeGeometry.from_spacing((81, 81, 81), spacing_xyz_mm=(0.1, 0.1, 0.1))
    semantic_volume = torch.zeros((1, 2, 81, 81, 81), dtype=torch.float32)
    semantic_volume[:, 0] = 1.0
    field = _point_field(
        torch.tensor([[[4.0, 4.0, 4.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    original_arange = torch.arange

    def bounded_arange(*args, **kwargs):
        values = original_arange(*args, **kwargs)
        assert values.numel() <= 4096
        return values

    monkeypatch.setattr(pou_module.torch, "arange", bounded_arange)
    with pytest.raises(PointGuidedGeometryError, match="spherical PoU neighbourhood exceeds"):
        build_sparse_pou(field, semantic_volume, geometry, max_local_voxels_per_point=4096)


def test_sparse_pou_contract_rejects_bad_relational_records() -> None:
    field = _point_field(
        torch.tensor([[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    edges = build_sparse_pou(field, _semantic_volume(), _geometry())

    with pytest.raises(ValueError, match="sum to one"):
        replace(edges, normalized_weight=edges.normalized_weight * 0.5)

    with pytest.raises(ValueError, match="disjoint"):
        replace(
            edges,
            unsupported_batch_indices=edges.batch_indices[:1],
            unsupported_voxel_indices_dhw=edges.voxel_indices_dhw[:1],
        )
