"""Independent exact support checks for the shared fallback traversal."""

from __future__ import annotations

import math

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite import sparse_write as sw
from smagm.features.point_guided.pfgr_lite.footprint import (
    PLANE_NAMES,
    PFGRQueryLattice,
)
from smagm.features.point_guided.pfgr_lite.types import ActionProposal
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes


def _geometry(shape=(9, 11, 13)):
    # Rotation, shear, anisotropy, translation, and a different feature affine.
    output = VolumeGeometry(
        shape,
        (
            (1.3, -0.2, 0.1, 3.0),
            (0.1, 1.7, 0.2, -2.0),
            (0.0, 0.15, 2.1, 5.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    feature = VolumeGeometry(
        tuple((s + 1) // 2 for s in shape),
        (
            (2.6, -0.3, 0.15, 3.3),
            (0.3, 3.4, 0.35, -1.2),
            (0.1, 0.2, 4.2, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    return output, FeatureGridGeometry(
        output,
        feature,
        "conv1_pre_maxpool",
        (2.0, 2.0, 2.0),
        (-0.2, 0.3, 0.1),
        ("synthetic-conv",),
    )


def _action(lattice, point=None):
    dtype = lattice.query_dtype
    if point is None:
        point = lattice.feature_geometry.feature_dhw_to_ras_mm(
            torch.tensor([2.2, 2.4, 3.1], dtype=dtype)
        )
    return ActionProposal(
        context_id="synthetic-ctx",
        context_version="pfgr-lite-types-v1",
        producer_compatibility_hash="x" * 64,
        state_version=0,
        state_digest="s" * 64,
        point_id=0,
        point_ras_mm=point,
        o270=torch.zeros(270, dtype=dtype),
        v126=torch.zeros(126, dtype=dtype),
        delta=torch.ones(96, dtype=dtype),
        legal=True,
        updater_version="u-v1",
        updater_producer_hash="u" * 64,
        writer_version="compact-writeback-4mm-v1",
        writer_hash="w" * 64,
        query_version=lattice.query_version,
        query_hash=lattice.geometry_hash,
        geometry_version="g-v1",
        geometry_hash=lattice.geometry_hash,
        point_version="p-v1",
        point_identity_hash="p" * 64,
        action_id="synthetic-action",
    )


def _all_ids(lattice):
    d, h, w = lattice.output_shape_dhw
    ids = torch.arange(d * h * w)
    return torch.stack((ids // (h * w), ids // w % h, ids % w), -1)


def _full_scan_reference(lattice, nodes, weights):
    # Independent dense node-vector gather (no searchsorted, chunk traversal,
    # output accumulation, or production response helper).
    stencils = lattice._chunk_stencils(_all_ids(lattice))
    results = []
    for plane, node, weight in zip(PLANE_NAMES, nodes, weights):
        stencil = stencils[plane]
        field = torch.zeros(stencil.rows * stencil.columns, dtype=lattice.dtype)
        field[node] = weight
        safe = torch.where(stencil.valid, stencil.indices, 0)
        coefficients = (field[safe] * stencil.weights * stencil.valid).sum(-1)
        positive = coefficients > 0
        results.append((positive.nonzero().flatten(), coefficients[positive]))
    return results


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk", [1, 113, 10000])
@pytest.mark.parametrize(
    "position", [(2.2, 2.4, 3.1), (0.0, 0.0, 0.0), (4.0, 5.0, 6.0)]
)
def test_shared_scan_matches_independent_full_scan_and_dense_writer(
    dtype, chunk, position
):
    output, feature = _geometry()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=dtype, build_chunk_size=41, memory_bound_bytes=1
    )
    action = _action(
        lattice, feature.feature_dhw_to_ras_mm(torch.tensor(position, dtype=dtype))
    )
    footprint = sw.build_footprint(lattice, action, chunk_size=chunk)
    nodes, weights = sw._positive_writer_nodes(lattice, action)
    expected = _full_scan_reference(lattice, nodes, weights)
    for plane, (ids, coefficients) in zip(PLANE_NAMES, expected):
        actual_ids, actual_coefficients = sw.footprint_plane_support(footprint, plane)
        assert torch.equal(actual_ids, ids)
        assert torch.equal(actual_coefficients, coefficients)
    assert footprint.scanned_voxel_count == 3 * math.prod(output.shape_dhw)
    ids = _all_ids(lattice)
    d, h, w = feature.shape_dhw
    zero = DynamicTriPlanes(
        *(torch.zeros(1, 32, a, b, dtype=dtype) for a, b in ((h, w), (d, w), (d, h)))
    )
    written = sw.reference_full_write(lattice, zero, action)
    dense = lattice.query(written, ids, chunk_size=79)
    sparse = sw.query_write_delta(lattice, footprint, ids, action.delta, chunk_size=73)
    assert torch.equal((dense > 0).any(-1), (sparse > 0).any(-1))
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(sparse, dense, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_sub_ulp_boundary_coefficients_are_not_pruned(dtype, direction):
    # Translation places queries one representable step from integer nodes.
    one = torch.tensor(1.0, dtype=dtype)
    adjacent = torch.nextafter(one, torch.tensor(direction * float("inf"), dtype=dtype))
    shift = float(adjacent - one)
    output = VolumeGeometry.from_spacing((3, 4, 5), (1.0, 1.0, 1.0))
    feature = FeatureGridGeometry(
        output,
        VolumeGeometry.from_spacing((3, 4, 5), (1.0, 1.0, 1.0), (shift, shift, shift)),
        "conv1_pre_maxpool",
        (1.0, 1.0, 1.0),
        (shift, shift, shift),
        ("synthetic",),
    )
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=dtype, build_chunk_size=7, memory_bound_bytes=1
    )
    # Single sparse node isolates the tiny positive bilinear coefficients.
    nodes = tuple(torch.tensor([0]) for _ in PLANE_NAMES)
    weights = tuple(torch.tensor([1.0], dtype=dtype) for _ in PLANE_NAMES)
    actual = sw._scan_plane_responses(lattice, nodes, weights, chunk_size=11)
    expected = _full_scan_reference(lattice, nodes, weights)
    for index, (ids, coefficients) in enumerate(expected):
        assert torch.equal(actual[0][index], ids)
        assert torch.equal(actual[1][index], coefficients)
    coordinates = lattice._feature_coordinates(
        _all_ids(lattice), output_geometry=output, feature_geometry=feature, dtype=dtype
    )
    for index, (name, row_axis, col_axis) in enumerate(
        (("xy", 1, 2), ("xz", 0, 2), ("yz", 0, 1))
    ):
        stencil = lattice._chunk_stencils(_all_ids(lattice))[name]
        field = torch.zeros(1, 1, stencil.rows, stencil.columns, dtype=dtype)
        field[0, 0, 0, 0] = 1
        dense = lattice._query_plane_grid(
            field, coordinates, row_axis=row_axis, column_axis=col_axis
        )[:, 0]
        assert torch.equal((dense > 0).nonzero().flatten(), actual[0][index])
        torch.testing.assert_close(
            dense[dense > 0], actual[1][index], atol=2 * torch.finfo(dtype).eps, rtol=0
        )
    if direction > 0:
        assert any(
            bool(((v > 0) & (v < torch.finfo(dtype).eps)).any()) for v in actual[1]
        )


def test_one_geometry_pass_per_chunk_and_empty_planes_are_counted(monkeypatch):
    output, feature = _geometry()
    lattice = PFGRQueryLattice.build(
        output,
        feature,
        query_dtype=torch.float32,
        build_chunk_size=41,
        memory_bound_bytes=1,
    )
    original = PFGRQueryLattice._chunk_stencils
    calls = []

    def spy(self, ids):
        calls.append(ids.shape[0])
        return original(self, ids)

    monkeypatch.setattr(PFGRQueryLattice, "_chunk_stencils", spy)
    nodes = (torch.tensor([1]), torch.empty(0, dtype=torch.long), torch.tensor([3]))
    weights = (torch.ones(1), torch.empty(0), torch.ones(1))
    result = sw._scan_plane_responses(lattice, nodes, weights, chunk_size=113)
    assert sum(calls) == math.prod(output.shape_dhw)
    assert len(calls) == math.ceil(math.prod(output.shape_dhw) / 113)
    assert max(calls) <= 113
    assert result[2] == 2 * math.prod(output.shape_dhw)
    assert result[0][1].numel() == result[1][1].numel() == 0
    assert not lattice._plane_stencils and not lattice._inverse_indices
    before = len(calls)
    empty = sw._scan_plane_responses(
        lattice, (nodes[1],) * 3, (weights[1],) * 3, chunk_size=113
    )
    assert empty[2:] == (0, 0, 0)
    assert len(calls) == before


@pytest.mark.parametrize("mutation", ["stencil", "inverse", "raw_bytes", "geometry"])
def test_build_rejects_mutated_lattice_before_response_and_cache_rebuilds(mutation):
    PFGRQueryLattice.clear_cache()
    output, feature = _geometry()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=41
    )
    action = _action(lattice)
    if mutation == "stencil":
        lattice._plane_stencils["xy"].weights[0, 0] += 0.125
    elif mutation == "inverse":
        lattice._inverse_indices["xy"].weights[0] += 0.125
    elif mutation == "raw_bytes":
        lattice._plane_stencils["xy"].weights.numpy()[0, 0] += 0.125
    else:
        object.__setattr__(lattice.feature_geometry, "operator_chain", ("mutated",))
    with pytest.raises(RuntimeError, match="mutation|identity"):
        sw.build_footprint(lattice, action, chunk_size=17)
    # A fresh build of the original immutable geometry must not return stale data.
    output, feature = _geometry()
    rebuilt = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=41
    )
    assert rebuilt is not lattice
    rebuilt.validate_integrity()
    PFGRQueryLattice.clear_cache()


def test_stale_and_mutated_footprint_metadata_is_rejected():
    output, feature = _geometry()
    lattice = PFGRQueryLattice.build(
        output,
        feature,
        query_dtype=torch.float32,
        build_chunk_size=41,
        memory_bound_bytes=1,
    )
    action = _action(lattice)
    footprint = sw.build_footprint(lattice, action, chunk_size=113)
    footprint._pfgr_plane_coefficients[0][0] += 0.1
    with pytest.raises(RuntimeError, match="accounting metadata mutation"):
        sw.query_write_delta(lattice, footprint, _all_ids(lattice), action.delta)
    footprint = sw.build_footprint(lattice, action, chunk_size=113)
    object.__setattr__(footprint, "_pfgr_geometry_hash", "stale")
    with pytest.raises(ValueError, match="identity is stale"):
        sw.query_write_delta(lattice, footprint, _all_ids(lattice), action.delta)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_indexed_and_shared_fallback_support_agree(dtype):
    output, feature = _geometry()
    indexed = PFGRQueryLattice.build(
        output, feature, query_dtype=dtype, build_chunk_size=41
    )
    fallback = PFGRQueryLattice.build(
        output, feature, query_dtype=dtype, build_chunk_size=41, memory_bound_bytes=1
    )
    action = _action(indexed)
    left = sw.build_footprint(indexed, action, chunk_size=73)
    right = sw.build_footprint(fallback, action, chunk_size=37)
    assert torch.equal(left.voxel_ids_dhw, right.voxel_ids_dhw)
    assert torch.equal(left.multiplicity, right.multiplicity)
    assert left.plane_counts == right.plane_counts
    for plane in PLANE_NAMES:
        x_ids, x = sw.footprint_plane_support(left, plane)
        y_ids, y = sw.footprint_plane_support(right, plane)
        assert torch.equal(x_ids, y_ids)
        # CSR gathers sum node edges in a different order from the four slots.
        torch.testing.assert_close(x, y, atol=2 * torch.finfo(dtype).eps, rtol=0)
