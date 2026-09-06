from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
from smagm.features.point_guided.reward import DynamicStatePointQuery
from smagm.features.point_guided.sampling import voxel_dhw_to_ras_mm
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes


def _geometries() -> tuple[VolumeGeometry, FeatureGridGeometry]:
    output = VolumeGeometry(
        (4, 5, 6),
        (
            (1.3, 0.2, 0.1, 3.0),
            (0.1, 1.7, 0.2, -2.0),
            (0.0, 0.15, 2.1, 5.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    feature = VolumeGeometry(
        (3, 4, 5),
        (
            (2.0, 0.1, 0.0, 3.3),
            (0.2, 2.4, 0.1, -1.2),
            (0.1, 0.2, 2.5, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    return output, FeatureGridGeometry(
        output,
        feature,
        "conv1_pre_maxpool",
        (1.5, 1.7, 1.4),
        (-0.2, 0.3, 0.1),
        ("synthetic-conv",),
    )


def _planes(
    feature_geometry: FeatureGridGeometry, *, dtype: torch.dtype = torch.float64
) -> DynamicTriPlanes:
    depth, height, width = feature_geometry.shape_dhw
    generator = torch.Generator().manual_seed(91)
    return DynamicTriPlanes(
        torch.randn((1, 32, height, width), dtype=dtype, generator=generator),
        torch.randn((1, 32, depth, width), dtype=dtype, generator=generator),
        torch.randn((1, 32, depth, height), dtype=dtype, generator=generator),
    )


def _ids(output_geometry: VolumeGeometry) -> torch.Tensor:
    depth, height, width = output_geometry.shape_dhw
    return torch.tensor(
        [
            (0, 0, 0),
            (0, height - 1, width - 1),
            (depth - 1, 0, width - 1),
            (depth - 1, height - 1, 0),
            (1, 2, 3),
            (2, 1, 4),
        ],
        dtype=torch.long,
    )


def _legacy_query(
    planes: DynamicTriPlanes, ids: torch.Tensor, feature_geometry: FeatureGridGeometry
) -> torch.Tensor:
    output_geometry = feature_geometry.source_geometry
    points = voxel_dhw_to_ras_mm(
        ids.to(dtype=planes.xy.dtype).unsqueeze(0), output_geometry
    )
    return DynamicStatePointQuery()(planes, points, feature_geometry).packed[0]


def test_canonical_query_matches_independent_legacy_query_on_full_affine() -> None:
    output, feature = _geometries()
    planes = _planes(feature)
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    ids = _ids(output)
    actual = lattice.query(planes, ids, chunk_size=2)
    expected = _legacy_query(planes, ids, feature)
    assert actual.shape == (ids.shape[0], 96)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_stencils_retain_four_unclipped_neighbours_and_zero_flags() -> None:
    output, feature = _geometries()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=5
    )
    assert set(lattice.stencils) == {"xy", "xz", "yz"}
    for stencil in lattice.stencils.values():
        assert stencil.neighbour_indices.shape == (
            torch.tensor(output.shape_dhw).prod().item(),
            4,
        )
        assert stencil.valid.dtype == torch.bool
        assert stencil.weights.dtype == torch.float64
        # The translated/sheared fixture crosses the feature boundary.  At
        # least one row therefore has explicit invalid zero-padding slots.
        assert bool((~stencil.valid).any())
        assert not bool((stencil.neighbour_indices[~stencil.valid] == 0).all())


def test_inverse_index_matches_enumerated_positive_stencil_support() -> None:
    output, feature = _geometries()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=4
    )
    voxel_count = int(torch.tensor(output.shape_dhw).prod().item())
    for name, stencil in lattice.stencils.items():
        assert lattice.node_to_voxel(name) is not None
        inverse = lattice.node_to_voxel(name)
        assert inverse is not None
        enumerated: list[torch.Tensor] = []
        for slot in range(4):
            positive = stencil.valid[:, slot] & (stencil.weights[:, slot] > 0)
            enumerated.append(torch.arange(voxel_count, dtype=torch.long)[positive])
        expected = torch.unique(torch.cat(enumerated), sorted=True)
        assert torch.equal(lattice.positive_support_linear(name), expected)
        assert torch.equal(
            lattice.support_voxel_ids(name)[:, 0]
            * output.shape_dhw[1]
            * output.shape_dhw[2]
            + lattice.support_voxel_ids(name)[:, 1] * output.shape_dhw[2]
            + lattice.support_voxel_ids(name)[:, 2],
            expected,
        )
        assert inverse.edge_count == sum(part.numel() for part in enumerated)


def test_fallback_is_exact_and_explicitly_accounted() -> None:
    output, feature = _geometries()
    indexed = PFGRQueryLattice.build(
        output,
        feature,
        query_dtype=torch.float32,
        build_chunk_size=3,
        memory_bound_bytes=1 << 20,
    )
    fallback = PFGRQueryLattice.build(
        output,
        feature,
        query_dtype=torch.float32,
        build_chunk_size=3,
        memory_bound_bytes=1,
    )
    planes = _planes(feature, dtype=torch.float32)
    ids = _ids(output)
    assert fallback.footprint_mode == "full_scan_fallback"
    assert fallback.inverse_indices == {}
    # Build itself performs no fallback scan; counters increase only when the
    # support enumeration below actually visits all three plane rows.
    assert fallback.memory_accounting["scanned_voxel_count"] == 0
    assert torch.allclose(
        fallback.query(planes, ids, chunk_size=2),
        indexed.query(planes, ids, chunk_size=5),
        atol=1e-6,
        rtol=1e-5,
    )
    assert fallback.memory_accounting["query_stencil_voxel_count"] == 3 * ids.shape[0]
    assert torch.equal(fallback.support_voxel_ids(), indexed.support_voxel_ids())
    assert fallback.memory_accounting["scanned_voxel_count"] == 3 * int(
        torch.tensor(output.shape_dhw).prod().item()
    )
    assert fallback.memory_accounting["scan_calls"] == 1
    scanned_once = fallback.memory_accounting["scanned_bytes"]
    assert torch.equal(fallback.support_voxel_ids(), indexed.support_voxel_ids())
    assert fallback.memory_accounting["scan_calls"] == 2
    assert fallback.memory_accounting["scanned_bytes"] > scanned_once
    assert fallback.memory_accounting["would_materialized_peak_bytes"] > 1


def test_duplicate_ids_empty_queries_and_membership() -> None:
    output, feature = _geometries()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    planes = _planes(feature)
    ids = _ids(output)
    duplicate = torch.cat((ids[:2], ids[:1]), dim=0)
    queried = lattice.query(planes, duplicate, chunk_size=1)
    assert torch.equal(queried[0], queried[2])
    empty = lattice.query(planes, torch.empty((0, 3), dtype=torch.long), chunk_size=1)
    assert empty.shape == (0, 96)
    membership = lattice.positive_support_membership(duplicate)
    assert membership.shape == (3,)
    assert torch.equal(membership[:2], lattice.positive_support_membership(ids[:2]))


def test_gradient_matches_legacy_query() -> None:
    output, feature = _geometries()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    ids = _ids(output)
    base = _planes(feature)
    left = DynamicTriPlanes(
        *(
            value.detach().clone().requires_grad_(True)
            for value in (base.xy, base.xz, base.yz)
        )
    )
    right = DynamicTriPlanes(
        *(
            value.detach().clone().requires_grad_(True)
            for value in (base.xy, base.xz, base.yz)
        )
    )
    canonical_loss = lattice.query(left, ids, chunk_size=2).square().sum()
    legacy_loss = _legacy_query(right, ids, feature).square().sum()
    canonical_loss.backward()
    legacy_loss.backward()
    for canonical, legacy in zip(
        (left.xy, left.xz, left.yz), (right.xy, right.xz, right.yz)
    ):
        assert canonical.grad is not None and legacy.grad is not None
        assert torch.allclose(canonical.grad, legacy.grad, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize(
    ("output_shape", "feature_shape"),
    [((2, 3, 4), (2, 2, 3)), ((1, 2, 3), (2, 3, 2))],
)
def test_shape_and_chunk_variants_match_legacy_query(
    output_shape: tuple[int, int, int], feature_shape: tuple[int, int, int]
) -> None:
    output = VolumeGeometry.from_spacing(
        output_shape, spacing_xyz_mm=(1.3, 2.1, 0.8), origin_ras_mm=(4.0, -2.0, 1.0)
    )
    feature = VolumeGeometry(
        feature_shape,
        (
            (1.7, 0.2, 0.1, 3.2),
            (0.1, 2.3, 0.2, -1.1),
            (0.0, 0.15, 1.9, 2.2),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    feature_geometry = FeatureGridGeometry(
        output,
        feature,
        "conv1_pre_maxpool",
        (1.3, 1.7, 1.2),
        (0.1, -0.3, 0.2),
        ("conv", "pool"),
    )
    planes = _planes(feature_geometry, dtype=torch.float32)
    depth, height, width = output_shape
    flat = torch.arange(depth * height * width, dtype=torch.long)
    ids = torch.stack(
        (flat // (height * width), (flat % (height * width)) // width, flat % width),
        dim=-1,
    )
    lattice = PFGRQueryLattice.build(
        output, feature_geometry, query_dtype=torch.float32, build_chunk_size=2
    )
    canonical = lattice.query(planes, ids, chunk_size=1)
    legacy = _legacy_query(planes, ids, feature_geometry)
    assert torch.allclose(canonical, legacy, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        canonical,
        lattice.query(planes, ids, chunk_size=ids.shape[0]),
        atol=1e-6,
        rtol=1e-5,
    )


def test_query_validation_enforces_batch_dtype_shape_and_integer_ids() -> None:
    output, feature = _geometries()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float32, build_chunk_size=3
    )
    planes = _planes(feature, dtype=torch.float32)
    with pytest.raises(ValueError, match="batch size exactly one"):
        lattice.query(
            DynamicTriPlanes(
                planes.xy.expand(2, -1, -1, -1),
                planes.xz.expand(2, -1, -1, -1),
                planes.yz.expand(2, -1, -1, -1),
            ),
            _ids(output),
            chunk_size=2,
        )
    with pytest.raises(TypeError, match="dtype"):
        lattice.query(_planes(feature, dtype=torch.float64), _ids(output), chunk_size=2)
    with pytest.raises(TypeError, match="torch.long"):
        lattice.query(planes, _ids(output).to(torch.int32), chunk_size=2)
    with pytest.raises(ValueError, match="within output"):
        lattice.query(
            planes,
            torch.tensor([[0, 0, output.shape_dhw[2]]], dtype=torch.long),
            chunk_size=2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        lattice.query(planes, _ids(output), chunk_size=0)


def test_cache_identity_and_stale_tensor_rejection() -> None:
    PFGRQueryLattice.clear_cache()
    output, feature = _geometries()
    first = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    second = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    assert first is second
    # Deliberately mutate the owned private tensor; public ``stencils``
    # inspection returns defensive clones.
    first._plane_stencils["xy"].weights[0, 0] += 0.125
    with pytest.raises(RuntimeError, match="mutation detected"):
        first.validate_integrity()
    rebuilt = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    assert rebuilt is not first
    PFGRQueryLattice.clear_cache()


def test_query_hot_path_uses_fast_versions_and_no_full_plane_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, feature = _geometries()
    lattice = PFGRQueryLattice.build(
        output, feature, query_dtype=torch.float64, build_chunk_size=3
    )
    planes = _planes(feature)
    ids = _ids(output)
    original_cat = torch.cat
    cat_shapes: list[tuple[tuple[int, ...], ...]] = []
    full_before = lattice.memory_accounting["validation_full_calls"]

    def spy_cat(values: object, *args: object, **kwargs: object) -> torch.Tensor:
        tensors = tuple(values)  # type: ignore[arg-type]
        cat_shapes.append(tuple(tuple(item.shape) for item in tensors))
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", spy_cat)
    lattice.query(planes, ids, chunk_size=2)
    # The only query concatenation packs the three bounded [chunk,32] plane
    # results; no [32, plane_nodes] full-plane copy is made.
    assert cat_shapes
    assert all(
        all(shape[0] <= 2 and shape[1] in (32, 96) for shape in shapes)
        for shapes in cat_shapes
    )
    assert lattice.memory_accounting["validation_fast_calls"] >= 1
    assert lattice.memory_accounting["validation_full_calls"] == full_before


def test_process_cache_has_bounded_entries_and_explicit_eviction() -> None:
    PFGRQueryLattice.clear_cache()
    output, feature = _geometries()
    for chunk_size in range(1, 12):
        PFGRQueryLattice.build(
            output,
            feature,
            query_dtype=torch.float32,
            build_chunk_size=chunk_size,
            memory_bound_bytes=1,
        )
    stats = PFGRQueryLattice.cache_stats()
    assert stats["entries"] <= stats["max_entries"]
    assert stats["retained_bytes"] <= stats["max_bytes"]
    assert stats["evictions"] >= 1
    assert not hasattr(
        PFGRQueryLattice.build(
            output,
            feature,
            query_dtype=torch.float32,
            build_chunk_size=2,
            memory_bound_bytes=1,
        ),
        "_support_cache",
    )
    released = PFGRQueryLattice.build(
        output,
        feature,
        query_dtype=torch.float64,
        build_chunk_size=2,
        memory_bound_bytes=1,
    )
    assert released.release() is True
    assert released.release() is False
    PFGRQueryLattice.clear_cache()
