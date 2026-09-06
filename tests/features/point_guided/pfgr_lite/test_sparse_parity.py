from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.decoder import ImplicitTriPlaneDecoder
from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
from smagm.features.point_guided.pfgr_lite.provenance import ProducerCompatibility
from smagm.features.point_guided.pfgr_lite.sparse_write import (
    build_footprint,
    footprint_plane_support,
    query_write_delta,
    reference_full_write,
)
from smagm.features.point_guided.pfgr_lite.types import ActionProposal
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes


def _fixture(dtype: torch.dtype, *, fallback: bool = False):
    output = VolumeGeometry(
        (4, 5, 6),
        (
            (1.3, 0.2, 0.1, 3.0),
            (0.1, 1.7, 0.2, -2.0),
            (0.0, 0.15, 2.1, 5.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    feature_volume = VolumeGeometry(
        (3, 4, 5),
        (
            (2.0, 0.1, 0.0, 3.3),
            (0.2, 2.4, 0.1, -1.2),
            (0.1, 0.2, 2.5, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    geometry = FeatureGridGeometry(
        output,
        feature_volume,
        "conv1_pre_maxpool",
        (1.5, 1.7, 1.4),
        (-0.2, 0.3, 0.1),
        ("synthetic-conv",),
    )
    lattice = PFGRQueryLattice.build(
        output,
        geometry,
        query_dtype=dtype,
        build_chunk_size=7,
        memory_bound_bytes=1 if fallback else 1 << 20,
    )
    generator = torch.Generator().manual_seed(18)
    depth, height, width = geometry.shape_dhw
    planes = DynamicTriPlanes(
        torch.randn((1, 32, height, width), dtype=dtype, generator=generator),
        torch.randn((1, 32, depth, width), dtype=dtype, generator=generator),
        torch.randn((1, 32, depth, height), dtype=dtype, generator=generator),
    )
    hashes = {
        name: "x" * 64
        for name in (
            "observation_normalization_hash",
            "geometry_query_version_hash",
            "medicalnet_provenance_hash",
            "frozen_bn_hash",
            "static_head_hash",
            "semantic_head_hash",
            "point_refiner_hash",
            "spectral_projector_hash",
            "state_initializer_hash",
            "updater_hash",
            "decoder_hash",
            "writer_hash",
            "candidate_geometry_hash",
            "label_definition_hash",
        )
    }
    producer = ProducerCompatibility(**hashes)
    state_digest = "state-" + "x" * 60
    point = torch.tensor((4.0, 2.0, 3.0), dtype=dtype)
    delta = torch.randn((96,), dtype=dtype, generator=generator)
    action = ActionProposal(
        context_id="ctx",
        context_version="pfgr-lite-types-v1",
        producer_compatibility_hash=producer.digest,
        state_version=0,
        state_digest=state_digest,
        point_id=0,
        point_ras_mm=point,
        o270=torch.zeros((270,), dtype=dtype),
        v126=torch.zeros((126,), dtype=dtype),
        delta=delta,
        legal=True,
        updater_version="u-v1",
        updater_producer_hash="u-hash",
        writer_version="w-v1",
        writer_hash="w-hash",
        query_version="pfgr-lite-query-lattice-v1",
        query_hash=lattice.geometry_hash,
        geometry_version="g-v1",
        geometry_hash=lattice.geometry_hash,
        point_version="p-v1",
        point_identity_hash="p-hash",
        action_id="action-0",
    )
    # ``ActionProposal`` only stores the state digest; parity uses planes
    # independently and does not require constructing PFGRState here.
    return output, geometry, lattice, planes, action


def _all_ids(shape: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = shape
    flat = torch.arange(depth * height * width, dtype=torch.long)
    d = flat // (height * width)
    rem = flat - d * height * width
    h = rem // width
    w = rem - h * width
    return torch.stack((d, h, w), dim=-1)


def test_sparse_delta_matches_independent_legacy_write_fp64() -> None:
    output, _, lattice, planes, action = _fixture(torch.float64)
    footprint = build_footprint(lattice, action, chunk_size=5)
    ids = _all_ids(output.shape_dhw)
    sparse = query_write_delta(lattice, footprint, ids, action.delta, chunk_size=3)
    written = reference_full_write(lattice, planes, action)
    expected = lattice.query(written, ids, chunk_size=4) - lattice.query(
        planes, ids, chunk_size=4
    )
    assert torch.allclose(sparse, expected, atol=1e-10, rtol=1e-9)


def test_sparse_delta_matches_independent_legacy_write_fp32() -> None:
    output, _, lattice, planes, action = _fixture(torch.float32)
    footprint = build_footprint(lattice, action, chunk_size=5)
    ids = _all_ids(output.shape_dhw)
    sparse = query_write_delta(lattice, footprint, ids, action.delta, chunk_size=3)
    written = reference_full_write(lattice, planes, action)
    expected = lattice.query(written, ids, chunk_size=4) - lattice.query(
        planes, ids, chunk_size=4
    )
    assert torch.allclose(sparse, expected, atol=1e-6, rtol=1e-5)


def test_fallback_support_is_exact_and_accounted() -> None:
    output, _, lattice, planes, action = _fixture(torch.float64, fallback=True)
    assert lattice.footprint_mode == "full_scan_fallback"
    footprint = build_footprint(lattice, action, chunk_size=3)
    assert footprint.mode == "full_scan_fallback"
    assert (
        footprint.scanned_voxel_count
        == 3 * torch.tensor(output.shape_dhw).prod().item()
    )
    assert lattice.memory_accounting["scan_calls"] == 1
    ids = footprint.voxel_ids_dhw
    sparse = query_write_delta(lattice, footprint, ids, action.delta, chunk_size=2)
    written = reference_full_write(lattice, planes, action)
    expected = lattice.query(written, ids, chunk_size=2) - lattice.query(
        planes, ids, chunk_size=2
    )
    assert torch.allclose(sparse, expected, atol=1e-10, rtol=1e-9)


def test_delta_query_preserves_gradient_without_full_plane_clone() -> None:
    _, _, lattice, planes, action = _fixture(torch.float64)
    footprint = build_footprint(lattice, action, chunk_size=5)
    ids = footprint.voxel_ids_dhw
    delta = action.delta.detach().clone().requires_grad_(True)
    response = query_write_delta(lattice, footprint, ids, delta, chunk_size=2)
    loss = response.square().sum()
    loss.backward()
    assert delta.grad is not None and bool(torch.isfinite(delta.grad).all())
    assert response.shape == (footprint.union_size, 96)
    # Sparse query-delta keeps state planes entirely out of the operation.
    assert planes.xy.grad_fn is None


@pytest.mark.parametrize("dtype", (torch.float64, torch.float32))
@pytest.mark.parametrize("fallback", (False, True))
def test_nonlinear_sparse_delta_gradient_matches_actual_writer(
    dtype: torch.dtype, fallback: bool
) -> None:
    """The frozen decoder sees the same differentiable correction as writeback."""

    output, _, lattice, planes, action = _fixture(dtype, fallback=fallback)
    footprint = build_footprint(lattice, action, chunk_size=3)
    ids = _all_ids(output.shape_dhw)
    decoder = ImplicitTriPlaneDecoder().to(dtype=dtype)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    delta = action.delta.detach().clone().requires_grad_(True)
    sparse_query = lattice.query(planes, ids, chunk_size=2)
    sparse_delta = query_write_delta(lattice, footprint, ids, delta, chunk_size=2)
    sparse_loss = decoder.mlp(sparse_query + sparse_delta).square().sum()
    sparse_gradient = torch.autograd.grad(sparse_loss, delta)[0]

    graded_action = ActionProposal(
        context_id=action.context_id,
        context_version=action.context_version,
        producer_compatibility_hash=action.producer_compatibility_hash,
        state_version=action.state_version,
        state_digest=action.state_digest,
        point_id=action.point_id,
        point_ras_mm=action.point_ras_mm,
        o270=action.o270,
        v126=action.v126,
        delta=delta,
        legal=action.legal,
        updater_version=action.updater_version,
        updater_producer_hash=action.updater_producer_hash,
        writer_version=action.writer_version,
        writer_hash=action.writer_hash,
        query_version=action.query_version,
        query_hash=action.query_hash,
        geometry_version=action.geometry_version,
        geometry_hash=action.geometry_hash,
        point_version=action.point_version,
        point_identity_hash=action.point_identity_hash,
        action_id=action.action_id,
    )
    written = reference_full_write(lattice, planes, graded_action)
    full_loss = decoder.mlp(lattice.query(written, ids, chunk_size=2)).square().sum()
    full_gradient = torch.autograd.grad(full_loss, delta)[0]
    tolerance = (1e-10, 1e-9) if dtype == torch.float64 else (1e-6, 1e-5)
    assert torch.allclose(sparse_loss, full_loss, atol=tolerance[0], rtol=tolerance[1])
    assert torch.allclose(
        sparse_gradient, full_gradient, atol=tolerance[0], rtol=tolerance[1]
    )


def test_support_coefficients_are_positive_and_plane_union_is_deduplicated() -> None:
    _, _, lattice, _, action = _fixture(torch.float64)
    footprint = build_footprint(lattice, action, chunk_size=5)
    assert footprint.union_size == footprint.voxel_ids_dhw.shape[0]
    assert footprint.multiplicity is not None
    assert bool((footprint.multiplicity >= 1).all())
    assert bool((footprint.multiplicity <= 3).all())
    for plane in ("xy", "xz", "yz"):
        voxels, coefficients = footprint_plane_support(footprint, plane)
        assert voxels.numel() == footprint.plane_counts[("xy", "xz", "yz").index(plane)]
        assert bool((coefficients > 0).all())


def test_query_delta_duplicate_and_empty_ids() -> None:
    _, _, lattice, _, action = _fixture(torch.float64)
    footprint = build_footprint(lattice, action, chunk_size=5)
    ids = footprint.voxel_ids_dhw
    duplicate = torch.cat((ids[:1], ids[:1]), dim=0)
    response = query_write_delta(
        lattice, footprint, duplicate, action.delta, chunk_size=1
    )
    assert torch.equal(response[0], response[1])
    empty = query_write_delta(
        lattice, footprint, torch.empty((0, 3), dtype=torch.long), action.delta
    )
    assert empty.shape == (0, 96)


def test_reference_decoder_parity_uses_shared_canonical_lattice() -> None:
    _, _, lattice, planes, action = _fixture(torch.float64)
    footprint = build_footprint(lattice, action, chunk_size=5)
    decoder = ImplicitTriPlaneDecoder().double()
    ids = footprint.voxel_ids_dhw
    before = decoder.mlp(lattice.query(planes, ids, chunk_size=3))
    delta_query = query_write_delta(lattice, footprint, ids, action.delta, chunk_size=3)
    sparse_after = decoder.mlp(lattice.query(planes, ids, chunk_size=3) + delta_query)
    full_after = decoder.mlp(
        lattice.query(reference_full_write(lattice, planes, action), ids, chunk_size=3)
    )
    assert before.shape == sparse_after.shape == full_after.shape
    assert torch.allclose(sparse_after, full_after, atol=1e-10, rtol=1e-9)
